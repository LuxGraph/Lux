/* Copyright 2018 Stanford, UT Austin, LANL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include "../graph.h"
#include "../lux_mapper.h"
#include "legion.h"
#include <string.h>

LegionRuntime::Logger::Category log_cc("cc");
LegionRuntime::Logger::Category log_lux("lux");

void parse_input_args(char **argv, int argc, int &num_gpu,
                      std::string &file_name);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int numGPU = 0;
  std::string filename;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, numGPU, filename);
    log_cc.print("CC settings: numPartitions(%d) filename(%s)\n",
                 numGPU, filename.c_str());
    if (numGPU <= 0) {
      fprintf(stderr, "numGPU(%d) must be greater than zero.\n",
              numGPU);
      return;
    }
  }

  Graph graph(ctx, runtime, numGPU, filename);

  ArgumentMap local_args;
  // Init phase
  Rect<1> task_rect(0, graph.numParts - 1);
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);
  PushLoadTask load_task(graph, task_is, local_args, filename);
  FutureMap fm = runtime->execute_index_space(ctx, load_task);
  fm.wait_all_results();

  PushInitTask init_task(graph, task_is, local_args);
  fm = runtime->execute_index_space(ctx, init_task);
  fm.wait_all_results();
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    GraphPiece piece = fm.get_result<GraphPiece>(*it);
    local_args.set_point(*it, TaskArgument(&piece, sizeof(GraphPiece)));
  }

  // CC phase
  int iteration = 0;
  log_cc.print("Start Connected Components computation...");
  double ts_start = Realm::Clock::current_time_in_microseconds();
  while (true) {
    PushAppTask app_task(graph, task_is, local_args, iteration);
    fm = runtime->execute_index_space(ctx, app_task);
    //fm.wait_all_results();
    bool halt = true;
    //for (PointInRectIterator<1> it(task_rect); it(); it++) {
    //  V_ID numNodes = fm.get_result<V_ID>(*it);
    //  if (numNodes > 0) halt = false;
    //}
    if (iteration > 20) break;
    iteration ++;
  }
  fm.wait_all_results();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double sim_time = 1e-6 * (ts_end - ts_start);
  log_cc.print("Finish Connected Components computation...");
  printf("ELAPSED TIME = %7.7f s\n", sim_time);
}

static void update_mappers(Machine machine, Runtime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new LuxMapper(machine, rt, *it), *it);
  }
}

void parse_input_args(char **argv, int argc,
                      int &numGPU, std::string &filename)
{
  for (int i = 1; i < argc; i++) 
  {
    if (!strcmp(argv[i], "-ng")) 
    {
      numGPU = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-file"))
    {
      filename = std::string(argv[++i]);
      continue;
    }
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(PUSH_LOAD_TASK_ID, "load_task (push)");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<push_load_task_impl>(registrar, "load_task (push)");
  }
  {
    TaskVariantRegistrar registrar(PUSH_INIT_TASK_ID, "init_task (push)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<GraphPiece, push_init_task_impl>(
        registrar, "init_task (push)");
  }
  {
    TaskVariantRegistrar registrar(PUSH_APP_TASK_ID, "app_task (push)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<V_ID, push_app_task_impl>(
        registrar, "app_task (push)");
  }
  Runtime::add_registration_callback(update_mappers);

  return Runtime::start(argc, argv);
}

template<typename T>
void alloc_fs(Context ctx, Runtime *runtime, FieldSpace fs)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(T), FID_DATA);
}

PushLoadTask::PushLoadTask(const Graph &graph,
                           const IndexSpaceT<1> &domain,
                           const ArgumentMap &arg_map,
                           std::string &fn)
  : IndexLauncher(PUSH_LOAD_TASK_ID, domain,
                  TaskArgument(fn.c_str(), MAX_FILE_LEN), arg_map)
{
  // regions[0]: raw_rows
  {
    RegionRequirement rr(graph.raw_row_lp, 0/*projection id*/,
                         WRITE_ONLY, EXCLUSIVE, graph.raw_row_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: raw_cols
  {
    RegionRequirement rr(graph.raw_col_lp, 0/*projection id*/,
                         WRITE_ONLY, EXCLUSIVE, graph.raw_col_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

void push_load_task_impl(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  char* file_name = (char*) task->args;
  const AccessorWO<E_ID, 1> acc_raw_rows(regions[0], FID_DATA);
  const AccessorWO<V_ID, 1> acc_raw_cols(regions[1], FID_DATA);
  Rect<1> rect_raw_rows = runtime->get_index_space_domain(
                              ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_raw_cols = runtime->get_index_space_domain(
                              ctx, task->regions[1].region.get_index_space());
  V_ID rowLeft = rect_raw_rows.lo[0], rowRight = rect_raw_rows.hi[0];
  E_ID colLeft = rect_raw_cols.lo[0], colRight = rect_raw_cols.hi[0];
  assert(acc_raw_rows.accessor.is_dense_arbitrary(rect_raw_rows));
  assert(acc_raw_cols.accessor.is_dense_arbitrary(rect_raw_cols));
  E_ID* raw_rows = acc_raw_rows.ptr(rect_raw_rows.lo);
  V_ID* raw_cols = acc_raw_cols.ptr(rect_raw_cols.lo);
  // Load row pointers and col indices
  log_lux.print("Load task: file(%s) colLeft(%zu) colRight(%zu)",
                file_name, colLeft, colRight);
  FILE* fd = fopen(file_name, "rb");
  assert(fd != NULL);
  int fseek_ret;
  size_t fread_ret;
  V_ID nv;
  E_ID ne;
  assert(fread(&nv, sizeof(V_ID), 1, fd) == 1);
  assert(fread(&ne, sizeof(E_ID), 1, fd) == 1);
  fseek_ret =
    fseeko(fd, FILE_HEADER_SIZE + sizeof(E_ID) * (size_t)rowLeft, SEEK_SET);
  assert(fseek_ret == 0);
  fread_ret =
    fread(raw_rows, sizeof(E_ID), (size_t)(rowRight - rowLeft + 1), fd);
  assert(fread_ret == rowRight - rowLeft + 1);
  fseek_ret =
    fseeko(fd, FILE_HEADER_SIZE + sizeof(E_ID) * (size_t)nv
               + sizeof(V_ID) * (size_t)colLeft, SEEK_SET);
  fread_ret =
    fread(raw_cols, sizeof(V_ID), (size_t)(colRight - colLeft + 1), fd);
  assert(fread_ret == colRight - colLeft + 1);
  fclose(fd);
}

PushInitTask::PushInitTask(const Graph &graph,
                           const IndexSpaceT<1> &domain,
                           const ArgumentMap &arg_map)
  : IndexLauncher(PUSH_INIT_TASK_ID, domain,
        TaskArgument(&graph, sizeof(Graph)), arg_map)
{
  // regions[0]: pull_row_ptrs
  {
    RegionRequirement rr(graph.pull_row_ptr_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.pull_row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: pull_col_idxs
  {
    RegionRequirement rr(graph.pull_col_idx_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.pull_col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: push_row_ptrs
  {
    RegionRequirement rr(graph.push_row_ptr_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.push_row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: push_col_idxs
  {
    RegionRequirement rr(graph.push_col_idx_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.push_col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[4]: new_fq
  {
    RegionRequirement rr(graph.frontier_lp[0], 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.frontier_lr[0],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[5]: new_pr
  {
    RegionRequirement rr(graph.dist_lp[0], 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.dist_lr[0],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[6]: raw_rows
  {
    RegionRequirement rr(graph.raw_row_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.raw_row_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[7]: raw_cols
  {
    RegionRequirement rr(graph.raw_col_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.raw_col_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

PushAppTask::PushAppTask(const Graph &graph,
                         const IndexSpaceT<1> &domain,
                         const ArgumentMap &arg_map,
                         int iter)
  : IndexLauncher(PUSH_APP_TASK_ID, domain,
                  TaskArgument(&graph, sizeof(Graph)), arg_map)
{
  // regions[0]: pull_row_ptrs
  {
    RegionRequirement rr(graph.pull_row_ptr_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.pull_row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: pull_col_idxs
  {
    RegionRequirement rr(graph.pull_col_idx_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.pull_col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: push_row_ptrs
  {
    RegionRequirement rr(graph.push_row_ptr_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.push_row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: push_col_idxs
  {
    RegionRequirement rr(graph.push_col_idx_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.push_col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[4]: old_fq
  {
    RegionRequirement rr(graph.frontier_lr[iter%2], 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.frontier_lr[iter%2],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[5]: new_fq
  {
    RegionRequirement rr(graph.frontier_lp[(iter+1)%2], 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.frontier_lr[(iter+1)%2],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[6]: old_pr
  {
    RegionRequirement rr(graph.dist_lr[iter%2], 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.dist_lr[iter%2],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[7]: new_pr
  {
    RegionRequirement rr(graph.dist_lp[(iter+1)%2], 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.dist_lr[(iter+1)%2],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

Graph::Graph(Context ctx, HighLevelRuntime *runtime,
             int _numParts, std::string& file_name)
: numParts(_numParts)
{
  if (numParts > MAX_NUM_PARTS) {
    fprintf(stderr, "ERROR: change MAX_NUM_PARTS to at least %d\n", numParts);
    assert(false);
  }
  //{
    FILE* fd = fopen(file_name.c_str(), "rb");
    assert(fd != NULL);
    size_t fread_ret = fread(&nv, sizeof(V_ID), 1, fd);
    assert(fread_ret == 1);
    fread_ret = fread(&ne, sizeof(E_ID), 1, fd);
    assert(fread_ret == 1);
    log_lux.print("Load graph: numNodes(%u) numEdges(%zu)\n", nv, ne);
    Rect<1> vtx_rect(Point<1>(0), Point<1>(nv - 1));
    IndexSpaceT<1> vtx_is =
      runtime->create_index_space(ctx, vtx_rect);
    runtime->attach_name(vtx_is, "vertices_index_space");
    Rect<1> row_rect(Point<1>(0), Point<1>(nv*numParts-1));
    IndexSpaceT<1> row_is =
      runtime->create_index_space(ctx, row_rect);
    runtime->attach_name(row_is, "row_index_space");
    Rect<1> edge_rect(Point<1>(0), Point<1>(ne - 1));
    IndexSpaceT<1> edge_is =
      runtime->create_index_space(ctx, edge_rect);
    runtime->attach_name(edge_is, "edges_index_space");

    FieldSpace row_ptr_fs = runtime->create_field_space(ctx);
    runtime->attach_name(row_ptr_fs, "row_ptrs(NodeStruct)");
    FieldSpace raw_row_fs = runtime->create_field_space(ctx);
    runtime->attach_name(raw_row_fs, "raw_rows(E_ID)");
    FieldSpace push_col_idx_fs = runtime->create_field_space(ctx);
    runtime->attach_name(push_col_idx_fs, "col_idxs(EdgeStruct)");
    FieldSpace pull_col_idx_fs = runtime->create_field_space(ctx);
    runtime->attach_name(pull_col_idx_fs, "col_idxs(EdgeStruct2)");
    FieldSpace raw_col_fs = runtime->create_field_space(ctx);
    runtime->attach_name(raw_col_fs, "raw_cols(V_ID)");
    FieldSpace frontier_fs = runtime->create_field_space(ctx);
    runtime->attach_name(frontier_fs, "frontier(char)");
    FieldSpace dist_fs = runtime->create_field_space(ctx);
    runtime->attach_name(dist_fs, "out_field_space");

    // Allocate fields
    alloc_fs<NodeStruct>(ctx, runtime, row_ptr_fs);
    alloc_fs<E_ID>(ctx, runtime, raw_row_fs);
    alloc_fs<EdgeStruct>(ctx, runtime, push_col_idx_fs);
    alloc_fs<EdgeStruct2>(ctx, runtime, pull_col_idx_fs);
    alloc_fs<V_ID>(ctx, runtime, raw_col_fs);
    alloc_fs<char>(ctx, runtime, frontier_fs);
    alloc_fs<Vertex>(ctx, runtime, dist_fs);

    // Make logical regions
    push_row_ptr_lr = runtime->create_logical_region(ctx, row_is, row_ptr_fs);
    pull_row_ptr_lr = runtime->create_logical_region(ctx, vtx_is, row_ptr_fs);
    raw_row_lr = runtime->create_logical_region(ctx, vtx_is, raw_row_fs);
    push_col_idx_lr = runtime->create_logical_region(ctx, edge_is, push_col_idx_fs);
    pull_col_idx_lr = runtime->create_logical_region(ctx, edge_is, pull_col_idx_fs);
    raw_col_lr = runtime->create_logical_region(ctx, edge_is, raw_col_fs);
    for (int i = 0; i < 2; i++)
    {
      dist_lr[i] = runtime->create_logical_region(ctx, vtx_is, dist_fs);
    }
  //}

  E_ID* raw_rows = (E_ID*) malloc(nv * sizeof(E_ID));
  //double ts_start = Realm::Clock::current_time_in_microseconds();
  assert(fread(raw_rows, sizeof(E_ID), (size_t)nv, fd) == (size_t)nv);
  for (V_ID v = 1; v < nv; v++)
    assert(raw_rows[v] >= raw_rows[v-1]);
  assert(raw_rows[nv-1] == ne);
  fclose(fd);

  // Partition the graph
  //double ts_mid = Realm::Clock::current_time_in_microseconds();
  //printf("Loading time = %.2lfus\n", ts_mid - ts_start);
  V_ID left_bound = 0;
  E_ID edge_cnt = 0;
  E_ID edge_cap = (ne + numParts - 1) / numParts;
  int count = 0;
  frontierSize = 0;
  for (V_ID v = 0; v < nv; v++)
  {
    if (v == 0)
      edge_cnt += raw_rows[v];
    else
      edge_cnt += raw_rows[v] - raw_rows[v-1];
    if (edge_cnt > edge_cap)
    {
      rowLeft[count] = left_bound;
      rowRight[count] = v;
      fqLeft[count] = frontierSize;
      // 10 extra slots to handle concer cases
      V_ID mySlots = (rowRight[count] - rowLeft[count]) / SPARSE_THRESHOLD + 10;
      frontierSize += sizeof(FrontierHeader) + mySlots * sizeof(V_ID);
      fqRight[count] = frontierSize - 1;
      count++;
      edge_cnt = 0;
      left_bound = v + 1;
    }
  }
  if (edge_cnt > 0)
  {
    rowLeft[count] = left_bound;
    rowRight[count] = nv - 1;
    fqLeft[count] = frontierSize;
    // 10 extra slots to handle concer cases
    V_ID mySlots = (rowRight[count] - rowLeft[count]) / SPARSE_THRESHOLD + 10;
    frontierSize += sizeof(FrontierHeader) + mySlots * sizeof(V_ID);
    fqRight[count] = frontierSize - 1;
    count++;
  }
  // Make logial regions for frontier queues
  Rect<1> frontier_rect(Point<1>(0), Point<1>(frontierSize-1));
  IndexSpaceT<1> frontier_is =
    runtime->create_index_space(ctx, frontier_rect);
  runtime->attach_name(frontier_is, "frontier_index_space");
  for (int i = 0; i < 2; i++)
    frontier_lr[i] = runtime->create_logical_region(ctx, frontier_is, frontier_fs);
  //double ts_end = Realm::Clock::current_time_in_microseconds();
  //printf("Partitioning time = %.2lfus\n", ts_end - ts_mid);
  assert(count == numParts);
  // First, we partition the vertices
  LegionRuntime::Arrays::Rect<1> color_rect(
      LegionRuntime::Arrays::Point<1>(0), LegionRuntime::Arrays::Point<1>(numParts - 1));
  Domain color_domain = Domain::from_rect<1>(color_rect);
  {
    DomainColoring pvt_vtx_coloring;
    for (int color = 0; color < numParts; color++)
    {
      LegionRuntime::Arrays::Rect<1> subrect_pvt(
          LegionRuntime::Arrays::Point<1>(rowLeft[color]),
          LegionRuntime::Arrays::Point<1>(rowRight[color]));
      pvt_vtx_coloring[color] = Domain::from_rect<1>(subrect_pvt);
    }
    IndexPartition vtx_ip
      = runtime->create_index_partition(ctx, vtx_is, color_domain,
                                        pvt_vtx_coloring, true);
    assert(runtime->is_index_partition_disjoint(ctx, vtx_ip));
    assert(runtime->is_index_partition_complete(ctx, vtx_ip));
    pull_row_ptr_lp = runtime->get_logical_partition(ctx, pull_row_ptr_lr, vtx_ip);
    raw_row_lp = runtime->get_logical_partition(ctx, raw_row_lr, vtx_ip);
    for (int i = 0; i < 2; i++)
    {
      dist_lp[i] = runtime->get_logical_partition(ctx, dist_lr[i], vtx_ip);
    }
  }
  // Next, we partition row_ptrs
  {
    DomainColoring row_coloring;
    for (int color = 0; color < numParts; color++)
    {
      LegionRuntime::Arrays::Rect<1> subrect(
          LegionRuntime::Arrays::Point<1>(color * nv),
          LegionRuntime::Arrays::Point<1>((color + 1) * nv - 1));
      row_coloring[color] = Domain::from_rect<1>(subrect);
    }
    IndexPartition row_ip
        = runtime->create_index_partition(ctx, row_is, color_domain,
                                          row_coloring, true);
    assert(runtime->is_index_partition_disjoint(ctx, row_ip));
    assert(runtime->is_index_partition_complete(ctx, row_ip));
    push_row_ptr_lp = runtime->get_logical_partition(ctx, push_row_ptr_lr, row_ip);
  }
  // Second, we partition the frontiers
  {
    DomainColoring fq_coloring;
    for (int color = 0; color < numParts; color++)
    {
      LegionRuntime::Arrays::Rect<1> subrect(
          LegionRuntime::Arrays::Point<1>(fqLeft[color]),
          LegionRuntime::Arrays::Point<1>(fqRight[color]));
      fq_coloring[color] = Domain::from_rect<1>(subrect);
    }
    IndexPartition fq_ip
      = runtime->create_index_partition(ctx, frontier_is, color_domain,
                                        fq_coloring, true);
    assert(runtime->is_index_partition_disjoint(ctx, fq_ip));
    assert(runtime->is_index_partition_disjoint(ctx, fq_ip));
    frontier_lp[0] = runtime->get_logical_partition(ctx, frontier_lr[0], fq_ip);
    frontier_lp[1] = runtime->get_logical_partition(ctx, frontier_lr[1], fq_ip);
  }
  // Third, we partition the edges
  {
    DomainColoring edges_coloring;
    E_ID index = 0;
    for (int color = 0; color < numParts; color++)
    {
      log_lux.print("rowLeft = %u rowRight = %u",
                    rowLeft[color], rowRight[color]);
      LegionRuntime::Arrays::Rect<1> subrect(
          LegionRuntime::Arrays::Point<1>(index),
          LegionRuntime::Arrays::Point<1>(raw_rows[rowRight[color]]- 1));
      index = raw_rows[rowRight[color]];
      edges_coloring[color] = Domain::from_rect<1>(subrect);
    }
    IndexPartition col_idx_ip
      = runtime->create_index_partition(ctx, edge_is, color_domain,
                                        edges_coloring, true);
    push_col_idx_lp =
        runtime->get_logical_partition(ctx, push_col_idx_lr, col_idx_ip);
    pull_col_idx_lp =
        runtime->get_logical_partition(ctx, pull_col_idx_lr, col_idx_ip);
    raw_col_lp =
        runtime->get_logical_partition(ctx, raw_col_lr, col_idx_ip);
  }
  free(raw_rows);
}

