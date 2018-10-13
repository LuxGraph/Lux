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
#include "graph.h"
#include "legion.h"
#include <map>
#include <string.h>

LegionRuntime::Logger::Category log_lux("graph");

template<typename T>
void alloc_fs(Context ctx, Runtime *runtime, FieldSpace fs)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(T), FID_DATA);
}

Graph::Graph(Context ctx, HighLevelRuntime *runtime,
             int _numParts, std::string& file_name)
: numParts(_numParts)
{
  //{
    FILE* fd = fopen(file_name.c_str(), "rb");
    assert(fd != NULL);
    size_t fread_ret = fread(&nv, sizeof(V_ID), 1, fd);
    assert(fread_ret == 1);
    fread_ret = fread(&ne, sizeof(E_ID), 1, fd);
    assert(fread_ret == 1);
    log_lux.print("Load graph: numNodes(%u) numEdges(%zu)", nv, ne);
    Rect<1> vtx_rect(Point<1>(0), Point<1>(nv - 1));
    IndexSpaceT<1> vtx_is =
      runtime->create_index_space(ctx, vtx_rect);
    runtime->attach_name(vtx_is, "vertices_index_space");
    Rect<1> edge_rect(Point<1>(0), Point<1>(ne - 1));
    IndexSpaceT<1> edge_is =
      runtime->create_index_space(ctx, edge_rect);
    runtime->attach_name(edge_is, "edges_index_space");

    FieldSpace row_ptr_fs = runtime->create_field_space(ctx);
    runtime->attach_name(row_ptr_fs, "row_ptrs(NodeStruct)");
    FieldSpace raw_row_fs = runtime->create_field_space(ctx);
    runtime->attach_name(raw_row_fs, "raw_rows(E_ID)");
    FieldSpace in_vtx_fs = runtime->create_field_space(ctx);
    runtime->attach_name(in_vtx_fs, "in_vtxs(V_ID)");
    FieldSpace col_idx_fs = runtime->create_field_space(ctx);
    runtime->attach_name(col_idx_fs, "col_idxs(EdgeStruct)");
    FieldSpace raw_col_fs = runtime->create_field_space(ctx);
    runtime->attach_name(raw_col_fs, "raw_cols(V_ID)");
    FieldSpace dist_fs = runtime->create_field_space(ctx);
    runtime->attach_name(dist_fs, "out_field_space");

    // Allocate fields
    alloc_fs<NodeStruct>(ctx, runtime, row_ptr_fs);
    alloc_fs<E_ID>(ctx, runtime, raw_row_fs);
    alloc_fs<V_ID>(ctx, runtime, in_vtx_fs);
    alloc_fs<EdgeStruct>(ctx, runtime, col_idx_fs);
    alloc_fs<V_ID>(ctx, runtime, raw_col_fs);
    alloc_fs<Vertex>(ctx, runtime, dist_fs);

    // Make logical regions
    pull_row_ptr_lr = runtime->create_logical_region(ctx, vtx_is, row_ptr_fs);
    raw_row_lr = runtime->create_logical_region(ctx, vtx_is, raw_row_fs);
    in_vtx_lr = runtime->create_logical_region(ctx, edge_is, in_vtx_fs);
    pull_col_idx_lr = runtime->create_logical_region(ctx, edge_is, col_idx_fs);
    raw_col_lr = runtime->create_logical_region(ctx, edge_is, raw_col_fs);
    for (int i = 0; i < 2; i++)
    {
      dist_lr[i] = runtime->create_logical_region(ctx, vtx_is, dist_fs);
    }
#ifdef VERTEX_DEGREE
    degree_lr = runtime->create_logical_region(ctx, vtx_is, in_vtx_fs);
#else
    degree_lr = LogicalRegion::NO_REGION;
#endif

#ifdef EDGE_WEIGHT
    FieldSpace raw_weight_fs = runtime->create_field_space(ctx);
    runtime->attach_name(raw_weight_fs, "raw_weight");
    alloc_fs<WeightType>(ctx, runtime, raw_weight_fs);
    raw_weight_lr = runtime->create_logical_region(ctx, edge_is, raw_weight_fs);
#else
    raw_weight_lr = LogicalRegion::NO_REGION;
#endif
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
  std::vector<std::pair<V_ID, V_ID> > bounds;
  for (V_ID v = 0; v < nv; v++)
  {
    if (v == 0)
      edge_cnt += raw_rows[v];
    else
      edge_cnt += raw_rows[v] - raw_rows[v-1];
    if (edge_cnt > edge_cap)
    {
      bounds.push_back(std::make_pair(left_bound, v));
      edge_cnt = 0;
      left_bound = v + 1;
    }
  }
  if (edge_cnt > 0)
  {
    bounds.push_back(std::make_pair(left_bound, nv - 1));
  }
  //double ts_end = Realm::Clock::current_time_in_microseconds();
  //printf("Partitioning time = %.2lfus\n", ts_end - ts_mid);
  assert(bounds.size() == (size_t)numParts);
  // First, we partition the vertices
  LegionRuntime::Arrays::Rect<1> color_rect(
      LegionRuntime::Arrays::Point<1>(0), LegionRuntime::Arrays::Point<1>(numParts - 1));
  Domain color_domain = Domain::from_rect<1>(color_rect);
  {
    DomainColoring pvt_vtx_coloring;
    for (int color = 0; color < numParts; color++)
    {
      LegionRuntime::Arrays::Rect<1> subrect_pvt(
          LegionRuntime::Arrays::Point<1>(bounds[color].first),
          LegionRuntime::Arrays::Point<1>(bounds[color].second));
      pvt_vtx_coloring[color] = Domain::from_rect<1>(subrect_pvt);
    }
    IndexPartition vtx_ip
      = runtime->create_index_partition(ctx, vtx_is, color_domain,
                                        pvt_vtx_coloring, true);
    pull_row_ptr_lp = runtime->get_logical_partition(ctx, pull_row_ptr_lr, vtx_ip);
    raw_row_lp = runtime->get_logical_partition(ctx, raw_row_lr, vtx_ip);
    for (int i = 0; i < 2; i++)
    {
      dist_lp[i] = runtime->get_logical_partition(ctx, dist_lr[i], vtx_ip);
    }
#ifdef VERTEX_DEGREE
    degree_lp = runtime->get_logical_partition(ctx, degree_lr, vtx_ip);
#else
    degree_lp = LogicalPartition::NO_PART;
#endif
  }
  // Second, we partition the edges
  {
    DomainColoring edges_coloring;
    E_ID index = 0;
    for (int color = 0; color < numParts; color++)
    {
      log_lux.print("left_bound = %u right_bound = %u",
                    bounds[color].first, bounds[color].second);
      LegionRuntime::Arrays::Rect<1> subrect(
          LegionRuntime::Arrays::Point<1>(index),
          LegionRuntime::Arrays::Point<1>(raw_rows[bounds[color].second]- 1));
      index = raw_rows[bounds[color].second];
      edges_coloring[color] = Domain::from_rect<1>(subrect);
    }
    IndexPartition col_idx_ip
      = runtime->create_index_partition(ctx, edge_is, color_domain,
                                        edges_coloring, true);
    pull_col_idx_lp =
        runtime->get_logical_partition(ctx, pull_col_idx_lr, col_idx_ip);
    raw_col_lp =
        runtime->get_logical_partition(ctx, raw_col_lr, col_idx_ip);
    in_vtx_lp =
        runtime->get_logical_partition(ctx, in_vtx_lr, col_idx_ip);
#ifdef EDGE_WEIGHT
    raw_weight_lp =
        runtime->get_logical_partition(ctx, raw_weight_lr, col_idx_ip);
#else
    raw_weight_lp = LogicalPartition::NO_PART;
#endif
  }
  free(raw_rows);
}

PullLoadTask::PullLoadTask(const Graph &graph,
                           const IndexSpaceT<1> &domain,
                           const ArgumentMap &arg_map,
                           std::string &fn)
  : IndexLauncher(PULL_LOAD_TASK_ID, domain,
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
#ifdef EDGE_WEIGHT
  // regions[2]: raw_weights
  {
    RegionRequirement rr(graph.raw_weight_lp, 0/*projection id*/,
                         WRITE_ONLY, EXCLUSIVE, graph.raw_weight_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
#endif
}

PullScanTask::PullScanTask(const Graph &graph,
                           Context ctx, Runtime* runtime)
  : TaskLauncher(PULL_SCAN_TASK_ID, TaskArgument(NULL, 0))
{
  // regions[0]: degrees
  {
    RegionRequirement rr(graph.degree_lr,
                         WRITE_ONLY, EXCLUSIVE, graph.degree_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1..numParts]: raw_cols[i]
  Rect<1> task_rect(0, graph.numParts - 1);
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    LogicalRegion raw_col = runtime->get_logical_subregion_by_color(
                                ctx, graph.raw_col_lp, DomainPoint(*it));
    RegionRequirement rr(raw_col,
                         READ_ONLY, EXCLUSIVE, graph.raw_col_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

void pull_load_task_impl(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime* runtime)
{
#ifdef EDGE_WEIGHT
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
#else
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
#endif
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
#ifdef EDGE_WEIGHT
  const AccessorWO<WeightType, 1> acc_raw_weights(regions[2], FID_DATA);
  Rect<1> rect_raw_weights = runtime->get_index_space_domain(
                               ctx, task->regions[2].region.get_index_space());
  assert(rect_raw_weights == rect_raw_cols);
  assert(acc_raw_weights.accessor.is_dense_arbitrary(rect_raw_weights));
  WeightType* raw_weights = acc_raw_weights.ptr(rect_raw_weights.lo);
#endif
  // Load row pointers and col indices
  log_lux.print("Load task: file(%s) rowLeft(%u) rowRight(%u) colLeft(%zu) colRight(%zu)",
               file_name, rowLeft, rowRight, colLeft, colRight);
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
  assert(fseek_ret == 0);
  fread_ret =
    fread(raw_cols, sizeof(V_ID), (size_t)(colRight - colLeft + 1), fd);
  assert(fread_ret == colRight - colLeft + 1);
#ifdef EDGE_WEIGHT
  fseek_ret =
    fseeko(fd, FILE_HEADER_SIZE + sizeof(E_ID) * (size_t)nv 
               + sizeof(V_ID) * (size_t)ne
               + sizeof(WeightType) * (size_t)colLeft, SEEK_SET);
  assert(fseek_ret == 0);
  fread_ret =
    fread(raw_weights, sizeof(WeightType), (size_t)(colRight - colLeft + 1), fd);
  assert(fread_ret == colRight - colLeft + 1);
#endif
  fclose(fd);
}

void pull_scan_task_impl(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime* runtime)
{
  const AccessorWO<V_ID, 1> acc_degrees(regions[0], FID_DATA);
  Rect<1> rect_degrees = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  V_ID rowLeft = rect_degrees.lo[0], rowRight = rect_degrees.hi[0];
  assert(rowLeft == 0);
  assert(acc_degrees.accessor.is_dense_arbitrary(rect_degrees));
  V_ID* degrees = acc_degrees.ptr(rect_degrees.lo);
  for (V_ID n = rowLeft; n <= rowRight; n++)
    degrees[n] = 0;
  for (unsigned i = 1; i < regions.size(); i++) {
    const AccessorRO<V_ID, 1> acc_raw_col(regions[i], FID_DATA);
    Rect<1> rect_raw_col = runtime->get_index_space_domain(
                               ctx, task->regions[i].region.get_index_space());
    E_ID colLeft = rect_raw_col.lo[0], colRight = rect_raw_col.hi[0];
    assert(acc_raw_col.accessor.is_dense_arbitrary(rect_raw_col));
    const V_ID* raw_col = acc_raw_col.ptr(rect_raw_col.lo);
    for (E_ID e = colLeft; e <= colRight; e++)
      degrees[raw_col[e - colLeft]] ++;
  }
}

PullInitTask::PullInitTask(const Graph &graph,
                           const IndexSpaceT<1> &domain,
                           const ArgumentMap &arg_map)
  : IndexLauncher(PULL_INIT_TASK_ID, domain,
                  TaskArgument(&graph, sizeof(Graph)), arg_map)
{
  // regions[0]: row_ptrs
  {
    RegionRequirement rr(graph.pull_row_ptr_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.pull_row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: in_vtxs
  {
    RegionRequirement rr(graph.in_vtx_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.in_vtx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: col_idxs
  {
    RegionRequirement rr(graph.pull_col_idx_lp, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.pull_col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: new_pr
  {
    RegionRequirement rr(graph.dist_lp[0], 0/*idenity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.dist_lr[0],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[4]: raw_rows
  {
    RegionRequirement rr(graph.raw_row_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.raw_row_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[5]: raw_cols
  {
    RegionRequirement rr(graph.raw_col_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.raw_col_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
#ifdef VERTEX_DEGREE
  // regions[6]: degrees
  {
    RegionRequirement rr(graph.degree_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.degree_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
#endif
#ifdef EDGE_WEIGHT
  // regions[6/7]: raw_weights
  {
    RegionRequirement rr(graph.raw_weight_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.raw_weight_lr,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
#endif
}

PullAppTask::PullAppTask(const Graph &graph,
                         const IndexSpaceT<1> &domain,
                         const ArgumentMap &arg_map,
                         int iter)
  : IndexLauncher(PULL_APP_TASK_ID, domain,
                  TaskArgument(&iter, sizeof(int)), arg_map)
{
  // regions[0]: row_ptrs
  {
    RegionRequirement rr(graph.pull_row_ptr_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.pull_row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: in_vtxs
  {
    RegionRequirement rr(graph.in_vtx_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.in_vtx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: col_idxs
  {
    RegionRequirement rr(graph.pull_col_idx_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.pull_col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: old_pr
  {
    RegionRequirement rr(graph.dist_lr[iter%2], 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.dist_lr[iter%2],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[4]: new_pr
  {
    RegionRequirement rr(graph.dist_lp[(iter+1)%2], 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, graph.dist_lr[(iter+1)%2],
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
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

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  // Pull-based Load Task
  {
    TaskVariantRegistrar registrar(PULL_LOAD_TASK_ID, "load_task (pull)");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<pull_load_task_impl>(
        registrar, "load_task (pull)");
  }
  {
    TaskVariantRegistrar registrar(PULL_SCAN_TASK_ID, "scan_task (pull)");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<pull_scan_task_impl>(registrar, "scan_task (pull)");
  }
  {
    TaskVariantRegistrar registrar(PULL_INIT_TASK_ID, "init_task (pull)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<GraphPiece, pull_init_task_impl>(
        registrar, "init_task (pull)");
  }
  {
    TaskVariantRegistrar registrar(PULL_APP_TASK_ID, "app_task (pull)");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<pull_app_task_impl>(
        registrar, "app_task (pull)");
  }
  Runtime::add_registration_callback(update_mappers);

  return Runtime::start(argc, argv);
}
