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
#include "../core/graph.h"
#include "../core/lux_mapper.h"
#include "legion.h"
#include <string.h>

LegionRuntime::Logger::Category log_cc("cc");
LegionRuntime::Logger::Category log_lux("lux");

void parse_input_args(char **argv, int argc, int &num_gpu,
                      std::string &file_name, bool &verbose, bool &check);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int numGPU = 0;
  std::string filename;
  bool verbose = false;
  bool check = false;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, numGPU, filename, verbose, check);
    log_cc.print("CC settings: numPartitions(%d) filename(%s)",
                 numGPU, filename.c_str());
    if (numGPU <= 0) {
      fprintf(stderr, "numGPU(%d) must be greater than zero.\n",
              numGPU);
      return;
    }
    size_t numNodes = Realm::Machine::get_machine().get_address_space_count();
    assert(numNodes > 0);
    numGPU = numGPU * numNodes;
  }

  Graph graph(ctx, runtime, numGPU, filename);
  graph.verbose = verbose;
  Rect<1> task_rect(0, graph.numParts - 1);

  // First we compute and print memory requriements
  size_t max_zc_usage = 0, max_fb_usage = 0, max_num_edges = 0;
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    size_t fb_usage = 0;
    LogicalRegion col_idx = runtime->get_logical_subregion_by_color(
                                ctx, graph.pull_col_idx_lp, DomainPoint(*it));
    LogicalRegion row_ptr = runtime->get_logical_subregion_by_color(
                                ctx, graph.pull_row_ptr_lp, DomainPoint(*it));
    Rect<1> r = runtime->get_index_space_domain(ctx,
                    col_idx.get_index_space());
    size_t myNumEdges = r.hi[0] - r.lo[0] + 1;
    r = runtime->get_index_space_domain(ctx, row_ptr.get_index_space());
    size_t myNumNodes = r.hi[0] - r.lo[0] + 1;
    fb_usage = myNumEdges * sizeof(EdgeStruct2) //pull_col_idxs
               + myNumEdges * sizeof(EdgeStruct) //push_col_idxs
               + myNumNodes * sizeof(NodeStruct) //pull_raw_rows
               + graph.nv * sizeof(NodeStruct) //push_raw_rows
               + myNumNodes * 2 * sizeof(Vertex) //newPrFb+oldPrFb
               + graph.nv * sizeof(Vertex)   //allPrFb
               + graph.frontierSize * 2; //newFqFb+oldFqFb
    max_fb_usage = fb_usage > max_fb_usage ? fb_usage : max_fb_usage;
    max_num_edges = myNumEdges > max_num_edges ? myNumEdges : max_num_edges;
  }
  max_zc_usage = graph.ne * sizeof(V_ID) //raw_cols
                 + graph.nv * sizeof(E_ID) //raw_rows
                 + graph.nv * 2 * sizeof(Vertex) //new_pr+old_pr
                 + graph.frontierSize * 2 //new_fq+old_fq
                 + graph.nv * sizeof(NodeStruct) //temp_memory
                 + max_num_edges * sizeof(EdgeStruct); //temp_memory

  printf("[Memory Setting] Set ll:fsize >= %zuMB and ll:zsize >= %zuMB\n",
          max_fb_usage / 1024 / 1024 + 1, max_zc_usage / 1024 / 1024 + 1);

  ArgumentMap local_args;
  // Init phase
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);
  PushLoadTask load_task(graph, task_is, local_args, filename);
  FutureMap fm = runtime->execute_index_space(ctx, load_task);
  fm.wait_all_results();
  PushInitVtxTask init_vtx_task(graph);
  Future f = runtime->execute_task(ctx, init_vtx_task);
  f.get_void_result();

  PushInitTask init_task(graph, task_is, local_args);
  fm = runtime->execute_index_space(ctx, init_task);
  fm.wait_all_results();
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    GraphPiece piece = fm.get_result<GraphPiece>(*it);
    local_args.set_point(*it, TaskArgument(&piece, sizeof(GraphPiece)));
  }

  // CC phase
  FutureMap fms[SLIDING_WINDOW];
  int iter = 0;
  log_cc.print("Start Connected Components computation...");
  double ts_start = Realm::Clock::current_time_in_microseconds();
  while (true) {
    if (iter >= SLIDING_WINDOW) {
      fm = fms[iter % SLIDING_WINDOW];
      fm.wait_all_results();
      bool halt = true;
      for (PointInRectIterator<1> it(task_rect); it(); it++) {
        V_ID numNodes = fm.get_result<V_ID>(*it);
        if (numNodes > 0) halt = false;
      }
      if (halt) break;
    }
    PushAppTask app_task(graph, task_is, local_args, iter);
    fms[iter % SLIDING_WINDOW] = runtime->execute_index_space(ctx, app_task);
    iter ++;
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double sim_time = 1e-6 * (ts_end - ts_start);
  runtime->issue_execution_fence(ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = runtime->issue_timing_measurement(ctx, timer);
  future.get_void_result();
  log_cc.print("Finish Connected Components computation...");
  printf("ELAPSED TIME = %7.7f s\n", sim_time);

  //check tasks
  if (check) {
    CheckTask check_task(graph, task_is, local_args, iter);
    fm = runtime->execute_index_space(ctx, check_task);
    fm.wait_all_results();
    log_cc.print("Correctness check completed...");
  }
}

void parse_input_args(char **argv, int argc,
                      int &numGPU, std::string &filename,
                      bool &verbose, bool &check)
{
  for (int i = 1; i < argc; i++) 
  {
    if ((!strcmp(argv[i], "-ng")) || (!strcmp(argv[i], "-ll:gpu"))) 
    {
      numGPU = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-file"))
    {
      filename = std::string(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "-verbose")) || (!strcmp(argv[i], "-v")))
    {
      verbose = true;
      continue;
    }
    if ((!strcmp(argv[i], "-check")) || (!strcmp(argv[i], "-c")))
    {
      check = true;
      continue;
    }
  }
}

#include "../core/push_model.inl"
