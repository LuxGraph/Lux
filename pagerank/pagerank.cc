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
#include "../core/lux_mapper.h"
#include "../core/graph.h"
#include "realm/machine.h"
#include "math.h"
#include "legion.h"
#include "queue"
#include "map"
#include <string.h>

LegionRuntime::Logger::Category log_pr("pagerank");

void parse_input_args(char **argv, int argc, int &num_gpu,
                      int &num_iter, std::string &file_name,
                      bool &verbose);

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int numGPU = 0, numIter = 0;
  std::string filename;
  bool verbose = false;
  // parse input arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, numGPU, numIter, filename, verbose);
    log_pr.print("PageRank settings: numPartitions(%d) numIter(%d)" 
                 " filename = %s", numGPU, numIter, filename.c_str());
    if ((numGPU <= 0) || (numIter <= 0)) {
      fprintf(stderr, "numGPU(%d) and numIter(%d) must be greater than zero.\n",
              numGPU, numIter);
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
  size_t max_zc_usage = 0, max_fb_usage = 0;
  max_zc_usage = graph.ne * sizeof(V_ID) //raw_cols
                 + graph.nv * sizeof(E_ID) //raw_rows
                 + graph.nv * sizeof(V_ID) //degrees
                 + graph.nv * 2 * sizeof(Vertex); //new_pr+old_pr
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
    fb_usage = myNumEdges * sizeof(EdgeStruct) //row_ptrs
               + myNumNodes * sizeof(NodeStruct) //col_idxs
               + myNumNodes * sizeof(V_ID)  //in_vtxs
               + myNumNodes * sizeof(Vertex) //newPrFb
               + graph.nv * sizeof(Vertex); //oldPrFb
    max_fb_usage = fb_usage > max_fb_usage ? fb_usage : max_fb_usage;
  }
  printf("[Memory Setting] Set ll:fsize >= %zuMB and ll:zsize >= %zuMB\n",
          max_fb_usage / 1024 / 1024 + 1, max_zc_usage / 1024 / 1024 + 1);
  
  ArgumentMap local_args;
  // Init phase
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);
  PullLoadTask load_task(graph, task_is, local_args, filename);
  FutureMap fm = runtime->execute_index_space(ctx, load_task);
  fm.wait_all_results();
  PullScanTask scan_task(graph, ctx, runtime);
  Future f = runtime->execute_task(ctx, scan_task);
  f.get_void_result();

  PullInitTask init_task(graph, task_is, local_args);
  fm = runtime->execute_index_space(ctx, init_task);
  fm.wait_all_results();
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    GraphPiece piece = fm.get_result<GraphPiece>(*it);
    local_args.set_point(*it, TaskArgument(&piece, sizeof(GraphPiece)));
  }
 
  // PageRank phase
  int iteration = 0;
  log_pr.print("Start PageRank computation...");
  double ts_start = Realm::Clock::current_time_in_microseconds(); 
  for (int i = 0; i < numIter; i++) {
    iteration = i;
    PullAppTask app_task(graph, task_is, local_args, iteration);
    fm = runtime->execute_index_space(ctx, app_task);
  }
  fm.wait_all_results();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double sim_time = 1e-6 * (ts_end - ts_start);
  log_pr.print("Finish PageRank computation...");
  printf("ELAPSED TIME = %7.7f s\n", sim_time);
}

void parse_input_args(char **argv, int argc,
                      int &numGPU, int &numIter,
                      std::string &filename, bool &verbose)
{
  for (int i = 1; i < argc; i++) 
  {
    if ((!strcmp(argv[i], "-ng")) || (!strcmp(argv[i], "-ll:gpu"))) 
    {
      numGPU = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ni")) 
    {
      numIter = atoi(argv[++i]);
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
  }
}

#include "../core/pull_model.inl"
