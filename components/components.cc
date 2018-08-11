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

  ArgumentMap local_args;
  // Init phase
  Rect<1> task_rect(0, graph.numParts - 1);
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
