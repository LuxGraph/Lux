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
#include "math.h"
#include "legion.h"
#include "queue"
#include "map"
#include <string.h>

LegionRuntime::Logger::Category log_pr("pagerank");

void parse_input_args(char **argv, int argc, int &num_gpu,
                      int &num_iter, std::string &file_name);

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int numGPU = 0, numIter = 0;
  std::string filename;
  // parse input arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc, numGPU, numIter, filename);
    log_pr.print("PageRank settings: numPartitions(%d) numIter(%d)" 
                 "filename = %s\n", numGPU, numIter, filename.c_str());
    if ((numGPU <= 0) || (numIter <= 0)) {
      fprintf(stderr, "numGPU(%d) and numIter(%d) must be greater than zero.\n",
              numGPU, numIter);
      return;
    }
  }

  Graph graph(ctx, runtime, numGPU, filename);
 
  ArgumentMap local_args;
  // Init phase
  Rect<1> task_rect(0, graph.numParts - 1);
  IndexSpaceT<1> task_is = runtime->create_index_space(ctx, task_rect);
  LoadTask load_task(graph, task_is, local_args, filename);
  FutureMap fm = runtime->execute_index_space(ctx, load_task);
  fm.wait_all_results();
  ScanTask scan_task(graph);
  Future f = runtime->execute_task(ctx, scan_task);
  f.get_void_result();

  InitTask init_task(graph, task_is, local_args);
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
    AppTask app_task(graph, task_is, local_args, iteration);
    fm = runtime->execute_index_space(ctx, app_task);
  }
  fm.wait_all_results();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double sim_time = 1e-6 * (ts_end - ts_start);
  log_pr.print("Finish PageRank computation...");
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
                      int &numGPU, int &numIter, std::string &filename)
{
  for (int i = 1; i < argc; i++) 
  {
    if (!strcmp(argv[i], "-ng")) 
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
  // Load Task
  {
    TaskVariantRegistrar registrar(LOAD_TASK_ID, "load_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<load_task_impl>(registrar, "load_task");
  }
  {
    TaskVariantRegistrar registrar(SCAN_TASK_ID, "scan_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<scan_task_impl>(registrar, "scan_task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<GraphPiece, init_task_impl>(
        registrar, "init_task");
  }
  {
    TaskVariantRegistrar registrar(APP_TASK_ID, "pagerank_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<app_task_impl>(
        registrar, "pagerank_task");
  }
  Runtime::add_registration_callback(update_mappers);

  return Runtime::start(argc, argv);
}

