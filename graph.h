/* Copyright 2016 Stanford, UT Austin, LANL
   Add a comment to this line
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

#ifndef _LUX_GRAPH_H_
#define _LUX_GRAPH_H_

#include <cstdio>
#include "legion.h"
#include <unistd.h>

using namespace Legion;
template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

typedef uint32_t V_ID;
typedef uint64_t E_ID;
#define MAX_FILE_LEN 64
#define ALPHA 0.15
#define FILE_HEADER_SIZE (sizeof(E_ID) + sizeof(V_ID))

enum {
  TOP_LEVEL_TASK_ID,
  LOAD_TASK_ID,
  SCAN_TASK_ID,
  INIT_TASK_ID,
  PAGERANK_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

struct NodeStruct {
  E_ID index;
  V_ID degree;
};

struct EdgeStruct {
  V_ID src, dst;
};

class Graph
{
public:
  Graph(Context ctx, Runtime* rt, int _numParts, std::string& file_name);
  int numParts;
  V_ID nv;
  E_ID ne;
  LogicalRegion row_ptr_lr;
  LogicalPartition row_ptr_lp;
  LogicalRegion raw_row_lr;
  LogicalPartition raw_row_lp;
  LogicalRegion in_vtx_lr;
  LogicalPartition in_vtx_lp;
  LogicalRegion col_idx_lr;
  LogicalPartition col_idx_lp;
  LogicalRegion raw_col_lr;
  LogicalPartition raw_col_lp;
  LogicalRegion degree_lr;
  LogicalPartition degree_lp;
  LogicalRegion dist_lr[2];
  LogicalPartition dist_lp[2];
};

class GraphPiece
{
public:
  V_ID myInVtxs;
  V_ID nv;
  E_ID ne;
  float *oldPrFb, *newPrFb;
};

class LoadTask : public IndexLauncher
{
public:
  LoadTask(const Graph &graph,
           const IndexSpaceT<1> &domain,
           const ArgumentMap &arg_map,
           std::string &fn);
};

class ScanTask : public TaskLauncher
{
public:
  ScanTask(const Graph &graph);
};

class InitTask : public IndexLauncher
{
public:
  InitTask(const Graph &graph,
           const IndexSpaceT<1> &domain,
           const ArgumentMap &arg_map);
};

class BFSTask : public IndexLauncher
{
public:
  BFSTask(const Graph &graph,
          const IndexSpaceT<1> &domain,
          const ArgumentMap &arg_map,
          int iteration);
};

void pagerank_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);

GraphPiece init_task_impl(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime);
#endif
