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

#include "../core/graph.h"
#include "../core/cuda_helper.h"
#include "realm/runtime_impl.h"
#include "realm/cuda/cuda_module.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

const int CUDA_NUM_THREADS = 128;
const int BLOCK_SIZE_LIMIT = 32768;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

__global__
void cf_kernel(V_ID rowLeft,
               V_ID rowRight,
               E_ID colLeft,
               const NodeStruct* row_ptrs,
               const EdgeStruct* col_idxs,
               Vertex* old_pr_fb,
               Vertex* new_pr_fb)
{
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ float vec[CUDA_NUM_THREADS * K];
  __shared__ float accErr[CUDA_NUM_THREADS * K];
  __shared__ float srcVec[CUDA_NUM_THREADS * K];
  __shared__ EdgeStruct es[CUDA_NUM_THREADS];
  __shared__ E_ID blkColStart;
  for (V_ID blkRowStart = blockIdx.x * blockDim.x + rowLeft;
       blkRowStart <= rowRight;
       blkRowStart += blockDim.x * gridDim.x)
  {
    E_ID myNumEdges = 0, scratchOffset, totalNumEdges = 0;
    V_ID curVtx = blkRowStart + threadIdx.x;
    if (curVtx <= rowRight) {
      NodeStruct ns = row_ptrs[curVtx - rowLeft];
      E_ID start_col_idx, end_col_idx = ns.index;
      if (curVtx == rowLeft)
        start_col_idx = colLeft;
      else
        start_col_idx = row_ptrs[curVtx - rowLeft - 1].index;
      myNumEdges = end_col_idx - start_col_idx;
      if (threadIdx.x == 0)
        blkColStart = start_col_idx;
      for (int i = 0; i < K; i++) {
        vec[threadIdx.x * K + i] = old_pr_fb[curVtx - rowLeft].v[i];
        accErr[threadIdx.x * K + i] = 0;
      }
    }
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
    E_ID done = 0;
    while (totalNumEdges > 0) {
      if (threadIdx.x < totalNumEdges) {
        es[threadIdx.x] = col_idxs[blkColStart + done + threadIdx.x - colLeft];
        __syncthreads();
        int blksize = totalNumEdges > CUDA_NUM_THREADS ?
                      CUDA_NUM_THREADS : totalNumEdges;
        for (int i = 0; i < K; i++) {
          srcVec[i * blksize + threadIdx.x] =
            old_pr_fb[es[(i * blksize + threadIdx.x) / K].src].v[threadIdx.x % K];
        }
        __syncthreads();
        int dst_tid = es[threadIdx.x].dst - blkRowStart;
        float dotProd = 0.0f;
        for (int i = 0; i < K; i++)
          dotProd += srcVec[threadIdx.x * K + i] * vec[dst_tid * K + i];
        float err = es[threadIdx.x].weight - dotProd;
        for (int i = 0; i < K; i++)
          accErr[dst_tid * K + i] += err * srcVec[threadIdx.x * K + i];
      }
      done += CUDA_NUM_THREADS;
      totalNumEdges -= (totalNumEdges > CUDA_NUM_THREADS) ?
                       CUDA_NUM_THREADS : totalNumEdges;
    }
    __syncthreads();
    if (curVtx <= rowRight) {
      for (int i = 0; i < K; i++) {
        int offset = threadIdx.x * K + i;
        new_pr_fb[curVtx].v[i] = 
          vec[offset] + GAMMA * (accErr[offset] - LAMBDA * vec[offset]);
       }
    }
  }
}

void pull_app_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 5);
  assert(task->regions.size() == 5);
  const GraphPiece *piece = (GraphPiece*) task->local_args;

  const AccessorRO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorRO<V_ID, 1> acc_in_vtx(regions[1], FID_DATA);
  const AccessorRO<EdgeStruct, 1> acc_col_idx(regions[2], FID_DATA);
  const AccessorRO<Vertex, 1> acc_old_pr(regions[3], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[4], FID_DATA);
  Rect<1> rect_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_in_vtx = runtime->get_index_space_domain(
                            ctx, task->regions[1].region.get_index_space());
  Rect<1> rect_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_old_pr = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[4].region.get_index_space());
  assert(acc_row_ptr.accessor.is_dense_arbitrary(rect_row_ptr));
  assert(acc_in_vtx.accessor.is_dense_arbitrary(rect_in_vtx));
  assert(acc_col_idx.accessor.is_dense_arbitrary(rect_col_idx));
  assert(acc_old_pr.accessor.is_dense_arbitrary(rect_old_pr));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  const NodeStruct* row_ptrs = acc_row_ptr.ptr(rect_row_ptr);
  const V_ID* in_vtxs = acc_in_vtx.ptr(rect_in_vtx);
  const EdgeStruct* col_idxs = acc_col_idx.ptr(rect_col_idx);
  const Vertex* old_pr = acc_old_pr.ptr(rect_old_pr);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  V_ID rowLeft = rect_row_ptr.lo[0], rowRight = rect_row_ptr.hi[0];
  E_ID colLeft = rect_col_idx.lo[0], colRight = rect_col_idx.hi[0];

  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  checkCUDA(cudaMemcpy(piece->oldPrFb, old_pr,
                       size_t(piece->nv) * sizeof(Vertex),
                       cudaMemcpyHostToDevice));
  cf_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft,
      row_ptrs, col_idxs, piece->oldPrFb, piece->newPrFb);
  // Need to copy results back to new_pr
  cudaDeviceSynchronize();
  checkCUDA(cudaMemcpy(new_pr, piece->newPrFb,
                       (rowRight - rowLeft + 1) * sizeof(Vertex),
                       cudaMemcpyDeviceToHost));
}

__global__
void init_kernel(V_ID rowLeft,
                 V_ID rowRight,
                 E_ID colLeft,
                 NodeStruct* row_ptrs,
                 EdgeStruct* col_idxs,
                 const E_ID* raw_rows,
                 const V_ID* raw_cols,
                 const WeightType* raw_weights)
{
  for (V_ID n = blockIdx.x * blockDim.x + threadIdx.x;
       n + rowLeft <= rowRight; n += blockDim.x * gridDim.x)
  {
    V_ID curVtx = n + rowLeft;
    E_ID startColIdx, endColIdx = raw_rows[n];
    if (n == 0)
      startColIdx = colLeft;
    else
      startColIdx = raw_rows[n - 1];
    row_ptrs[n].index = endColIdx;
    for (E_ID e = startColIdx; e < endColIdx; e++) {
      col_idxs[e - colLeft].src = raw_cols[e - colLeft];
      col_idxs[e - colLeft].dst = curVtx;
      col_idxs[e - colLeft].weight = raw_weights[e - colLeft];
    }
  }
}

GraphPiece pull_init_task_impl(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
#ifdef VERTEX_DEGREE
  assert(false);
#endif
#ifndef EDGE_WEIGHT
  assert(false);
#endif
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);

  const Graph *graph = (Graph*) task->args;
  const AccessorWO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorWO<V_ID, 1> acc_in_vtx(regions[1], FID_DATA);
  const AccessorWO<EdgeStruct, 1> acc_col_idx(regions[2], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[3], FID_DATA);
  const AccessorRO<E_ID, 1> acc_raw_rows(regions[4], FID_DATA);
  const AccessorRO<V_ID, 1> acc_raw_cols(regions[5], FID_DATA);

  Rect<1> rect_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_in_vtx = runtime->get_index_space_domain(
                            ctx, task->regions[1].region.get_index_space());
  Rect<1> rect_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_raw_rows = runtime->get_index_space_domain(
                              ctx, task->regions[4].region.get_index_space());
  Rect<1> rect_raw_cols = runtime->get_index_space_domain(
                              ctx, task->regions[5].region.get_index_space());

  assert(acc_row_ptr.accessor.is_dense_arbitrary(rect_row_ptr));
  assert(acc_in_vtx.accessor.is_dense_arbitrary(rect_in_vtx));
  assert(acc_col_idx.accessor.is_dense_arbitrary(rect_col_idx));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  assert(acc_raw_rows.accessor.is_dense_arbitrary(rect_raw_rows));
  assert(acc_raw_cols.accessor.is_dense_arbitrary(rect_raw_cols));
  NodeStruct* row_ptrs = acc_row_ptr.ptr(rect_row_ptr);
  V_ID* in_vtxs = acc_in_vtx.ptr(rect_in_vtx);
  EdgeStruct* col_idxs = acc_col_idx.ptr(rect_col_idx);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  const E_ID* raw_rows = acc_raw_rows.ptr(rect_raw_rows);
  const V_ID* raw_cols = acc_raw_cols.ptr(rect_raw_cols);
  V_ID rowLeft = rect_row_ptr.lo[0], rowRight = rect_row_ptr.hi[0];
  E_ID colLeft = rect_col_idx.lo[0], colRight = rect_col_idx.hi[0];
  std::vector<V_ID> edges(colRight - colLeft + 1);
  for (E_ID e = 0; e < colRight - colLeft + 1; e++)
    edges[e] = raw_cols[e];
  std::sort(edges.begin(), edges.end());
  V_ID curVtx = edges[0], myInVtx = 0;
  for (E_ID e = 0; e < colRight - colLeft + 1; e++) {
    if (curVtx != edges[e]) {
      edges[myInVtx++] = curVtx;
      curVtx = edges[e];
    }
  }
  edges[myInVtx++] = curVtx;
  checkCUDA(cudaMemcpy(in_vtxs, edges.data(), sizeof(V_ID) * myInVtx,
                       cudaMemcpyHostToDevice));
  // Add raw_weights if regions.size() == 7
  const WeightType* raw_weights = NULL;
  if (regions.size() == 7) {
    const AccessorRO<WeightType, 1> acc_raw_weights(regions[6], FID_DATA);
    Rect<1> rect_raw_weights = runtime->get_index_space_domain(
                               ctx, task->regions[6].region.get_index_space());
    assert(rect_raw_weights == rect_raw_cols);
    assert(acc_raw_weights.accessor.is_dense_arbitrary(rect_raw_weights));
    raw_weights = acc_raw_weights.ptr(rect_raw_weights.lo);
  }
  init_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft,
      row_ptrs, col_idxs, raw_rows, raw_cols, raw_weights);
  checkCUDA(cudaDeviceSynchronize());
  float value = std::sqrt(1.0f / K);
  for (V_ID n = 0; n + rowLeft <= rowRight; n++) {
    for (int i = 0; i < K; i++)
      new_pr[n].v[i] = value;
  }
  GraphPiece piece;
  piece.myInVtxs = myInVtx;
  piece.nv = graph->nv;
  piece.ne = graph->ne;
  // Allocate oldPrFb/newPrFb on the same memory as row_ptr
  std::set<Memory> memFB;
  regions[0].get_memories(memFB);
  assert(memFB.size() == 1);
  assert(memFB.begin()->kind() == Memory::GPU_FB_MEM);
  Realm::MemoryImpl* memImpl =
      Realm::get_runtime()->get_memory_impl(*memFB.begin());
  Realm::Cuda::GPUFBMemory* memFBImpl = (Realm::Cuda::GPUFBMemory*) memImpl;
  off_t offset = memFBImpl->alloc_bytes(sizeof(Vertex) * graph->nv);
  assert(offset >= 0);
  piece.oldPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(sizeof(Vertex) * (rowRight - rowLeft + 1));
  assert(offset >= 0);
  piece.newPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  //checkCUDA(cudaMalloc(&(piece.oldPrFb), sizeof(float) * graph->nv));
  //checkCUDA(cudaMalloc(&(piece.newPrFb), sizeof(float) * (rowRight-rowLeft+1)));
  return piece;
}
