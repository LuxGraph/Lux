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

#include "../graph.h"
#include "../cuda_helper.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__
void load_kernel(V_ID my_in_vtxs,
                 const V_ID* in_vtxs,
                 float* old_pr_fb,
                 const float* old_pr_zc)
{
  for (V_ID i = blockIdx.x * blockDim.x + threadIdx.x; i < my_in_vtxs;
       i+= blockDim.x * gridDim.x)
  {
    V_ID vtx = in_vtxs[i];
    float my_pr = old_pr_zc[vtx];
    cub::ThreadStore<cub::STORE_CG>(old_pr_fb + vtx, my_pr);
  }
}

__global__
void pr_kernel(V_ID rowLeft,
               V_ID rowRight,
               E_ID colLeft,
               float initRank,
               const NodeStruct* row_ptrs,
               const EdgeStruct* col_idxs,
               float* old_pr_fb,
               float* new_pr_fb)
{
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ float pr[CUDA_NUM_THREADS];
  __shared__ E_ID blkColStart;
  for (V_ID blkRowStart = blockIdx.x * blockDim.x + rowLeft; blkRowStart <= rowRight;
       blkRowStart += blockDim.x * gridDim.x)
  {
    E_ID myNumEdges = 0, scratchOffset, totalNumEdges = 0;
    V_ID myDegree = 0;
    V_ID curVtx = blkRowStart + threadIdx.x;
    if (curVtx <= rowRight)
    {
      NodeStruct ns = row_ptrs[curVtx - rowLeft];
      E_ID start_col_idx, end_col_idx = ns.index;
      myDegree = ns.degree;
      if (curVtx == rowLeft)
        start_col_idx = colLeft;
      else
        start_col_idx = row_ptrs[curVtx - rowLeft - 1].index;
      myNumEdges = end_col_idx - start_col_idx;
      if (threadIdx.x == 0)
        blkColStart = start_col_idx;
    }
    pr[threadIdx.x] = 0;

    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
    E_ID done = 0;
    while (totalNumEdges > 0)
    {
      if (threadIdx.x < totalNumEdges)
      {
        EdgeStruct es = col_idxs[blkColStart + done + threadIdx.x - colLeft];
        float src_pr = old_pr_fb[es.src];
        atomicAdd(new_pr_fb + es.dst - blkRowStart, src_pr);
      }
      done += CUDA_NUM_THREADS;
      totalNumEdges -= (totalNumEdges > CUDA_NUM_THREADS) ? 
                       CUDA_NUM_THREADS : totalNumEdges;
    }
    __syncthreads();
    float my_pr = initRank + ALPHA * pr[threadIdx.x];
    if (myDegree != 0)
      my_pr = my_pr / myDegree;
    new_pr_fb[curVtx - rowLeft] = my_pr;
  }
}

/*static*/
void pagerank_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 5);
  assert(task->regions.size() == 5);
  const GraphPiece *piece = (GraphPiece*) task->local_args;

  const AccessorRO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorRO<V_ID, 1> acc_in_vtx(regions[1], FID_DATA);
  const AccessorRO<EdgeStruct, 1> acc_col_idx(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_old_pr(regions[3], FID_DATA);
  const AccessorRW<float, 1> acc_new_pr(regions[4], FID_DATA);
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
  const float* old_pr = acc_old_pr.ptr(rect_old_pr);
  float* new_pr = acc_new_pr.ptr(rect_new_pr);
  V_ID rowLeft = rect_row_ptr.lo[0], rowRight = rect_row_ptr.hi[0];
  E_ID colLeft = rect_col_idx.lo[0], colRight = rect_col_idx.hi[0];

  load_kernel<<<GET_BLOCKS(piece->myInVtxs), CUDA_NUM_THREADS>>>(
      piece->myInVtxs, in_vtxs, piece->oldPrFb, old_pr);     
  pr_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft, (1 - ALPHA) / piece->nv,
      row_ptrs, col_idxs, piece->oldPrFb, piece->newPrFb);
  // Need to copy results back to new_pr
  cudaDeviceSynchronize();
  checkCUDA(cudaMemcpy(new_pr, piece->newPrFb,
            (rowRight - rowLeft + 1) * sizeof(float), cudaMemcpyDeviceToHost));
}

__global__
void init_kernel(V_ID rowLeft, V_ID rowRight, E_ID colLeft,
                 NodeStruct* row_ptrs, EdgeStruct* col_idxs,
                 const E_ID* raw_rows,
                 const V_ID* degrees,
                 const V_ID* raw_cols)
{
  for (V_ID n = blockIdx.x * blockDim.x + threadIdx.x;
       n + rowLeft <= rowRight; n += blockDim.x * gridDim.x)
  {
    E_ID startColIdx, endColIdx = raw_rows[n];
    if (n == 0)
      startColIdx = colLeft;
    else
      startColIdx = raw_rows[n - 1];
    row_ptrs[n].index = endColIdx;
    if (degrees != NULL)
      row_ptrs[n].degree = degrees[n];
    for (E_ID e = startColIdx; e < endColIdx; e++)
    {
      col_idxs[e].src = raw_cols[e - colLeft];
      col_idxs[e].dst = n + rowLeft;
    }
  }
}

GraphPiece init_task_impl(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 6 || regions.size() == 7);
  assert(task->regions.size() == 6 || regions.size() == 7);
  const Graph *graph = (Graph*) task->args;
  const AccessorWO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorWO<V_ID, 1> acc_in_vtx(regions[1], FID_DATA);
  const AccessorWO<EdgeStruct, 1> acc_col_idx(regions[2], FID_DATA);
  const AccessorWO<float, 1> acc_new_pr(regions[3], FID_DATA);
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
  float* new_pr = acc_new_pr.ptr(rect_new_pr);
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
  // Add degree if regions.size() == 7
  const V_ID *degrees = NULL;
  if (regions.size() == 7) {
    const AccessorRO<V_ID, 1> acc_degrees(regions[6], FID_DATA);
    Rect<1> rect_degrees = runtime->get_index_space_domain(
                               ctx, task->regions[6].region.get_index_space());
    assert(acc_degrees.accessor.is_dense_arbitrary(rect_degrees));
    degrees = acc_degrees.ptr(rect_degrees.lo);
  }
  init_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft, row_ptrs, col_idxs, raw_rows, degrees, raw_cols);
  checkCUDA(cudaDeviceSynchronize());
  float rank = 1.0f / graph->nv;
  for (V_ID n = 0; n + rowLeft <= rowRight; n++) {
    new_pr[n] = degrees[n] == 0 ? rank : rank / degrees[n];
  }
  GraphPiece piece;
  piece.myInVtxs = myInVtx;
  piece.nv = graph->nv;
  piece.ne = graph->ne;
  checkCUDA(cudaMalloc(&(piece.oldPrFb), sizeof(float) * graph->nv));
  checkCUDA(cudaMalloc(&(piece.newPrFb), sizeof(float) * (rowRight-rowLeft+1)));
  return piece;
}

