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

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 512;
const int BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

__global__
void load_kernel(V_ID nv,
                 Vertex* old_pr_fb,
                 const Vertex* old_pr_zc)
{
  for (V_ID i = blockIdx.x * blockDim.x + threadIdx.x; i < nv;
       i+= blockDim.x * gridDim.x)
  {
    old_pr_fb[i] = old_pr_zc[i];
  }
}

__device__ __forceinline__
void process_edge_dense(const EdgeStruct* colIdxs,
                        Vertex* myNewPrFb,
                        E_ID colIdx,
                        V_ID myRowLeft,
                        Vertex newLabel)
{
  EdgeStruct dst = cub::ThreadLoad<cub::LOAD_CG>(colIdxs + colIdx);
  Vertex oldLabel = cub::ThreadLoad<cub::LOAD_CG>(myNewPrFb + dst - myRowLeft);
  if (newLabel < oldLabel) {
    //atomicMin(&myNewPrFb[dst - myRowLeft], newLabel);
    atomicMin(myNewPrFb + dst - myRowLeft, newLabel);
  }
}

__device__ __forceinline__
bool process_edge_sparse(const EdgeStruct* colIdxs,
                         Vertex* myOldPrFb,
                         Vertex* myNewPrFb,
                         E_ID colIdx,
                         V_ID myRowLeft,
                         Vertex newLabel,
                         V_ID &dstVtx)
{
  EdgeStruct es = cub::ThreadLoad<cub::LOAD_CG>(colIdxs + colIdx);
  dstVtx = es;
  Vertex oldLabel = cub::ThreadLoad<cub::LOAD_CG>(myNewPrFb + dstVtx - myRowLeft);
  if (newLabel < oldLabel) {
    Vertex lastLabel = cub::ThreadLoad<cub::LOAD_CG>(myOldPrFb + dstVtx - myRowLeft);
    Vertex actOldLabel = atomicMin(myNewPrFb + dstVtx - myRowLeft, newLabel);
    if (actOldLabel == lastLabel)
      return true;
  }
  return false;
}


__global__
void cc_pull_kernel(V_ID rowLeft,
                    V_ID rowRight,
                    E_ID colLeft,
                    const NodeStruct* row_ptrs,
                    const EdgeStruct2* col_idxs,
                    Vertex* old_pr_fb,
                    Vertex* new_pr_fb)
{
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ E_ID blkColStart;
  for (V_ID blkRowStart = blockIdx.x * blockDim.x + rowLeft; blkRowStart <= rowRight;
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
      new_pr_fb[curVtx - rowLeft] = old_pr_fb[curVtx];
    }

    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
    E_ID done = 0;
    while (totalNumEdges > 0) {
      if (threadIdx.x < totalNumEdges) {
        EdgeStruct2 es = col_idxs[blkColStart + done + threadIdx.x - colLeft];
        Vertex srcLabel = old_pr_fb[es.src];
        atomicMin(new_pr_fb + es.dst - rowLeft, srcLabel);
      }
      done += CUDA_NUM_THREADS;
      totalNumEdges -= (totalNumEdges > CUDA_NUM_THREADS) ? 
                       CUDA_NUM_THREADS : totalNumEdges;
    }
    __syncthreads();
  }
}

__global__
void cc_push_kernel(V_ID inRowLeft,
                    V_ID inRowRight,
                    V_ID myRowLeft,
                    E_ID colLeft,
                    const NodeStruct* row_ptrs,
                    const EdgeStruct* col_idxs,
                    char* old_fq_fb,
                    char* new_fq_fb,
                    const Vertex* in_old_pr_zc,
                    Vertex* my_old_pr_fb,
                    Vertex* my_new_pr_fb,
                    bool oldDense,
                    bool newDense,
                    V_ID maxNumNodes)
{
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ Vertex srcLabels[CUDA_NUM_THREADS];
  __shared__ E_ID offset[CUDA_NUM_THREADS], edgeOffset[CUDA_NUM_THREADS];
  __shared__ int queueIdx;
  char *oldBitmap = NULL;
  V_ID *oldQueue = NULL, *newQueue = NULL;
  V_ID *numNodes = NULL;
  if (!newDense) {
    FrontierHeader* header = (FrontierHeader*) new_fq_fb;
    numNodes = &(header->numNodes);
    newQueue = (V_ID*)(new_fq_fb + sizeof(FrontierHeader));
  }
  if (oldDense) 
    oldBitmap = old_fq_fb + sizeof(FrontierHeader);
  else
    oldQueue = (V_ID*)(old_fq_fb + sizeof(FrontierHeader));
  for (V_ID blkRowStart = blockIdx.x * blockDim.x + inRowLeft;
       blkRowStart <= inRowRight; blkRowStart += blockDim.x * gridDim.x)
  {
    E_ID myOffset = 0, myNumEdges = 0, scratchOffset, totalNumEdges = 0;
    V_ID curIdx = blkRowStart + threadIdx.x;
    if (curIdx <= inRowRight) {
      V_ID curVtx = (oldDense) ? curIdx : oldQueue[curIdx];
      NodeStruct ns = row_ptrs[curVtx];
      E_ID start_col_idx, end_col_idx = ns.index;
      if (curVtx == 0)
        start_col_idx = colLeft;
      else
        start_col_idx = row_ptrs[curVtx - 1].index;
      if (oldDense) {
        V_ID pos = (curVtx - inRowLeft) / 8;
        V_ID off = (curVtx - inRowLeft) % 8;
        if (oldBitmap[pos] & (1<<off)) {
          myNumEdges = end_col_idx - start_col_idx;
          myOffset = start_col_idx;
          srcLabels[threadIdx.x] = in_old_pr_zc[curVtx];
        }
      } else {
        myNumEdges = end_col_idx - start_col_idx;
        myOffset = start_col_idx;
        srcLabels[threadIdx.x] = in_old_pr_zc[curVtx];
      }
    }

    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
    offset[threadIdx.x] = scratchOffset;
    edgeOffset[threadIdx.x] = myOffset;
    __syncthreads();
    E_ID done = 0;
    int srcIdx = 0;
    if (newDense) {
      while (totalNumEdges > 0) {
        if (threadIdx.x < totalNumEdges) {
          while (srcIdx + 1 < CUDA_NUM_THREADS && done + threadIdx.x >= offset[srcIdx + 1])
            srcIdx ++;
          E_ID colIdx = edgeOffset[srcIdx] + done + threadIdx.x
                      - offset[srcIdx] - colLeft;
          process_edge_dense(col_idxs, my_new_pr_fb, colIdx,
                             myRowLeft, srcLabels[srcIdx]);
        }
        done += CUDA_NUM_THREADS;
        totalNumEdges -= (totalNumEdges > CUDA_NUM_THREADS)
                         ? CUDA_NUM_THREADS : totalNumEdges;
      }
    } else {
      while (totalNumEdges > 0) {
        E_ID myCnt = 0, myOffset, totalCnt;
        V_ID dstVtx;
        __syncthreads();
        if (threadIdx.x < totalNumEdges) {
          while (srcIdx + 1 < CUDA_NUM_THREADS && done + threadIdx.x >= offset[srcIdx + 1])
            srcIdx ++;
          E_ID colIdx = edgeOffset[srcIdx] + done + threadIdx.x
                      - offset[srcIdx] - colLeft;
          if (process_edge_sparse(col_idxs, my_old_pr_fb, my_new_pr_fb,
                                   colIdx, myRowLeft, srcLabels[srcIdx], dstVtx))
          {
            myCnt = 1;
          }
        }
        __syncthreads();
        BlockScan(temp_storage).ExclusiveSum(myCnt, myOffset, totalCnt);
        if (threadIdx.x == 0) {
          queueIdx = atomicAdd(numNodes, (V_ID)totalCnt);
        }
        __syncthreads();
        if (myCnt == 1) {
          if (queueIdx + myOffset < maxNumNodes)
            newQueue[queueIdx + myOffset] = dstVtx;
        }
        done += CUDA_NUM_THREADS;
        totalNumEdges -= (totalNumEdges > CUDA_NUM_THREADS)
                         ? CUDA_NUM_THREADS : totalNumEdges;
      }
    }
  }
}

__global__
void bitmap_kernel(V_ID rowLeft,
                   V_ID rowRight,
                   char* new_fq_fb,
                   Vertex* old_pr_fb,
                   Vertex* new_pr_fb)
{
  typedef cub::BlockScan<V_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  FrontierHeader* header = (FrontierHeader*) new_fq_fb;
  char* bitmap = new_fq_fb + sizeof(FrontierHeader);
  V_ID *numNodes = &(header->numNodes);
  for (V_ID blkStart = blockIdx.x * blockDim.x;
       blkStart * 8 + rowLeft <= rowRight; blkStart += blockDim.x * gridDim.x)
  {
    V_ID idx = blkStart + threadIdx.x;
    char bit = 0;
    V_ID cnt = 0, totalCnt = 0;
    for (int i = 0; i < 8; i ++) {
      V_ID curVtx = idx * 8 + rowLeft + i;
      if (curVtx <= rowRight)
        if (old_pr_fb[curVtx - rowLeft] != new_pr_fb[curVtx - rowLeft]) {
          bit = bit | (1 << i);
          cnt = cnt + 1;
        }
    }
    bitmap[idx] = bit;
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(cnt, cnt, totalCnt);
    if (threadIdx.x == 0)
      atomicAdd(numNodes, totalCnt);
    __syncthreads();
  }
}

__global__
void convert_d2s_kernel(V_ID rowLeft,
                        V_ID rowRight,
                        char* old_fq_fb,
                        char* new_fq_fb)
{
  typedef cub::BlockScan<V_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ V_ID queueIdx;
  const char* oldBitmap = (const char*) old_fq_fb + sizeof(FrontierHeader);
  V_ID* newQueue = (V_ID*)(new_fq_fb + sizeof(FrontierHeader));
  FrontierHeader* header = (FrontierHeader*) new_fq_fb;
  V_ID* numNodes = &(header->numNodes);
  for (V_ID blkRowStart = blockIdx.x * blockDim.x + rowLeft;
       blkRowStart <= rowRight; blkRowStart += blockDim.x * gridDim.x)
  {
    V_ID curVtx = blkRowStart + threadIdx.x;
    V_ID cnt = 0, offset, totalCnt;
    if (curVtx <= rowRight) {
      V_ID pos = (curVtx - rowLeft) / 8;
      V_ID off = (curVtx - rowLeft) % 8;
      if (oldBitmap[pos] & (1<<off))
        cnt = 1;
    }
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(cnt, offset, totalCnt);
    if (threadIdx.x == 0)
      queueIdx = atomicAdd(numNodes, totalCnt);
    __syncthreads();
    if (cnt == 1)
      newQueue[queueIdx + offset] = curVtx;
  }
}

__global__
void copy_kernel(V_ID numNodes,
                 V_ID rowLeft,
                 char* new_fq_fb,
                 Vertex* new_pr_fb,
                 Vertex* new_pr_zc)
{
  V_ID* newQueue = (V_ID*)(new_fq_fb + sizeof(FrontierHeader));
  for (V_ID idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < numNodes; idx += blockDim.x * gridDim.x)
  {
    if (idx < numNodes) {
      V_ID curVtx = newQueue[idx];
      new_pr_zc[curVtx - rowLeft] = new_pr_fb[curVtx - rowLeft];
    }
  }
}

V_ID push_app_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 8);
  assert(task->regions.size() == 8);
  const Graph* graph = (Graph*) task->args;
  const GraphPiece *piece = (GraphPiece*) task->local_args;

  const AccessorRO<NodeStruct, 1> acc_pull_row_ptr(regions[0], FID_DATA);
  const AccessorRO<EdgeStruct2, 1> acc_pull_col_idx(regions[1], FID_DATA);
  const AccessorRO<NodeStruct, 1> acc_push_row_ptr(regions[2], FID_DATA);
  const AccessorRO<EdgeStruct, 1> acc_push_col_idx(regions[3], FID_DATA);
  const AccessorRO<char, 1> acc_old_fq(regions[4], FID_DATA);
  const AccessorWO<char, 1> acc_new_fq(regions[5], FID_DATA);
  const AccessorRO<Vertex, 1> acc_old_pr(regions[6], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[7], FID_DATA);
  Rect<1> rect_pull_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_pull_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[1].region.get_index_space());
  Rect<1> rect_push_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_push_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_old_fq = runtime->get_index_space_domain(
                            ctx, task->regions[4].region.get_index_space());
  Rect<1> rect_new_fq = runtime->get_index_space_domain(
                            ctx, task->regions[5].region.get_index_space());
  Rect<1> rect_old_pr = runtime->get_index_space_domain(
                            ctx, task->regions[6].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[7].region.get_index_space());
  assert(acc_pull_row_ptr.accessor.is_dense_arbitrary(rect_pull_row_ptr));
  assert(acc_pull_col_idx.accessor.is_dense_arbitrary(rect_pull_col_idx));
  assert(acc_push_row_ptr.accessor.is_dense_arbitrary(rect_push_row_ptr));
  assert(acc_push_col_idx.accessor.is_dense_arbitrary(rect_push_col_idx));
  assert(acc_old_fq.accessor.is_dense_arbitrary(rect_old_fq));
  assert(acc_new_fq.accessor.is_dense_arbitrary(rect_new_fq));
  assert(acc_old_pr.accessor.is_dense_arbitrary(rect_old_pr));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  assert(rect_push_col_idx == rect_pull_col_idx);
  const NodeStruct* pull_row_ptrs = acc_pull_row_ptr.ptr(rect_pull_row_ptr);
  const EdgeStruct2* pull_col_idxs = acc_pull_col_idx.ptr(rect_pull_col_idx);
  const NodeStruct* push_row_ptrs = acc_push_row_ptr.ptr(rect_push_row_ptr);
  const EdgeStruct* push_col_idxs = acc_push_col_idx.ptr(rect_push_col_idx);
  const char* old_fq = acc_old_fq.ptr(rect_old_fq);
  char* new_fq = acc_new_fq.ptr(rect_new_fq);
  const Vertex* old_pr = acc_old_pr.ptr(rect_old_pr);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  V_ID rowLeft = rect_new_pr.lo[0], rowRight = rect_new_pr.hi[0];
  E_ID colLeft = rect_pull_col_idx.lo[0], colRight = rect_pull_col_idx.hi[0];
  V_ID fqLeft = rect_new_fq.lo[0], fqRight = rect_new_fq.hi[0];

  double ts_start = Realm::Clock::current_time_in_microseconds();
  // Copy piece->newPrFb to piece->oldPrFb by swaping the two pointers
  checkCUDA(cudaMemcpy(piece->oldPrFb, piece->newPrFb,
                       sizeof(Vertex) * (rowRight - rowLeft + 1),
                       cudaMemcpyDeviceToDevice));
  // Decide whether we should use sparse/dense frontier
  int denseParts = 0, sparseParts = 0;
  V_ID oldFqSize = 0;
  for (int i = 0; i < graph->numParts; i++) {
    FrontierHeader* header = (FrontierHeader*)(old_fq + graph->fqLeft[i]);
    oldFqSize += header->numNodes;
    if (header->type == FrontierHeader::DENSE_BITMAP)
      denseParts ++;
    else if (header->type == FrontierHeader::SPARSE_QUEUE)
      sparseParts ++;
    else
      assert(false);
  }
  FrontierHeader* newFqHeader = (FrontierHeader*) new_fq;
  bool denseFq = (denseParts >= sparseParts) ? true : false;
  assert((fqRight - fqLeft + 1 - sizeof(FrontierHeader)) % sizeof(V_ID) == 0);
  V_ID maxNumNodes = (fqRight - fqLeft + 1 - sizeof(FrontierHeader)) / sizeof(V_ID);
  // Initialize new frontier queue
  checkCUDA(cudaMemset(piece->newFqFb, 0, sizeof(FrontierHeader)));
  double cp0 = Realm::Clock::current_time_in_microseconds();
  if (oldFqSize > graph->nv / 16) {
    // If oldFqSize too large, use pull model
    denseFq = true; // Always use dense frontier queue for pull model
    load_kernel<<<GET_BLOCKS(graph->nv), CUDA_NUM_THREADS>>>(
        graph->nv, piece->oldAllPrFb, old_pr);
    cc_pull_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
        rowLeft, rowRight, colLeft, pull_row_ptrs, pull_col_idxs,
        piece->oldAllPrFb, piece->newPrFb);
  } else {
    // Otherwise use push model
    for (int i = 0; i < graph->numParts; i ++) {
      FrontierHeader* old_header = (FrontierHeader*)(old_fq + graph->fqLeft[i]);
      if (old_header->type == FrontierHeader::DENSE_BITMAP) {
        checkCUDA(cudaMemcpyAsync(piece->oldFqFb + graph->fqLeft[i],
                      old_fq + graph->fqLeft[i],
                      (graph->rowRight[i] - graph->rowLeft[i]) / 8 + 1
                      + sizeof(FrontierHeader),
                      cudaMemcpyHostToDevice, piece->streams[i]));
        int numBlocks = GET_BLOCKS(graph->rowRight[i] - graph->rowLeft[i] + 1);
        //printf("push_row_ptrs(%llx) push_col_idxs(%llx) oldFqFb(%llx) newFqFb(%llx) oldPrFb(%llx) newPrFb(%llx) oldPrZC(%llx) oldFqZC(%llx)\n",
        //       push_row_ptrs, push_col_idxs, piece->oldFqFb+graph->fqLeft[i], piece->newFqFb, piece->oldPrFb, piece->newPrFb, old_pr, old_fq);
        cc_push_kernel<<<numBlocks, CUDA_NUM_THREADS, 0, piece->streams[i]>>>(
            graph->rowLeft[i], graph->rowRight[i], rowLeft, colLeft,
            push_row_ptrs, push_col_idxs, piece->oldFqFb + graph->fqLeft[i],
            piece->newFqFb, old_pr, piece->oldPrFb, piece->newPrFb,
            true/*old_dense*/, denseFq, maxNumNodes);
      } else if (old_header->type == FrontierHeader::SPARSE_QUEUE) {
        checkCUDA(cudaMemcpyAsync(piece->oldFqFb + graph->fqLeft[i],
                      old_fq + graph->fqLeft[i],
                      old_header->numNodes * sizeof(V_ID) + sizeof(FrontierHeader),
                      cudaMemcpyHostToDevice, piece->streams[i]));
        int numBlocks = GET_BLOCKS(old_header->numNodes);
        // Avoid launching empty kernel
        if (numBlocks > 0) {
          cc_push_kernel<<<numBlocks, CUDA_NUM_THREADS, 0, piece->streams[i]>>>(
              0, old_header->numNodes - 1, rowLeft, colLeft,
              push_row_ptrs, push_col_idxs, piece->oldFqFb + graph->fqLeft[i],
              piece->newFqFb, old_pr, piece->oldPrFb, piece->newPrFb,
              false/*old_dense*/, denseFq, maxNumNodes);
        }
      } else {
        // Must be either dense or sparse frontier queue
        assert(false);
      }
    }
  }// else if
  checkCUDA(cudaDeviceSynchronize());
  double cp1 = Realm::Clock::current_time_in_microseconds();
  if (denseFq) {
    int numBlocks = GET_BLOCKS((rowRight - rowLeft) / 8 + 1);
    bitmap_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
        rowLeft, rowRight, piece->newFqFb, piece->oldPrFb, piece->newPrFb);
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaMemcpy(newFqHeader, piece->newFqFb, sizeof(FrontierHeader),
                         cudaMemcpyDeviceToHost));
    if (newFqHeader->numNodes < maxNumNodes) {
      // copy piece->newFqFb to piece->oldFqFb
      checkCUDA(cudaMemcpyAsync(piece->oldFqFb, piece->newFqFb,
                           fqRight - fqLeft + 1,
                           cudaMemcpyDeviceToDevice));
      checkCUDA(cudaMemsetAsync(piece->newFqFb, 0, sizeof(FrontierHeader)));
      numBlocks = GET_BLOCKS(rowRight - rowLeft + 1);
      denseFq = false;
      convert_d2s_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
          rowLeft, rowRight, piece->oldFqFb, piece->newFqFb);
      checkCUDA(cudaMemcpyAsync(newFqHeader, piece->newFqFb, sizeof(FrontierHeader),
                           cudaMemcpyDeviceToHost));
    }
  } else {
    checkCUDA(cudaMemcpy(newFqHeader, piece->newFqFb, sizeof(FrontierHeader),
                         cudaMemcpyDeviceToHost));
    if (newFqHeader->numNodes >= maxNumNodes) {
      denseFq = true;
      int numBlocks = GET_BLOCKS((rowRight - rowLeft) / 8 + 1);
      bitmap_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
          rowLeft, rowRight, piece->newFqFb, piece->oldPrFb, piece->newPrFb);
    }
  }
  // Copy piece->newFqFb to new_fq
  // Copy piece->newPrFb to new_pr
  checkCUDA(cudaDeviceSynchronize());
  double cp2 = Realm::Clock::current_time_in_microseconds();
  if (denseFq) {
    checkCUDA(cudaMemcpy(new_fq, piece->newFqFb,
                         rect_new_fq.hi[0] - rect_new_fq.lo[0] + 1,
                         cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(new_pr, piece->newPrFb,
                         (rowRight - rowLeft + 1) * sizeof(Vertex),
                         cudaMemcpyDeviceToHost));
  } else {
    checkCUDA(cudaMemcpy(new_fq + sizeof(FrontierHeader),
                         piece->newFqFb + sizeof(FrontierHeader),
                         sizeof(Vertex) * newFqHeader->numNodes,
                         cudaMemcpyDeviceToHost));
    int numBlocks = GET_BLOCKS(newFqHeader->numNodes);
    if (numBlocks > 0) {
      copy_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
          newFqHeader->numNodes, rowLeft, piece->newFqFb, piece->newPrFb, new_pr);
    }
  }
  checkCUDA(cudaDeviceSynchronize());
  double ts_end = Realm::Clock::current_time_in_microseconds();
  newFqHeader->type = denseFq ? FrontierHeader::DENSE_BITMAP
                              : FrontierHeader::SPARSE_QUEUE;
  if (graph->verbose)
    printf("rowLeft(%u) activeNodes(%u) loadTime(%.0lf) compTime(%.0lf) updateTime(%.0lf)",
           rowLeft, newFqHeader->numNodes, cp0 - ts_start, cp1 - cp0, ts_end - cp1);
  return newFqHeader->numNodes;
  //for (V_ID n = 0; n < 10; n++) printf("oldPr[%u]: %u\n", n + rowLeft, old_pr[n + rowLeft]);
  //for (V_ID n = 0; n < 10; n++) printf("newPr[%u]: %u\n", n + rowLeft, new_pr[n]);
}

__global__
void init_kernel(V_ID rowLeft,
                 V_ID rowRight,
                 E_ID colLeft,
                 NodeStruct* pull_row_ptrs,
                 EdgeStruct2* pull_col_idxs,
                 const E_ID* raw_rows,
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
    pull_row_ptrs[n].index = endColIdx;
    for (E_ID e = startColIdx; e < endColIdx; e++)
    {
      pull_col_idxs[e - colLeft].src = raw_cols[e - colLeft];
      pull_col_idxs[e - colLeft].dst = n + rowLeft;
    }
  }
}

static inline bool compareLess(const EdgeStruct2& a, const EdgeStruct2& b)
{
  return a.src < b.src;
}

GraphPiece push_init_task_impl(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
  assert(regions.size() == 8);
  assert(task->regions.size() == 8);
  const Graph *graph = (Graph*) task->args;
  const AccessorWO<NodeStruct, 1> acc_pull_row_ptr(regions[0], FID_DATA);
  const AccessorWO<EdgeStruct2, 1> acc_pull_col_idx(regions[1], FID_DATA);
  const AccessorWO<NodeStruct, 1> acc_push_row_ptr(regions[2], FID_DATA);
  const AccessorWO<EdgeStruct, 1> acc_push_col_idx(regions[3], FID_DATA);
  const AccessorWO<char, 1> acc_frontier(regions[4], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[5], FID_DATA);
  const AccessorRO<E_ID, 1> acc_raw_rows(regions[6], FID_DATA);
  const AccessorRO<V_ID, 1> acc_raw_cols(regions[7], FID_DATA);

  Rect<1> rect_pull_row_ptr = runtime->get_index_space_domain(ctx,
                                  task->regions[0].region.get_index_space());
  Rect<1> rect_pull_col_idx = runtime->get_index_space_domain(ctx,
                                  task->regions[1].region.get_index_space());
  Rect<1> rect_push_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_push_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_frontier = runtime->get_index_space_domain(
                              ctx, task->regions[4].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[5].region.get_index_space());
  Rect<1> rect_raw_rows = runtime->get_index_space_domain(
                              ctx, task->regions[6].region.get_index_space());
  Rect<1> rect_raw_cols = runtime->get_index_space_domain(
                              ctx, task->regions[7].region.get_index_space());

  assert(acc_pull_row_ptr.accessor.is_dense_arbitrary(rect_pull_row_ptr));
  assert(acc_pull_col_idx.accessor.is_dense_arbitrary(rect_pull_col_idx));
  assert(acc_push_row_ptr.accessor.is_dense_arbitrary(rect_push_row_ptr));
  assert(acc_push_col_idx.accessor.is_dense_arbitrary(rect_push_col_idx));
  assert(acc_frontier.accessor.is_dense_arbitrary(rect_frontier));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  assert(acc_raw_rows.accessor.is_dense_arbitrary(rect_raw_rows));
  assert(acc_raw_cols.accessor.is_dense_arbitrary(rect_raw_cols));
  assert(rect_pull_col_idx == rect_push_col_idx);
  NodeStruct* pull_row_ptrs = acc_pull_row_ptr.ptr(rect_pull_row_ptr);
  EdgeStruct2* pull_col_idxs = acc_pull_col_idx.ptr(rect_pull_col_idx);
  NodeStruct* push_row_ptrs = acc_push_row_ptr.ptr(rect_push_row_ptr);
  EdgeStruct* push_col_idxs = acc_push_col_idx.ptr(rect_push_col_idx);
  char* frontier = acc_frontier.ptr(rect_frontier);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  const E_ID* raw_rows = acc_raw_rows.ptr(rect_raw_rows);
  const V_ID* raw_cols = acc_raw_cols.ptr(rect_raw_cols);
  V_ID rowLeft = rect_raw_rows.lo[0], rowRight = rect_raw_rows.hi[0];
  E_ID colLeft = rect_pull_col_idx.lo[0], colRight = rect_pull_col_idx.hi[0];
  // Init pull_row_ptrs and pull_col_idxs
  init_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft, pull_row_ptrs, pull_col_idxs,
      raw_rows, raw_cols);
  checkCUDA(cudaDeviceSynchronize());
  // Init push_row_ptrs and push_col_idxs
  std::vector<EdgeStruct2> edges(colRight - colLeft + 1);
  E_ID startColIdx = colLeft;
  for (V_ID n = rowLeft; n <= rowRight; n++) {
    E_ID endColIdx = raw_rows[n - rowLeft];
    for (E_ID e = startColIdx; e < endColIdx; e++) {
      edges[e - colLeft].src = raw_cols[e - colLeft];
      edges[e - colLeft].dst = n;
    }
    startColIdx = endColIdx;
  }
  std::sort(edges.begin(), edges.end(), compareLess);
  assert(graph->nv == rect_push_row_ptr.hi[0] - rect_push_row_ptr.lo[0] + 1);
  // Allocate nodes on the same memory as new_pr
  std::set<Memory> memZC;
  regions[5].get_memories(memZC);
  assert(memZC.size() == 1);
  assert(memZC.begin()->kind() == Memory::Z_COPY_MEM);
  Realm::MemoryImpl* memImpl =
    Realm::get_runtime()->get_memory_impl(*memZC.begin());
  Realm::LocalCPUMemory* memZCImpl = (Realm::LocalCPUMemory*) memImpl;
  off_t offset = memZCImpl->alloc_bytes(sizeof(NodeStruct) * graph->nv);
  assert(offset >= 0);
  NodeStruct* nodes = (NodeStruct*) memZCImpl->get_direct_ptr(offset, 0);
  off_t offset2 = memZCImpl->alloc_bytes(sizeof(EdgeStruct) * (colRight - colLeft + 1));
  assert(offset2 >= 0);
  EdgeStruct* dsts = (EdgeStruct*) memZCImpl->get_direct_ptr(offset2, 0);
  E_ID cur = colLeft;
  for (V_ID n = 0; n < graph->nv; n++) {
    while ((cur <= colRight ) && (edges[cur - colLeft].src <= n))
      cur ++;
    nodes[n].index = cur;
  }
  checkCUDA(cudaMemcpy(push_row_ptrs, nodes, sizeof(NodeStruct) * graph->nv,
                       cudaMemcpyHostToDevice));
  for (E_ID e = colLeft; e <= colRight; e++)
    dsts[e - colLeft] = edges[e - colLeft].dst;
  checkCUDA(cudaMemcpy(push_col_idxs, dsts,
                       sizeof(EdgeStruct) * (colRight - colLeft + 1),
                       cudaMemcpyHostToDevice));
  memZCImpl->free_bytes(offset, sizeof(NodeStruct) * graph->nv);
  memZCImpl->free_bytes(offset2, sizeof(EdgeStruct) * (colRight - colLeft + 1));
  FrontierHeader* header = (FrontierHeader*) frontier;
  header->type = FrontierHeader::DENSE_BITMAP;
  header->numNodes = graph->nv;
  char* bitmap = frontier + sizeof(FrontierHeader);
  memset(bitmap, 0xFF, (rowRight - rowLeft) / 8 + 1);
  for (V_ID n = rowLeft; n <= rowRight; n++)
    new_pr[n - rowLeft] = n;
  GraphPiece piece;
  piece.nv = graph->nv;
  piece.ne = graph->ne;
  // Allocate oldPrFb/newPrFb on the same memory as row_ptr
  std::set<Memory> memFB;
  regions[0].get_memories(memFB);
  assert(memFB.size() == 1);
  assert(memFB.begin()->kind() == Memory::GPU_FB_MEM);
  memImpl = Realm::get_runtime()->get_memory_impl(*memFB.begin());
  Realm::Cuda::GPUFBMemory* memFBImpl = (Realm::Cuda::GPUFBMemory*) memImpl;
  offset = memFBImpl->alloc_bytes(sizeof(Vertex) * (rowRight - rowLeft + 1));
  assert(offset >= 0);
  piece.oldPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(sizeof(Vertex) * (rowRight - rowLeft + 1));
  assert(offset >= 0);
  piece.newPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(graph->frontierSize);
  assert(offset >= 0);
  piece.oldFqFb = (char*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(rect_frontier.hi[0] - rect_frontier.lo[0] + 1);
  assert(offset >= 0);
  piece.newFqFb = (char*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(sizeof(Vertex) * graph->nv);
  assert(offset >= 0);
  piece.oldAllPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  // Initialize newPrFb
  checkCUDA(cudaMemcpy(piece.newPrFb, new_pr,
                       sizeof(Vertex) * (rowRight - rowLeft + 1),
                       cudaMemcpyHostToDevice));
  for (int i = 0; i < graph->numParts; i++)
    checkCUDA(cudaStreamCreate(&(piece.streams[i])));
  return piece;
}

