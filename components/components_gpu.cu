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

__device__ __forceinline__
void process_edge_dense(const EdgeStruct* colIdxs,
                        Vertex* myNewPrFb,
                        E_ID colIdx,
                        V_ID myRowLeft,
                        Vertex newLabel)
{
  EdgeStruct es = cub::ThreadLoad<cub::LOAD_CG>(colIdxs + colIdx);
  Vertex oldLabel = cub::ThreadLoad<cub::LOAD_CG>(myNewPrFb + es.dst - myRowLeft);
  if (newLabel < oldLabel) {
    atomicMin(myNewPrFb + es.dst - myRowLeft, newLabel);
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
  dstVtx = es.dst;
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
void cc_kernel(V_ID inRowLeft,
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
        start_col_idx = 0;
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
    while (totalNumEdges > 0) {
      __syncthreads();
      if (threadIdx.x < totalNumEdges) {
        while (srcIdx + 1 < CUDA_NUM_THREADS && done + threadIdx.x >= offset[srcIdx + 1])
          srcIdx ++;
        E_ID colIdx = edgeOffset[srcIdx] + done + threadIdx.x
                    - offset[srcIdx] - colLeft;
        if (newDense)
          process_edge_dense(col_idxs, my_new_pr_fb, colIdx,
                             myRowLeft, srcLabels[srcIdx]);
        else {
          E_ID myCnt = 0, offset, totalCnt;
          V_ID dstVtx;
          if (process_edge_sparse(col_idxs, my_old_pr_fb, my_new_pr_fb,
                                  colIdx, myRowLeft, srcLabels[srcIdx], dstVtx))
          {
            myCnt = 1;
          }
          __syncthreads();
          BlockScan(temp_storage).ExclusiveSum(myCnt, offset, totalCnt);
          if (threadIdx.x == 0) {
            queueIdx = atomicAdd(numNodes, (V_ID)totalCnt);
          }
          __syncthreads();
          if (myCnt == 1) {
            if (queueIdx + offset < maxNumNodes)
              newQueue[queueIdx + offset] = dstVtx;
          }
        }
      }
      done += CUDA_NUM_THREADS;
      totalNumEdges -= (totalNumEdges > CUDA_NUM_THREADS)
                       ? CUDA_NUM_THREADS : totalNumEdges;
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
  for (V_ID idx = blockIdx.x * blockDim.x;
       idx * 8 + rowLeft <= rowRight; idx += blockDim.x * gridDim.x)
  {
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
  char* oldBitmap = old_fq_fb + sizeof(FrontierHeader);
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
    newQueue[queueIdx + offset] = curVtx;
  }
}

void push_app_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 6);
  assert(task->regions.size() == 6);
  const Graph* graph = (Graph*) task->args;
  const GraphPiece *piece = (GraphPiece*) task->local_args;

  const AccessorRO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorRO<EdgeStruct, 1> acc_col_idx(regions[1], FID_DATA);
  const AccessorRO<char, 1> acc_old_fq(regions[2], FID_DATA);
  const AccessorWO<char, 1> acc_new_fq(regions[3], FID_DATA);
  const AccessorRO<Vertex, 1> acc_old_pr(regions[4], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[5], FID_DATA);
  Rect<1> rect_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[1].region.get_index_space());
  Rect<1> rect_old_fq = runtime->get_index_space_domain(
                            ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_new_fq = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_old_pr = runtime->get_index_space_domain(
                            ctx, task->regions[4].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[5].region.get_index_space());
  assert(acc_row_ptr.accessor.is_dense_arbitrary(rect_row_ptr));
  assert(acc_col_idx.accessor.is_dense_arbitrary(rect_col_idx));
  assert(acc_old_fq.accessor.is_dense_arbitrary(rect_old_fq));
  assert(acc_new_fq.accessor.is_dense_arbitrary(rect_new_fq));
  assert(acc_old_pr.accessor.is_dense_arbitrary(rect_old_pr));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  const NodeStruct* row_ptrs = acc_row_ptr.ptr(rect_row_ptr);
  const EdgeStruct* col_idxs = acc_col_idx.ptr(rect_col_idx);
  const char* old_fq = acc_old_fq.ptr(rect_old_fq);
  char* new_fq = acc_new_fq.ptr(rect_new_fq);
  const Vertex* old_pr = acc_old_pr.ptr(rect_old_pr);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  V_ID rowLeft = rect_new_pr.lo[0], rowRight = rect_new_pr.hi[0];
  E_ID colLeft = rect_col_idx.lo[0], colRight = rect_col_idx.hi[0];
  V_ID fqLeft = rect_new_fq.lo[0], fqRight = rect_new_fq.hi[0];
  // Copy piece->newPrFb to piece->oldPrFb
  checkCUDA(cudaMemcpy(piece->oldPrFb, piece->newPrFb,
                       sizeof(Vertex) * (rowRight - rowLeft + 1),
                       cudaMemcpyDeviceToDevice));
  // Decide whether we should use sparse/dense frontier
  int denseParts = 0, sparseParts = 0;
  for (int i = 0; i < graph->numParts; i++) {
    FrontierHeader* header = (FrontierHeader*)(old_fq + graph->fqLeft[i]);
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
  for (int i = 0; i < graph->numParts; i ++) {
    FrontierHeader* old_header = (FrontierHeader*)(old_fq + graph->fqLeft[i]);
    if (old_header->type == FrontierHeader::DENSE_BITMAP) {
      checkCUDA(cudaMemcpyAsync(piece->oldFqFb + graph->fqLeft[i],
                    old_fq + graph->fqLeft[i],
                    (graph->rowRight[i] - graph->rowLeft[i]) / 8 + 1
                    + sizeof(FrontierHeader),
                    cudaMemcpyHostToDevice, piece->streams[i]));
      int numBlocks = GET_BLOCKS(graph->rowRight[i] - graph->rowLeft[i] + 1);
      cc_kernel<<<numBlocks, CUDA_NUM_THREADS, 0, piece->streams[i]>>>(
          graph->rowLeft[i], graph->rowRight[i], rowLeft, colLeft,
          row_ptrs, col_idxs, (char*)old_header, piece->newFqFb, old_pr,
          piece->oldPrFb, piece->newPrFb, true/*old_dense*/, denseFq, maxNumNodes);
    } else if (old_header->type == FrontierHeader::SPARSE_QUEUE) {
      checkCUDA(cudaMemcpyAsync(piece->oldFqFb + graph->fqLeft[i],
                    old_fq + graph->fqLeft[i],
                    old_header->numNodes * sizeof(V_ID) + sizeof(FrontierHeader),
                    cudaMemcpyHostToDevice, piece->streams[i]));
      int numBlocks = GET_BLOCKS(old_header->numNodes);
      cc_kernel<<<numBlocks, CUDA_NUM_THREADS, 0, piece->streams[i]>>>(
          0, old_header->numNodes - 1, rowLeft, colLeft,
          row_ptrs, col_idxs, (char*)old_header, piece->newFqFb, old_pr,
          piece->oldPrFb, piece->newPrFb, false/*old_dense*/, denseFq, maxNumNodes);
    } else {
      // Must be either dense or sparse frontier queue
      assert(false);
    }
  }
  checkCUDA(cudaDeviceSynchronize());
  if (denseFq) {
    int numBlocks = GET_BLOCKS((rowRight - rowLeft) / 8 + 1);
    bitmap_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
        rowLeft, rowRight, piece->newFqFb, piece->oldPrFb, piece->newPrFb);
    checkCUDA(cudaDeviceSynchronize());
    FrontierHeader* header = (FrontierHeader*) piece->newFqFb;

    if (header->numNodes < maxNumNodes) {
      // copy piece->newFqFb to piece->oldFqFb
      checkCUDA(cudaMemcpy(piece->oldFqFb, piece->newFqFb,
                           fqRight - fqLeft + 1,
                           cudaMemcpyDeviceToDevice));
      numBlocks = GET_BLOCKS(rowRight - rowLeft + 1);
      denseFq = false;
      header->numNodes = 0;
      convert_d2s_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
          rowLeft, rowRight, piece->oldFqFb, piece->newFqFb);
    }
  } else {
    FrontierHeader* header = (FrontierHeader*) piece->newFqFb;
    V_ID numNodes = header->numNodes;
    if (numNodes >= maxNumNodes) {
      denseFq = true;
      int numBlocks = GET_BLOCKS((rowRight - rowLeft) / 8 + 1);
      bitmap_kernel<<<numBlocks, CUDA_NUM_THREADS>>>(
          rowLeft, rowRight, piece->newFqFb, piece->oldPrFb, piece->newPrFb);
    }
  }
  // Copy piece->newFqFb to new_fq
  // Copy piece->newPrFb to new_pr
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaMemcpy(new_fq, piece->newFqFb,
                       rect_new_fq.hi[0] - rect_new_fq.lo[0] + 1,
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(new_pr, piece->newPrFb,
                       (rowRight - rowLeft + 1) * sizeof(Vertex),
                       cudaMemcpyDeviceToHost));
  newFqHeader->type = denseFq ? FrontierHeader::DENSE_BITMAP
                              : FrontierHeader::SPARSE_QUEUE;
}

static inline bool compareLess(const EdgeStruct& a, const EdgeStruct& b)
{
  return a.src < b.src;
}

GraphPiece push_init_task_impl(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
  assert(regions.size() == 6);
  assert(task->regions.size() == 6);
  const Graph *graph = (Graph*) task->args;
  const AccessorWO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorWO<EdgeStruct, 1> acc_col_idx(regions[1], FID_DATA);
  const AccessorWO<char, 1> acc_frontier(regions[2], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[3], FID_DATA);
  const AccessorRO<E_ID, 1> acc_raw_rows(regions[4], FID_DATA);
  const AccessorRO<V_ID, 1> acc_raw_cols(regions[5], FID_DATA);

  Rect<1> rect_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[1].region.get_index_space());
  Rect<1> rect_frontier = runtime->get_index_space_domain(
                              ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_raw_rows = runtime->get_index_space_domain(
                              ctx, task->regions[4].region.get_index_space());
  Rect<1> rect_raw_cols = runtime->get_index_space_domain(
                              ctx, task->regions[5].region.get_index_space());

  assert(acc_row_ptr.accessor.is_dense_arbitrary(rect_row_ptr));
  assert(acc_col_idx.accessor.is_dense_arbitrary(rect_col_idx));
  assert(acc_frontier.accessor.is_dense_arbitrary(rect_frontier));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  assert(acc_raw_rows.accessor.is_dense_arbitrary(rect_raw_rows));
  assert(acc_raw_cols.accessor.is_dense_arbitrary(rect_raw_cols));
  NodeStruct* row_ptrs = acc_row_ptr.ptr(rect_row_ptr);
  EdgeStruct* col_idxs = acc_col_idx.ptr(rect_col_idx);
  char* frontier = acc_frontier.ptr(rect_frontier);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  const E_ID* raw_rows = acc_raw_rows.ptr(rect_raw_rows);
  const V_ID* raw_cols = acc_raw_cols.ptr(rect_raw_cols);
  V_ID rowLeft = rect_raw_rows.lo[0], rowRight = rect_raw_rows.hi[0];
  E_ID colLeft = rect_col_idx.lo[0], colRight = rect_col_idx.hi[0];
  std::vector<EdgeStruct> edges(colRight - colLeft + 1);
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
  assert(graph->nv == rect_row_ptr.hi[0] - rect_row_ptr.lo[0] + 1);
  // Allocate nodes on the same memory as new_pr
  std::set<Memory> memZC;
  regions[3].get_memories(memZC);
  assert(memZC.size() == 1);
  assert(memZC.begin()->kind() == Memory::Z_COPY_MEM);
  Realm::MemoryImpl* memImpl =
    Realm::get_runtime()->get_memory_impl(*memZC.begin());
  Realm::LocalCPUMemory* memZCImpl = (Realm::LocalCPUMemory*) memImpl;
  off_t offset = memZCImpl->alloc_bytes(sizeof(NodeStruct) * graph->nv);
  NodeStruct* nodes = (NodeStruct*) memZCImpl->get_direct_ptr(offset, 0);
  E_ID cur = colLeft;
  for (V_ID n = 0; n < graph->nv; n++) {
    while ((cur <= colRight ) && (edges[cur - colLeft].src <= n))
      cur ++;
    nodes[n].index = cur;
  }
  checkCUDA(cudaMemcpy(row_ptrs, nodes, sizeof(NodeStruct) * graph->nv,
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(col_idxs, edges.data(),
                       sizeof(EdgeStruct) * (colRight - colLeft + 1),
                       cudaMemcpyHostToDevice));
  memZCImpl->free_bytes(offset, sizeof(NodeStruct) * graph->nv);
  FrontierHeader* header = (FrontierHeader*) frontier;
  header->type = FrontierHeader::DENSE_BITMAP;
  header->numNodes = 0;
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
  piece.oldPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(sizeof(Vertex) * (rowRight - rowLeft + 1));
  piece.newPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(graph->frontierSize);
  piece.oldFqFb = (char*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(rect_frontier.hi[0] - rect_frontier.lo[0] + 1);
  piece.newFqFb = (char*) memFBImpl->get_direct_ptr(offset, 0);
  // Initialize newPrFb
  checkCUDA(cudaMemcpy(piece.newPrFb, new_pr,
                       sizeof(Vertex) * (rowRight - rowLeft + 1),
                       cudaMemcpyHostToDevice));
  for (int i = 0; i < graph->numParts; i++)
    checkCUDA(cudaStreamCreate(&(piece.streams[i])));
  return piece;
}

