#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cudautil {

  #pragma region Compute dim size

  inline __device__ size_t compute1dSize(dim3 dim) {
    return dim.x;
  }

  inline __device__ size_t compute2dSize(dim3 dim) {
    return dim.x * dim.y;
  }

  inline __device__ size_t compute3dSize(dim3 dim) {
    return dim.x * dim.y * dim.z;
  }

  inline __device__ size_t computeSize(dim3 dim) {
    return computeSize(dim);
  }

  // functions for host

  inline __host__ size_t compute1dSizeHost(dim3 dim) {
    return dim.x;
  }

  inline __host__ size_t compute2dSizeHost(dim3 dim) {
    return dim.x * dim.y;
  }

  inline __host__ size_t compute3dSizeHost(dim3 dim) {
    return dim.x * dim.y * dim.z;
  }

  inline __host__ size_t computeSizeHost(dim3 dim) {
    return compute3dSizeHost(dim);
  }

  #pragma endregion

  #pragma region Compute block size

  //inline __device__ size_t computeThreadSizePer1dBlock(dim3 blockD) {
  //  return blockD.x;
  //}

  //inline __device__ size_t computeThreadSizePer2dBlock(dim3 blockD) {
  //  return blockD.x * blockD.y;
  //}

  //inline __device__ size_t computeThreadSizePer3dBlock(dim3 blockD) {
  //  return blockD.x * blockD.y * blockD.z;
  //}

  //inline __device__ size_t computeThreadSizePerBlock(dim3 blockD) {
  //  return computeThreadSizePer3dBlock(blockD);
  //}  

  #pragma endregion


  inline __device__ size_t computeBlockNum(size_t threadSize, dim3 blockD) {
    size_t threadSizePerBlock = computeSize(blockDim); // computeThreadSizePerBlock(blockD);
    return (threadSize + threadSizePerBlock - 1) / threadSizePerBlock;
  }

  inline __host__ size_t computeBlockNumHost(size_t threadSize, dim3 blockD) {
    size_t threadSizePerBlock = computeSizeHost(blockDim); // computeThreadSizePerBlock(blockD);
    return (threadSize + threadSizePerBlock - 1) / threadSizePerBlock;
  }

  #pragma region Compute id in dim

  inline __device__ size_t computeIdIn1d(uint3 id, dim3 dim) {
    return id.x;
  }

  inline __device__ size_t computeIdIn2d(uint3 id, dim3 dim) {
    return id.y * dim.x + id.x;
  }

  inline __device__ size_t computeIdIn3d(uint3 id, dim3 dim) {
    return id.z * (dim.x * dim.y) + id.y * dim.x + id.x;
  }

  inline __device__ size_t computeId(uint3 id, dim3 dim) {
    return computeIdIn3d(id, dim);
  }

  #pragma endregion

  #pragma region Compute id in a block and block id
  //inline __device__ size_t computeLocalIdIn1dBlock() {
  //  return computeIdIn1d(threadIdx, blockDim);
  //  // return threadIdx.x;
  //}

  //inline __device__ size_t computeLocalIdIn2dBlock() {
  //  return computeIdIn2d(threadIdx, blockDim);
  //  // return threadIdx.y * blockDim.x + threadIdx.x;
  //}

  //inline __device__ size_t computeLocalIdIn3dBlock() {
  //  return computeIdIn3d(threadIdx, blockDim);
  //  // return threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  //}

  //inline __device__ size_t computeLocalIdInBlock() {
  //  return computeLocalIdIn3dBlock();
  //}

  //inline __device__ size_t computeBlockIdIn1dGrid() {
  //  return computeIdIn1d(blockIdx, gridDim);
  //  // return blockIdx.x;
  //}

  //inline __device__ size_t computeBlockIdIn2dGrid() {
  //  return computeIdIn2d(blockIdx, gridDim);
  //  // return blockIdx.y * gridDim.x + blockIdx.x;
  //}

  //inline __device__ size_t computeBlockIdIn3dGrid() {
  //  return computeIdIn3d(blockIdx, gridDim);
  //  // return blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
  //}

  //inline __device__ size_t computeBlockIdInGrid() {
  //  return computeBlockIdIn3dGrid();
  //}

  #pragma endregion

  #pragma region Compute global id

    // 情况1：grid划分成1维，block划分为1维。
  __device__ inline size_t computeGlobalId1D1D() {
    // should return block id * block size + thread index in block
    return computeIdIn1d(blockIdx, gridDim) * compute1dSize(blockDim) + computeIdIn1d(threadIdx, blockDim);
    // return computeBlockIdIn1dGrid() * computeThreadSizePer1dBlock(blockDim) + computeLocalIdIn1dBlock();
  }

  // 情况2：grid划分成1维，block划分为2维。
  __device__ inline size_t computeGlobalId1D2D() {
    return computeIdIn1d(blockIdx, gridDim) * compute2dSize(blockDim) + computeIdIn2d(threadIdx, blockDim);
    // return computeBlockIdIn1dGrid() * computeThreadSizePer2dBlock(blockDim) + computeLocalIdIn2dBlock();
  }

  // 情况3：grid划分成1维，block划分为3维。
  __device__ inline size_t computeGlobalId1D3D() {
    return computeIdIn1d(blockIdx, gridDim) * compute3dSize(blockDim) + computeIdIn3d(threadIdx, blockDim);
    // return computeBlockIdIn1dGrid() * computeThreadSizePer3dBlock(blockDim) + computeLocalIdIn3dBlock();
  }

  // 情况4：grid划分成2维，block划分为1维。
  __device__ inline size_t computeGlobalId2D1D() {
    return computeIdIn2d(blockIdx, gridDim) * compute1dSize(blockDim) + computeIdIn1d(threadIdx, blockDim);
    // return computeBlockIdIn2dGrid() * computeThreadSizePer2dBlock(blockDim) + computeLocalIdIn1dBlock();
  }

  // 情况5：grid划分成2维，block划分为2维。
  __device__ inline size_t computeGlobalId2D2D() {
    return computeIdIn2d(blockIdx, gridDim) * compute2dSize(blockDim) + computeIdIn2d(threadIdx, blockDim);
    // return computeBlockIdIn2dGrid() * computeThreadSizePer2dBlock(blockDim) + computeLocalIdIn2dBlock();
  }

  // 情况6：grid划分成2维，block划分为3维。
  __device__ inline size_t computeGlobalId2D3D() {
    return computeIdIn2d(blockIdx, gridDim) * compute3dSize(blockDim) + computeIdIn3d(threadIdx, blockDim);
    // return computeBlockIdIn2dGrid() * computeThreadSizePer3dBlock(blockDim) + computeLocalIdIn3dBlock();
  }

  // 情况7：grid划分成3维，block划分为1维。
  __device__ inline size_t computeGlobalId3D1D() {
    return computeIdIn3d(blockIdx, gridDim) * compute1dSize(blockDim) + computeIdIn1d(threadIdx, blockDim);
    // return computeBlockIdIn3dGrid() * computeThreadSizePer1dBlock(blockDim) + computeLocalIdIn1dBlock();
  }

  // 情况8：grid划分成3维，block划分为2维。
  __device__ inline size_t computeGlobalId3D2D() {
    return computeIdIn3d(blockIdx, gridDim) * compute2dSize(blockDim) + computeIdIn2d(threadIdx, blockDim);
    // return computeBlockIdIn3dGrid() * computeThreadSizePer2dBlock(blockDim) + computeLocalIdIn2dBlock();
  }

  // 情况9：grid划分成3维，block划分为3维。
  __device__ inline size_t computeGlobalId3D3D() {
    return computeIdIn3d(blockIdx, gridDim) * compute3dSize(blockDim) + computeIdIn3d(threadIdx, blockDim);
    // return computeBlockIdIn3dGrid() * computeThreadSizePer3dBlock(blockDim) + computeLocalIdIn3dBlock();
  }

  __device__ inline size_t computeGlobalId() {
    return computeGlobalId3D3D();
  }
  #pragma endregion

}
