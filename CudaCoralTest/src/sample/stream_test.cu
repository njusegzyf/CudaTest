#include "sample_test.h"

#include <iostream>
#include <algorithm>
#include <memory>
#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../util/cuda_smart_pointer.cuh"
#include "../util/cuda_stream_helper.cuh"

static constexpr size_t STREAM_COUNT = 2;
static constexpr size_t THREAD_SIZE = 2048 * 2048;
static constexpr size_t THREADS_PER_BLOCK = 1024;
static constexpr size_t BLOCK_SIZE = (THREAD_SIZE + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
static constexpr size_t FULL_DATA_SIZE = STREAM_COUNT * THREAD_SIZE;

namespace cudatest {

  __global__ void stramKernel(int* d_a, int *d_b, int *d_out) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    // if (threadID < THREAD_SIZE) {
    d_out[threadID] = (d_a[threadID] + d_b[threadID]) / STREAM_COUNT;
    // }
  }

  //static cudaError_t submitWorkToStream(int* d_a1, int* d_b1, int* d_c1, int* h_a, int* h_b, int* h_c, cudaStream_t& stream, size_t i) {
  //  cudaError_t cudaStatus = cudaMemcpyAsync(d_a1, h_a + i * THREAD_SIZE, THREAD_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream);
  //  if (cudaSuccess != cudaStatus) {
  //    printf("cudaMemcpyAsync a failed!\n");
  //    return cudaStatus;
  //  }

  //  cudaStatus = cudaMemcpyAsync(d_b1, h_b + i * THREAD_SIZE, THREAD_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream);
  //  if (cudaSuccess != cudaStatus) {
  //    printf("cudaMemcpyAsync b failed!\n");
  //    return cudaStatus;
  //  }

  //  stramKernel << <BLOCK_SIZE, THREADS_PER_BLOCK, 0, stream >> > (d_a1, d_b1, d_c1);
  //  cudaStatus = cudaGetLastError();
  //  if (cudaSuccess != cudaStatus) { return cudaStatus; }

  //  cudaStatus = cudaMemcpyAsync(h_c + i * THREAD_SIZE, d_c1, THREAD_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream);
  //  return cudaStatus;
  //}

  cudaError_t runStreamTest() {
    cudaError_t cudaStatus = cudaSuccess;

    // 获取设备属性  
    cudaDeviceProp prop;
    int deviceID;
    cudaStatus = cudaGetDevice(&deviceID);
    cudaStatus = cudaGetDeviceProperties(&prop, deviceID);

    // 检查设备是否支持重叠功能  
    if (!prop.deviceOverlap) {
      printf("No device will handle overlaps. so no speed up from stream.\n");
      return cudaStatus;
    }

    // 启动计时器  
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaStatus = cudaEventCreate(&start);
    cudaStatus = cudaEventCreate(&stop);
    cudaStatus = cudaEventRecord(start); // cudaEventRecord(start, 0);

    using cudautil::makeUniqueHostMemory;
    using cudautil::makeUniqueDeviceMemory;

    // 在GPU上分配内存  
    //auto d_a1 = makeUniqueDeviceMemory<int>(THREAD_SIZE);
    //auto d_b1 = makeUniqueDeviceMemory<int>(THREAD_SIZE);
    //auto d_c1 = makeUniqueDeviceMemory<int>(THREAD_SIZE);

    //auto d_a2 = makeUniqueDeviceMemory<int>(THREAD_SIZE);
    //auto d_b2 = makeUniqueDeviceMemory<int>(THREAD_SIZE);
    //auto d_c2 = makeUniqueDeviceMemory<int>(THREAD_SIZE);

    //using UniquePtrInt = std::unique_ptr<int, std::function<void(int*)>>;
    //UniquePtrInt d_as[] = { std::move(d_a1), std::move(d_a2) };
    //UniquePtrInt d_bs[] = { std::move(d_b1), std::move(d_b2) };
    //UniquePtrInt d_cs[] = { std::move(d_c1), std::move(d_c2) };

    using UniquePtrInt = std::unique_ptr<int, std::function<void(int*)>>;
    UniquePtrInt d_as[] = { makeUniqueDeviceMemory<int>(THREAD_SIZE),  makeUniqueDeviceMemory<int>(THREAD_SIZE) };
    UniquePtrInt d_bs[] = { makeUniqueDeviceMemory<int>(THREAD_SIZE),  makeUniqueDeviceMemory<int>(THREAD_SIZE) };
    UniquePtrInt d_cs[] = { makeUniqueDeviceMemory<int>(THREAD_SIZE),  makeUniqueDeviceMemory<int>(THREAD_SIZE) };

    // 在CPU上分配页锁定内存  
    auto h_a = makeUniqueHostMemory<int>(FULL_DATA_SIZE);
    auto h_b = makeUniqueHostMemory<int>(FULL_DATA_SIZE);
    auto h_c = makeUniqueHostMemory<int>(FULL_DATA_SIZE);

    // 主机上的内存赋值  
    for (auto i = 0; i < STREAM_COUNT * THREAD_SIZE; i++) {
      h_a.get()[i] = i;
      h_b.get()[i] = FULL_DATA_SIZE - i;
    }

    // 创建 CUDA 流  
    cudaStream_t streams[STREAM_COUNT];
    for (auto& stream : streams) {
      cudaStatus = cudaStreamCreate(&stream);
    }

    // submit work to different streams
    for (size_t i = 0; i < STREAM_COUNT; ++i) {
      cudaStatus = cudautil::runKernelInStreamAsnyc<int, int, int>(
        h_a.get() + i * THREAD_SIZE, d_as[i].get(), THREAD_SIZE,
        h_b.get() + i * THREAD_SIZE, d_bs[i].get(), THREAD_SIZE,
        h_c.get() + i * THREAD_SIZE, d_cs[i].get(), THREAD_SIZE,
        // the lambda is of type std::function<cudaError_t(TIn1*, TIn2*, TOut*, dim3, dim3, size_t, cudaStream_t&)> startKernelFunction
        [](int* d_a, int* d_b, int* d_c, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
        stramKernel << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_a, d_b, d_c);
        return cudaGetLastError();
      }, dim3(BLOCK_SIZE), dim3(THREADS_PER_BLOCK), 0, streams[i]);

      //cudaStatus = submitWorkToStream(d_as[i].get(), d_bs[i].get(), d_cs[i].get(),
      //                                h_a.get(), h_b.get(), h_c.get(),
      //                                streams[i],
      //                                i);

      if (cudaSuccess != cudaStatus) {
        printf("Failed to start kernel.\n");
        return cudaStatus;
      }
    }

    // 等待Stream流执行完成
    for (auto& stream : streams) {
      cudaStatus = cudaStreamSynchronize(stream);
    }

    cudaStatus = cudaEventRecord(stop); // cudaEventRecord(stop, 0);
    cudaStatus = cudaEventSynchronize(stop);
    cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "消耗时间： " << elapsedTime << std::endl;

    // check results
    bool result = std::all_of(h_c.get(), h_c.get() + FULL_DATA_SIZE, [](int x) { return THREAD_SIZE == x; });
    //std::cout << h_a.get()[0] << ',' << h_a.get()[THREAD_SIZE] << ',' << h_a.get()[THREAD_SIZE + 1] << std::endl;
    //std::cout << h_b.get()[0] << ',' << h_b.get()[THREAD_SIZE] << ',' << h_b.get()[THREAD_SIZE + 1] << std::endl;
    //std::cout << h_c.get()[0] << ',' << h_c.get()[THREAD_SIZE] << ',' << h_c.get()[THREAD_SIZE + 1] << std::endl;
    std::cout << "Result check： " << result << std::endl;

    getchar();

    // free stream (memory is freed by unique pointers)
    for (auto& stream : streams) {
      cudaStatus = cudaStreamDestroy(stream);
    }

    return cudaStatus;
  }
}
