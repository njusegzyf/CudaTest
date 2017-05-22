#pragma once

#include <cstddef>
#include <memory>
#include <functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cudautil {

  template <typename T>
  T* newHostMemory(std::size_t size) {
    T *ptr;
    cudaError_t mallocStatus = cudaHostAlloc(&ptr, sizeof(T) * size, cudaHostAllocDefault);
    if (cudaSuccess == mallocStatus) {
      return ptr;
    } else {
      throw mallocStatus;
    }
  }

  template <typename T>
  auto makeUniqueHostMemory(std::size_t size) { // the return type is `std::unique_ptr<T, std::function<void(T*)>>`
    return std::unique_ptr<T, std::function<void(T*)>>(
      newHostMemory<T>(size),
      [](T* ptr) { cudaFreeHost(ptr); }
    );
  }

  template <typename T>
  T* newDeviceMemory(std::size_t size) {
    T *ptr;
    cudaError_t mallocStatus = cudaMalloc(&ptr, sizeof(T) * size);
    if (cudaSuccess == mallocStatus) {
      return ptr;
    } else {
      throw mallocStatus;
    }
  }

  // RAII for CUDA memory allocation
  template <typename T>
  auto makeUniqueDeviceMemory(std::size_t size) {
    return std::unique_ptr<T, std::function<void(T*)>>(
      newDeviceMemory<T>(size),
      [](T *ptr) { cudaFree(ptr); }
    );
  }
}
