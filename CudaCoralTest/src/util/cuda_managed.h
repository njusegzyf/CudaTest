#pragma once

#include <utility>

#include <cuda_runtime.h>

namespace cudautil {

  class ManagedCudaStream final {
  public:
    cudaStream_t stream;

    ManagedCudaStream(cudaStream_t& st) : stream(st) {}

    ~ManagedCudaStream() {
      if (nullptr != stream) {
        cudaStreamDestroy(stream);
      }
    }

    // disable copy constructor and copy assignment
    ManagedCudaStream(const ManagedCudaStream&) = delete;
    ManagedCudaStream& operator= (const ManagedCudaStream&) = delete;

    // enable move constructor and copy assignment
    ManagedCudaStream(ManagedCudaStream&& other) noexcept : stream(other.stream) {
      other.stream = nullptr;
    }

    ManagedCudaStream& operator= (ManagedCudaStream&& other) noexcept {
      std::swap(stream, other.stream);
      return *this;
    }
  };

  inline ManagedCudaStream createManagedCudaStream() {
    cudaStream_t stream;
    cudaError_t status = cudaStreamCreate(&stream);
    if (cudaSuccess == status) {
      return ManagedCudaStream(stream);
    } else {
      throw status;
    }
  }
}
