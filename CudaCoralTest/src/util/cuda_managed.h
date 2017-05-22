#pragma once

#include <cuda_runtime.h>

namespace cudautil {

  class ManagedCudaStream final {
  public:
    const cudaStream_t stream;

    ManagedCudaStream(cudaStream_t& st) : stream(st) {}

    ~ManagedCudaStream() {
      cudaStreamDestroy(stream);
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
