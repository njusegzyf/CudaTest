#pragma once

#include <functional>

#include <cuda_runtime.h>

// #include "cuda_util.cuh"

namespace cudautil {

  template<typename TIn, typename TOut>
  cudaError_t runKernelInStreamAsnyc(TIn* h_in, TIn* d_in, size_t inputSize,
                                     TOut* h_out, TOut* d_out, size_t outputSize,
                                     std::function<cudaError_t(TIn*, TOut*, dim3, dim3, size_t, cudaStream_t&)> startKernelFunction,
                                     dim3 blockSize, dim3 threadPerBlock, size_t sharedMemSize, cudaStream_t& stream) {

    // copy input from host to device
    cudaError_t status = cudaMemcpyAsync(d_in, h_in, inputSize * sizeof(TIn), cudaMemcpyHostToDevice, stream);
    if (cudaSuccess != status) { return status; }

    // start kernel
    status = startKernelFunction(d_in, d_out, blockSize, threadPerBlock, sharedMemSize, stream);
    if (cudaSuccess != status) { return status; }

    // copy output from device to host
    status = cudaMemcpyAsync(h_out, d_out, outputSize * sizeof(TOut), cudaMemcpyDeviceToHost, stream);
    return status;
  }

  template<typename TIn1, typename TIn2, typename TOut>
  cudaError_t runKernelInStreamAsnyc(TIn1* h_in1, TIn1* d_in1, size_t input1Size,
                                     TIn2* h_in2, TIn2* d_in2, size_t input2Size,
                                     TOut* h_out, TOut* d_out, size_t outputSize,
                                     std::function<cudaError_t(TIn1*, TIn2*, TOut*, dim3, dim3, size_t, cudaStream_t&)> startKernelFunction,
                                     dim3 blockSize, dim3 threadPerBlock, size_t sharedMemSize, cudaStream_t& stream) {

    // copy input1 from host to device
    cudaError_t status = cudaMemcpyAsync(d_in1, h_in1, input1Size * sizeof(TIn1), cudaMemcpyHostToDevice, stream);
    if (cudaSuccess != status) { return status; }

    // copy input2 from host to device
    status = cudaMemcpyAsync(d_in2, h_in2, input2Size * sizeof(TIn2), cudaMemcpyHostToDevice, stream);
    if (cudaSuccess != status) { return status; }

    // start kernel
    status = startKernelFunction(d_in1, d_in2, d_out, blockSize, threadPerBlock, sharedMemSize, stream);
    if (cudaSuccess != status) { return status; }

    // copy output from device to host
    status = cudaMemcpyAsync(h_out, d_out, outputSize * sizeof(TOut), cudaMemcpyDeviceToHost, stream);
    return status;
  }

  template<typename TIn1, typename TIn2, typename TIn3, typename TOut>
  cudaError_t runKernelInStreamAsnyc(TIn1* h_in1, TIn1* d_in1, size_t input1Size,
                                     TIn2* h_in2, TIn2* d_in2, size_t input2Size,
                                     TIn2* h_in3, TIn2* d_in3, size_t input3Size,
                                     TOut* h_out, TOut* d_out, size_t outputSize,
                                     std::function<cudaError_t(TIn1*, TIn2*, TIn3*, TOut*, dim3, dim3, size_t, cudaStream_t&)> startKernelFunction,
                                     dim3 blockSize, dim3 threadPerBlock, size_t sharedMemSize, cudaStream_t& stream) {

    // copy input1 from host to device
    cudaError_t status = cudaMemcpyAsync(d_in1, h_in1, input1Size * sizeof(TIn1), cudaMemcpyHostToDevice, stream);
    if (cudaSuccess != status) { return status; }

    // copy input2 from host to device
    status = cudaMemcpyAsync(d_in2, h_in2, input2Size * sizeof(TIn2), cudaMemcpyHostToDevice, stream);
    if (cudaSuccess != status) { return status; }

    // copy input3 from host to device
    status = cudaMemcpyAsync(d_in3, h_in3, input3Size * sizeof(TIn3), cudaMemcpyHostToDevice, stream);
    if (cudaSuccess != status) { return status; }

    // start kernel
    status = startKernelFunction(d_in1, d_in2, d_in3, d_out, blockSize, threadPerBlock, sharedMemSize, stream);
    if (cudaSuccess != status) { return status; }

    // copy output from device to host
    status = cudaMemcpyAsync(h_out, d_out, outputSize * sizeof(TOut), cudaMemcpyDeviceToHost, stream);
    return status;
  }
}
