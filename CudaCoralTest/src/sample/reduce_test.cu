#include "sample_test.h"

#include <stdio.h>

#include <functional>
#include <memory>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../util/cuda_smart_pointer.cuh"
#include "reduce.cuh"

//#define THREAD_SIZE (2048*2048)
static constexpr size_t THREAD_SIZE = 2048 * 2048;
//#define THREADS_PER_BLOCK 1024
static constexpr size_t THREADS_PER_BLOCK = 1024;
// Note: we will get an error if the BLOCK_SIZE is not a const
static constexpr size_t BLOCK_SIZE = (THREAD_SIZE + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;

// @see http://en.cppreference.com/w/cpp/chrono/duration
using std::chrono::system_clock;
using std::cout;
using std::endl;

namespace cudatest {

  void runReduceTest() {

    cudaError_t cudaStatus;
    int inputArrayByteSize = THREAD_SIZE * sizeof(float);
    int outputArrayByteSize = THREAD_SIZE / THREADS_PER_BLOCK * sizeof(float);

    /* allocate space for host memory */
    // float *h_xs, *h_outs;
    // h_xs = (float*)malloc(inputDoubleArrayByteSize);
    // h_outs = (float*)malloc(outputBoolArrayByteSize);
    auto h_xs = std::make_unique<float[]>(THREAD_SIZE);
    auto h_outs = std::make_unique<float[]>(BLOCK_SIZE);

    for (int i = 0; i < THREAD_SIZE; i++) {
      h_xs[i] = float(i);
    }

    /* allocate space for device copies */
    // float *d_xs, *d_outs;
    // cudaStatus = cudaMalloc((void **)&d_xs, inputDoubleArrayByteSize);
    // cudaStatus = cudaMalloc((void **)&d_outs, outputBoolArrayByteSize);
    // cudaStatus = cudaMemset(d_outs, 0, outputBoolArrayByteSize);
    auto d_xs = cudautil::makeUniqueDeviceMemory<float>(THREAD_SIZE);
    auto d_outs = cudautil::makeUniqueDeviceMemory<float>(BLOCK_SIZE);
    cudaStatus = cudaMemset(d_outs.get(), 0, outputArrayByteSize);

    /* copy inputs to device */
    cudaMemcpy(d_xs.get(), h_xs.get(), inputArrayByteSize, cudaMemcpyHostToDevice);

    // record kernel start time
    auto reduceGlobalStartTime = system_clock::now(); // of type system_clock::time_point

    /* launch the kernel on the GPU */
    cudatest::reduceGlobalKernel << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_xs.get(), d_outs.get());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
      // goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors after launching the kernel.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel, error : %s. \n",
              cudaStatus,
              cudaGetErrorString(cudaStatus));
      // goto Error;
    }

    auto reduceGlobalEndTime = system_clock::now(); // of type system_clock::time_point

                                                    /* copy result back to host */
    cudaMemcpy(h_outs.get(), d_outs.get(), outputArrayByteSize, cudaMemcpyDeviceToHost);

    if (cudaSuccess == cudaStatus) {
      system_clock::time_point cudaEndTime = system_clock::now();
      cout << "CUDA use: " << std::chrono::duration_cast<std::chrono::microseconds>(reduceGlobalEndTime - reduceGlobalStartTime).count() << " microseconds\n";
    }

    /* clean up */
    //free(h_xs);
    //free(h_outs);
    //cudaFree(d_xs);
    //cudaFree(d_outs);
  }
}
