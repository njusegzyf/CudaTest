#include "jpf_test.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <chrono>

#include "../util/cuda_util.cuh"
#include "jpf_condition_cuda.cuh"

namespace chrono = std::chrono;
using chrono::system_clock;
using std::cout;
using std::endl;

namespace jpftest = cudatest::jpftest;

// constant for device functions
// __constant__ long longx = 10L;

static constexpr size_t THREAD_SIZE = 2048 * 2048;
static constexpr size_t THREADS_PER_BLOCK = 1024;
static constexpr size_t BLOCK_SIZE = (THREAD_SIZE + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
static constexpr size_t inputArrayByteSize = THREAD_SIZE * sizeof(double);
static constexpr size_t outputArrayByteSize = THREAD_SIZE * sizeof(bool);

namespace cudatest {
  namespace jpftest {

    __global__ void vectorJpf52Test(double *d_x, double *d_y, double *d_z, bool *d_out) {
      /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
      int index = cudautil::computeGlobalId1D1D(); // blockIdx.x * blockDim.x + threadIdx.x;
      d_out[index] = jpftest::jpf52Test(d_x[index], d_y[index], d_z[index]);
    }

    void runJpf52Test() {

      double* h_xs = nullptr;
      double* h_ys = nullptr;
      double* h_zs = nullptr;
      bool* h_outs = nullptr;

      double* d_xs = nullptr;
      double* d_ys = nullptr;
      double* d_zs = nullptr;
      bool* d_outs = nullptr;
      cudaError_t cudaStatus = cudaSuccess;

      // @see http://en.cppreference.com/w/cpp/chrono/duration
      auto cudaStartTime = system_clock::now(); // of type system_clock::time_point

                                                /* allocate space for device copies */
      cudaMalloc(&d_xs, inputArrayByteSize);
      cudaMalloc(&d_ys, inputArrayByteSize);
      cudaMalloc(&d_zs, inputArrayByteSize);
      cudaMalloc(&d_outs, outputArrayByteSize);

      /* allocate space for host copies of a, b, c and setup input values */
      h_xs = static_cast<double*>(malloc(inputArrayByteSize));
      h_ys = static_cast<double*>(malloc(inputArrayByteSize));
      h_zs = static_cast<double*>(malloc(inputArrayByteSize));
      h_outs = static_cast<bool*>(malloc(outputArrayByteSize));

      for (size_t i = 0; i < THREAD_SIZE; i++) {
        h_xs[i] = h_ys[i] = h_zs[i] = double(i);
      }

      auto kernelCopyStartTime = system_clock::now();

      /* copy inputs to device */
      /* fix the parameters needed to copy data to the device */
      cudaMemcpy(d_xs, h_xs, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_ys, h_ys, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_zs, h_zs, inputArrayByteSize, cudaMemcpyHostToDevice);

      auto kernelStartTime = system_clock::now();

      /* launch the kernel on the GPU */
      /* insert the launch parameters to launch the kernel properly using blocks and threads */
      vectorJpf52Test << < BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_xs, d_ys, d_zs, d_outs);

      // Check for any errors launching the kernel
      cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // goto Error;
      }

      // cudaDeviceSynchronize waits for the kernel to finish, and returns
      // any errors encountered during the launch.
      cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        // goto Error;
      }

      auto kernelEndTime = system_clock::now();

      /* copy result back to host */
      /* fix the parameters needed to copy data back to the host */
      cudaMemcpy(h_outs, d_outs, outputArrayByteSize, cudaMemcpyDeviceToHost);

      auto kernelCopyEndTime = system_clock::now();

      // Error:
      /* clean up */
      free(h_xs);
      free(h_ys);
      free(h_zs);
      free(h_outs);
      cudaFree(d_xs);
      cudaFree(d_ys);
      cudaFree(d_zs);
      cudaFree(d_outs);

      if (cudaSuccess == cudaStatus) {
        auto cudaEndTime = system_clock::now();

        cout << "CUDA use: " << chrono::duration_cast<chrono::microseconds>(cudaEndTime - cudaStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel copy and compute use: " << chrono::duration_cast<chrono::microseconds>(kernelCopyEndTime - kernelCopyStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel compute use: " << chrono::duration_cast<chrono::microseconds>(kernelEndTime - kernelStartTime).count() << " microseconds\n";
      }

      getchar();

      return;
    }
  }

  void runJpf52TestFloat() {}
}
