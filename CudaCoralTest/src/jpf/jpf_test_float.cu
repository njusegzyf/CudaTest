#include "jpf_test.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <chrono>

#include "../util/cuda_util.cuh"
#include "../util/cuda_smart_pointer.cuh"
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
using InputType = float;
static constexpr size_t inputArrayByteSize = THREAD_SIZE * sizeof(InputType);
using TestOutputType = bool;
static constexpr size_t testOutputArrayByteSize = THREAD_SIZE * sizeof(TestOutputType);

namespace cudatest {
  namespace jpftest {

    __global__ void kernelJpf52TestFloat(InputType* d_x, InputType* d_y, InputType* d_z, TestOutputType* d_out) {
      int index = cudautil::computeGlobalId1D1D(); // blockIdx.x * blockDim.x + threadIdx.x;
      d_out[index] = jpf52Test(d_x[index], d_y[index], d_z[index]);
    }

    void runJpf52TestFloat() {

      cudaError_t cudaStatus = cudaSuccess;
      auto cudaStartTime = system_clock::now(); // of type system_clock::time_point

      /* allocate space for device copies */
      auto d_xsPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_xs = d_xsPtr.get();
      auto d_ysPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_ys = d_ysPtr.get();
      auto d_zsPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_zs = d_zsPtr.get();
      auto d_outsPtr = cudautil::makeUniqueDeviceMemory<TestOutputType>(THREAD_SIZE);
      TestOutputType* d_outs = d_outsPtr.get();

      /* allocate space for host copies and setup input values */
      auto h_xsPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_xs = h_xsPtr.get();
      auto h_ysPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_ys = h_ysPtr.get();
      auto h_zsPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_zs = h_zsPtr.get();
      auto h_outsPtr = cudautil::makeUniqueHostMemory<TestOutputType>(THREAD_SIZE);
      TestOutputType* h_outs = h_outsPtr.get();

      for (size_t i = 0; i < THREAD_SIZE; i++) {
        h_xs[i] = h_ys[i] = h_zs[i] = InputType(i);
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
      kernelJpf52TestFloat << < BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_xs, d_ys, d_zs, d_outs);

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
      cudaMemcpy(h_outs, d_outs, testOutputArrayByteSize, cudaMemcpyDeviceToHost);

      auto kernelCopyEndTime = system_clock::now();

      if (cudaSuccess == cudaStatus) {
        auto cudaEndTime = system_clock::now();

        cout << "CUDA use: " << chrono::duration_cast<chrono::microseconds>(cudaEndTime - cudaStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel copy and compute use: " << chrono::duration_cast<chrono::microseconds>(kernelCopyEndTime - kernelCopyStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel compute use: " << chrono::duration_cast<chrono::microseconds>(kernelEndTime - kernelStartTime).count() << " microseconds\n";
      }
    }

    void runJpfMultiConditionsTestFloat() {}
  }
}
