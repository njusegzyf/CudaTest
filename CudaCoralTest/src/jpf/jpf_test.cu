#include "jpf_test.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <vector>
#include <chrono>

#include "../util/cuda_util.cuh"
#include "../util/cuda_smart_pointer.cuh"
#include "../util/cuda_managed.h"
#include "../util/cuda_stream_helper.cuh"
#include "jpf_condition_cuda.cuh"

namespace chrono = std::chrono;
using chrono::system_clock;
using std::cout;
using std::endl;

namespace jpftest = cudatest::jpftest;

// constant for device functions
// __constant__ long longx = 10L;

static constexpr size_t THREADS_PER_BLOCK = 1024; // 256; 
static constexpr size_t THREAD_SIZE = 1024 * THREADS_PER_BLOCK; // 2048 * 2048;
static constexpr size_t BLOCK_SIZE = (THREAD_SIZE + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
using InputType = double;
static constexpr size_t inputArrayByteSize = THREAD_SIZE * sizeof(InputType);
using ConditionOutputType = double;
static constexpr size_t conditionOutputArrayByteSize = THREAD_SIZE * sizeof(ConditionOutputType);
using TestOutputType = bool;
static constexpr size_t testOutputArrayByteSize = THREAD_SIZE * sizeof(TestOutputType);
using ConditionOutputPtrType = decltype(cudautil::makeUniqueDeviceMemory<ConditionOutputType>(THREAD_SIZE));

namespace cudatest {
  namespace jpftest {

    #pragma region runJpf52Test

    __global__ void kernelJpf52Test(InputType *d_x, InputType *d_y, InputType *d_z, TestOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D(); // blockIdx.x * blockDim.x + threadIdx.x;
      d_out[index] = jpf52Test(d_x[index], d_y[index], d_z[index]);
    }

    void runJpf52Test() {

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
      cudaMemcpy(d_xs, h_xs, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_ys, h_ys, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_zs, h_zs, inputArrayByteSize, cudaMemcpyHostToDevice);

      auto kernelStartTime = system_clock::now();

      /* launch the kernel on the GPU */
      /* insert the launch parameters to launch the kernel properly using blocks and threads */
      kernelJpf52Test << < BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_xs, d_ys, d_zs, d_outs);

      // check for any errors launching the kernel
      cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        // goto Error;
      }

      // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
      cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        // goto Error;
      }

      auto kernelEndTime = system_clock::now();

      /* copy result back to host */
      cudaMemcpy(h_outs, d_outs, testOutputArrayByteSize, cudaMemcpyDeviceToHost);

      auto kernelCopyEndTime = system_clock::now();

      if (cudaSuccess == cudaStatus) {
        auto cudaEndTime = system_clock::now();

        cout << "CUDA use: " << chrono::duration_cast<chrono::microseconds>(cudaEndTime - cudaStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel copy and compute use: " << chrono::duration_cast<chrono::microseconds>(kernelCopyEndTime - kernelCopyStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel compute use: " << chrono::duration_cast<chrono::microseconds>(kernelEndTime - kernelStartTime).count() << " microseconds\n";
      }
    }

    #pragma endregion

    #pragma region runJpfMultiConditionsTest

    #pragma region Kernels

    __global__ void kernelJpf52Cond1(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf52Cond1(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf52Cond2(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf52Cond1(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf53Cond1(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf53Cond1(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf53Cond2(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf53Cond2(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf53Cond3(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf53Cond3(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf54Cond1(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf54Cond1(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf54Cond2(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf54Cond2(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf54Cond3(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf54Cond3(d_x[index], d_y[index], d_z[index]);
    }

    __global__ void kernelJpf54Cond4(InputType *d_x, InputType *d_y, InputType *d_z, ConditionOutputType *d_out) {
      int index = cudautil::computeGlobalId1D1D();
      d_out[index] = jpf54Cond4(d_x[index], d_y[index], d_z[index]);
    }

    #pragma endregion

    static constexpr size_t kernelNum = 9;

    /* Runs multi conditions in separate streams. */
    void runJpfMultiConditionsTest() {

      using KernelFunctionType = std::function<cudaError_t(InputType*, InputType*, InputType*, ConditionOutputType*, dim3, dim3, size_t, cudaStream_t&)>;
      KernelFunctionType kernelFunctions[kernelNum] = {
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf52Cond1 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
        return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf52Cond2 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf53Cond1 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf53Cond2 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf53Cond3 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf54Cond1 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf54Cond2 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf54Cond3 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        },
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
          kernelJpf54Cond4 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
          return cudaGetLastError();
        }
      };

      cudaError_t cudaStatus = cudaSuccess;
      auto cudaStartTime = system_clock::now(); // of type system_clock::time_point

      // create CUDA streams
      std::vector<cudautil::ManagedCudaStream> managedStreams;
      for (size_t i = 0; i < kernelNum; ++i) {
        cudaStream_t stream;
        cudaError_t status = cudaStreamCreate(&stream);
        if (cudaSuccess == status) {
          managedStreams.emplace_back(stream);
        } else {
          throw status;
        }
      }

      /* allocate space for device copies */
      // Note: the inputs are shared, while each kernel has its own output
      auto d_xsPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_xs = d_xsPtr.get();
      auto d_ysPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_ys = d_ysPtr.get();
      auto d_zsPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_zs = d_zsPtr.get();
      ConditionOutputPtrType d_outsPtrs[kernelNum];
      for (auto& d_outPtr : d_outsPtrs) {
        d_outPtr = cudautil::makeUniqueDeviceMemory<ConditionOutputType>(THREAD_SIZE);
      }
      ConditionOutputType* d_outss[kernelNum];
      for (size_t i = 0; i < kernelNum; ++i) {
        d_outss[i] = d_outsPtrs[i].get();
      }

      /* allocate space for host copies and setup input values */
      auto h_xsPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_xs = h_xsPtr.get();
      auto h_ysPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_ys = h_ysPtr.get();
      auto h_zsPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_zs = h_zsPtr.get();
      ConditionOutputPtrType h_outsPtrs[kernelNum];
      for (auto& h_outPtr : h_outsPtrs) {
        h_outPtr = cudautil::makeUniqueHostMemory<ConditionOutputType>(THREAD_SIZE);
      }
      ConditionOutputType* h_outss[kernelNum];
      for (size_t i = 0; i < kernelNum; ++i) {
        h_outss[i] = h_outsPtrs[i].get();
      }

      for (size_t i = 0; i < THREAD_SIZE; i++) {
        h_xs[i] = h_ys[i] = h_zs[i] = InputType(i);
      }

      auto kernelCopyStartTime = system_clock::now();

      /* copy shared inputs to device */
      cudaMemcpy(d_xs, h_xs, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_ys, h_ys, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_zs, h_zs, inputArrayByteSize, cudaMemcpyHostToDevice);

      /* launch kernels in separate streams */
      for (size_t i = 0; i < kernelNum; ++i) {
        cudaStatus = cudautil::runKernelInStreamAsnyc<InputType, InputType, InputType, InputType>(
          h_xs, d_xs, 0, // since we have copied the shared inputs, we do not need copy here
          h_ys, d_ys, 0,
          h_zs, d_zs, 0,
          h_outss[i], d_outss[i], THREAD_SIZE,
          kernelFunctions[i],
          dim3(BLOCK_SIZE), dim3(THREADS_PER_BLOCK), 0, managedStreams[i].stream);

        if (cudaSuccess != cudaStatus) {
          printf("Failed to start kernel.\n");
          return;
        }
      }

      // 等待所有 Stream 执行完成
      for (auto& stream : managedStreams) {
        if (cudaSuccess != cudaStreamSynchronize(stream.stream)) {
          printf("Failed to sync stream.\n");
          return;
        }
      }

      auto kernelCopyEndTime = system_clock::now();

      if (cudaSuccess == cudaStatus) {
        auto cudaEndTime = system_clock::now();

        cout << "CUDA use: " << chrono::duration_cast<chrono::microseconds>(cudaEndTime - cudaStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel copy and compute use: " << chrono::duration_cast<chrono::microseconds>(kernelCopyEndTime - kernelCopyStartTime).count() << " microseconds\n";
      }
    }

    #pragma endregion

    #pragma region runJpfMultiRepeatedConditionsTest

    constexpr size_t streamRepeatNum = 64;
    constexpr size_t kernelRepeatNum = 8;

    void runJpfMultiRepeatedConditionsTest() {
      using KernelFunctionType = std::function<cudaError_t(InputType*, InputType*, InputType*, ConditionOutputType*, dim3, dim3, size_t, cudaStream_t&)>;
      KernelFunctionType kernelFunction =
        [](InputType* d_i1, InputType* d_i2, InputType* d_i3, ConditionOutputType* d_o, dim3 blockSize, dim3 threadPreBlock, size_t sharedMemSize, cudaStream_t& stream) {
        for (size_t i = 0; i < kernelRepeatNum; ++i) {
          kernelJpf54Cond4 << < blockSize, threadPreBlock, sharedMemSize, stream >> > (d_i1, d_i2, d_i3, d_o);
        }
        return cudaGetLastError();
      };

      cudaError_t cudaStatus = cudaSuccess;
      auto cudaStartTime = system_clock::now(); // of type system_clock::time_point

      // create CUDA streams
      std::vector<cudautil::ManagedCudaStream> managedStreams;
      for (size_t i = 0; i < streamRepeatNum; ++i) {
        cudaStream_t stream;
        cudaError_t status = cudaStreamCreate(&stream);
        if (cudaSuccess == status) {
          managedStreams.emplace_back(stream);
        } else {
          throw status;
        }
      }

      /* allocate space for device copies */
      // Note: the inputs are shared, while each kernel has its own output
      auto d_xsPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_xs = d_xsPtr.get();
      auto d_ysPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_ys = d_ysPtr.get();
      auto d_zsPtr = cudautil::makeUniqueDeviceMemory<InputType>(THREAD_SIZE);
      InputType* d_zs = d_zsPtr.get();
      ConditionOutputPtrType d_outsPtrs[streamRepeatNum];
      for (auto& d_outPtr : d_outsPtrs) {
        d_outPtr = cudautil::makeUniqueDeviceMemory<ConditionOutputType>(THREAD_SIZE);
      }
      ConditionOutputType* d_outss[streamRepeatNum];
      for (size_t i = 0; i < streamRepeatNum; ++i) {
        d_outss[i] = d_outsPtrs[i].get();
      }

      /* allocate space for host copies and setup input values */
      auto h_xsPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_xs = h_xsPtr.get();
      auto h_ysPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_ys = h_ysPtr.get();
      auto h_zsPtr = cudautil::makeUniqueHostMemory<InputType>(THREAD_SIZE);
      InputType* h_zs = h_zsPtr.get();
      ConditionOutputPtrType h_outsPtrs[streamRepeatNum];
      for (auto& h_outPtr : h_outsPtrs) {
        h_outPtr = cudautil::makeUniqueHostMemory<ConditionOutputType>(THREAD_SIZE);
      }
      ConditionOutputType* h_outss[streamRepeatNum];
      for (size_t i = 0; i < streamRepeatNum; ++i) {
        h_outss[i] = h_outsPtrs[i].get();
      }

      for (size_t i = 0; i < THREAD_SIZE; i++) {
        h_xs[i] = h_ys[i] = h_zs[i] = InputType(i);
      }

      auto kernelCopyStartTime = system_clock::now();

      /* copy shared inputs to device */
      cudaMemcpy(d_xs, h_xs, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_ys, h_ys, inputArrayByteSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_zs, h_zs, inputArrayByteSize, cudaMemcpyHostToDevice);

      /* launch kernels in separate streams */
      for (size_t i = 0; i < streamRepeatNum; ++i) {
        cudaStatus = cudautil::runKernelInStreamAsnyc<InputType, InputType, InputType, InputType>(
          h_xs, d_xs, 0, // since we have copied the shared inputs, we do not need copy here
          h_ys, d_ys, 0,
          h_zs, d_zs, 0,
          h_outss[i], d_outss[i], THREAD_SIZE,
          kernelFunction,
          dim3(BLOCK_SIZE), dim3(THREADS_PER_BLOCK), 0, managedStreams[i].stream);

        if (cudaSuccess != cudaStatus) {
          printf("Failed to start kernel.\n");
          return;
        }
      }

      // 等待所有 Stream 执行完成
      for (auto& stream : managedStreams) {
        if (cudaSuccess != cudaStreamSynchronize(stream.stream)) {
          printf("Failed to sync stream.\n");
          return;
        }
      }

      auto kernelCopyEndTime = system_clock::now();

      if (cudaSuccess == cudaStatus) {
        auto cudaEndTime = system_clock::now();

        cout << "CUDA use: " << chrono::duration_cast<chrono::microseconds>(cudaEndTime - cudaStartTime).count() << " microseconds\n";
        cout << "CUDA Kernel copy and compute use: " << chrono::duration_cast<chrono::microseconds>(kernelCopyEndTime - kernelCopyStartTime).count() << " microseconds\n";
      }
    }

    #pragma endregion

  }
}
