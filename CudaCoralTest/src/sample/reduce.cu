#include "reduce.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

namespace cudatest {
  __global__ void reduceGlobalKernel(float *d_in, float *d_out) {
    int id = threadIdx.x + blockDim.x * threadIdx.x; // the id in the whole input array
    int tid = threadIdx.x;

    // do reduction in global memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
      // in each iteration, only half threads of the previous iteration do work
      if (tid < s) {
        d_in[id] += d_in[id + s];
      }
      __syncthreads();
    }

    // only thread 0 writes result for this block to global memory
    if (0 == tid) {
      d_out[blockIdx.x] = d_in[id];
    }
  }

  __global__ void reduceSharedKernel(float *d_in, float *d_out) {
    // shared memory can be allocated statically like:
    // __shared__ float sh_data[128];
    // or allocated in the kernel call: 3rd argument of <<<b, t, shMenSize>>>
    extern __shared__ float sh_data[];

    int id = threadIdx.x + blockDim.x * threadIdx.x; // the id in the whole input array
    int tid = threadIdx.x;

    // copy global memory to shared memory
    sh_data[tid] = d_in[id];
    __syncthreads();

    // do reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
      // in each iteration, only half threads of the previous iteration do work
      if (tid < s) {
        sh_data[tid] += sh_data[tid + s];
      }
      __syncthreads();
    }

    // only thread 0 writes result for this block to global memory
    if (0 == tid) {
      d_out[blockIdx.x] = sh_data[0];
    }
  }
}

