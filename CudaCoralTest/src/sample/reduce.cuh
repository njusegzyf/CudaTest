#pragma once

#include "cuda_runtime.h"

namespace cudatest {
  extern __global__ void reduceGlobalKernel(float *d_in, float *d_out);
  extern __global__ void reduceSharedKernel(float *d_in, float *d_out);
}