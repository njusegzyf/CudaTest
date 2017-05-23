#pragma once

#include <cuda_runtime.h>

namespace cudatest {

  void runReduceTest();

  cudaError_t runStreamTest();
}
