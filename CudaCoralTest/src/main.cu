#include <stdio.h>

#include "jpf/jpf_test.cuh"
#include "sample/reduce_test.cuh"
#include "sample/cuda_stream.cuh"

int main() {

  cudatest::jpftest::runJpf52Test();
  // cudatest::runReduceTest();
  // cudatest::runStreamTest();

  return 0;
}
