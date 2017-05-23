#include <cstdio>

#include "src/jpf_comparsion_test.h"

namespace testns = cudatest::jpftest::comparsion;

int main() {
  // testns::runJpf52TestOnCpuSerial();
  testns::runJpfMultiConditionsTestOnCpuSerial();

  getchar();
  return 0;
}

