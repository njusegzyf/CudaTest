#include <iostream>

#include "jpf/jpf_test.cuh"
#include "sample/sample_test.h"

using std::cout;

int main() {
  //cout << "\nRun for warm up:\n";
  //cudatest::jpftest::runJpf52Test();
  //cout << "\nRun JPF Benchmark52 test with input type double:\n";
  //cudatest::jpftest::runJpf52Test();

  //cout << "\nRun for warm up:\n";
  //cudatest::jpftest::runJpf52TestFloat();
  //cout << "\nRun JPF Benchmark52 test with input type float:\n";
  //cudatest::jpftest::runJpf52TestFloat();

  // cudatest::runReduceTest();
  // cudatest::runReduceTest();

  // cudatest::runReduceTest();
  // cudatest::runReduceTest();

  //cout << "\nRun for warm up:\n";
  //cudatest::jpftest::runJpfMultiConditionsTest();
  //cout << "\nRun JPF multiple condition test with input type double:\n";
  //cudatest::jpftest::runJpfMultiConditionsTest();

  cout << "\nRun for warm up:\n";
  cudatest::jpftest::runJpfMultiRepeatedConditionsTest();
  cout << "\nRun JPF multiple repeated(128) condition(JPF53Cond3) test with input type double:\n";
  cudatest::jpftest::runJpfMultiRepeatedConditionsTest();

  // getchar();
  return 0;
}
