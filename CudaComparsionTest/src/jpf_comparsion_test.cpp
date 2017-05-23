#include "./jpf_comparsion_test.h"

#include <iostream>
#include <chrono>
#include <memory>

#include "jpf_condition.h"

// @see http://en.cppreference.com/w/cpp/chrono/duration
namespace chrono = std::chrono;
using chrono::system_clock;
using std::cout;
using std::endl;

static constexpr size_t THREADS_PER_BLOCK = 256; // 1024;
static constexpr size_t THREAD_SIZE = THREADS_PER_BLOCK; // 2048 * 2048;
static constexpr size_t BLOCK_SIZE = (THREAD_SIZE + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
using InputType = double;
static constexpr size_t inputArrayByteSize = THREAD_SIZE * sizeof(InputType);
using ConditionOutputType = double;
static constexpr size_t conditionOutputArrayByteSize = THREAD_SIZE * sizeof(ConditionOutputType);
using TestOutputType = bool;
static constexpr size_t testOutputArrayByteSize = THREAD_SIZE * sizeof(TestOutputType);
using ConditionOutputPtrType = decltype(std::make_unique<ConditionOutputType[]>(0));

static constexpr size_t PARALLEL_COMPUTE_THREADS = 8;

namespace cudatest {
  namespace jpftest {
    namespace comparsion {

      void runJpf52TestOnCpuSerial() {
        auto cpuStartTime = system_clock::now(); // of type system_clock::time_point

        /* allocate space for host copies of a, b, c and setup input values */
        auto h_xsPtr = std::make_unique<InputType[]>(THREAD_SIZE);
        InputType* h_xs = h_xsPtr.get();
        auto h_ysPtr = std::make_unique<InputType[]>(THREAD_SIZE);
        InputType* h_ys = h_ysPtr.get();
        auto h_zsPtr = std::make_unique<InputType[]>(THREAD_SIZE);
        InputType* h_zs = h_zsPtr.get();
        auto h_outsPtr = std::make_unique<TestOutputType[]>(THREAD_SIZE);
        TestOutputType* h_outs = h_outsPtr.get();

        for (size_t i = 0; i < THREAD_SIZE; i++) {
          h_xs[i] = h_ys[i] = h_zs[i] = InputType(i);
        }

        auto cpuComputeStartTime = system_clock::now();
        for (size_t index = 0; index < THREAD_SIZE; index++) {
          h_outs[index] = jpf52Test(h_xs[index], h_ys[index], h_zs[index]);
        }
        auto cpuComputeEndTime = system_clock::now();

        auto cpuEndTime = system_clock::now();

        cout << "CPU use: " << chrono::duration_cast<chrono::microseconds>(cpuEndTime - cpuStartTime).count() << " microseconds\n";
        cout << "CPU compute use: " << chrono::duration_cast<chrono::microseconds>(cpuComputeEndTime - cpuComputeStartTime).count() << " microseconds\n";
      }

      void runJpf52TestOnCpuParallel() {
        // TODO
      }

      static constexpr size_t kernelNum = 9;

      void runJpfMultiConditionsTestOnCpuSerial() {
        auto cpuStartTime = system_clock::now(); // of type system_clock::time_point

        using ConditionFunctionType = ConditionOutputType (*)(InputType, InputType, InputType) ;
        ConditionFunctionType testFunctions[kernelNum] = {
          &jpftest::jpf52Cond1,
          &jpftest::jpf52Cond2,
          &jpftest::jpf53Cond1,
          &jpftest::jpf53Cond2,
          &jpftest::jpf53Cond3,
          &jpftest::jpf54Cond1,
          &jpftest::jpf54Cond2,
          &jpftest::jpf54Cond3,
          &jpftest::jpf54Cond4,
        };

        /* allocate space for host copies of a, b, c and setup input values */
        auto h_xsPtr = std::make_unique<InputType[]>(THREAD_SIZE);
        InputType* h_xs = h_xsPtr.get();
        auto h_ysPtr = std::make_unique<InputType[]>(THREAD_SIZE);
        InputType* h_ys = h_ysPtr.get();
        auto h_zsPtr = std::make_unique<InputType[]>(THREAD_SIZE);
        InputType* h_zs = h_zsPtr.get();
        ConditionOutputPtrType h_outsPtrs[kernelNum];
        for (auto& h_outPtr : h_outsPtrs) {
          h_outPtr = std::make_unique<ConditionOutputType[]>(THREAD_SIZE);
        }
        ConditionOutputType* h_outss[kernelNum];
        for (size_t i = 0; i < kernelNum; ++i) {
          h_outss[i] = h_outsPtrs[i].get();
        }

        for (size_t i = 0; i < THREAD_SIZE; i++) {
          h_xs[i] = h_ys[i] = h_zs[i] = InputType(i);
        }

        auto cpuComputeStartTime = system_clock::now();
        for (size_t kernelIndex = 0; kernelIndex < kernelNum; ++kernelIndex) {
          auto h_outs = h_outss[kernelIndex];
          auto testFunction = testFunctions[kernelIndex];
          for (size_t index = 0; index < THREAD_SIZE; index++) {
            h_outs[index] = testFunction(h_xs[index], h_ys[index], h_zs[index]);
          }
        }
        auto cpuComputeEndTime = system_clock::now();

        auto cpuEndTime = system_clock::now();

        cout << "CPU use: " << chrono::duration_cast<chrono::microseconds>(cpuEndTime - cpuStartTime).count() << " microseconds\n";
        cout << "CPU compute use: " << chrono::duration_cast<chrono::microseconds>(cpuComputeEndTime - cpuComputeStartTime).count() << " microseconds\n";
      }
    }
  }
}