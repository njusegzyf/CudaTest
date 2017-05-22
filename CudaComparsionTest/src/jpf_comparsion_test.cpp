#include "./jpf_comparsion_test.h"

#include <iostream>
#include <chrono>

#include "jpf_condition.h"

// @see http://en.cppreference.com/w/cpp/chrono/duration
namespace chrono = std::chrono;
using chrono::system_clock;
using std::cout;
using std::endl;

namespace cudatest {
  namespace jpftest {
    namespace comparsion {

      constexpr size_t THREAD_SIZE = 2048 * 2048;
      constexpr size_t THREADS_PER_BLOCK = 1024;
      constexpr size_t BLOCK_SIZE = (THREAD_SIZE + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;

      constexpr size_t PARALLEL_COMPUTE_THREADS = 8;

      constexpr size_t inputDoubleArrayByteSize = THREAD_SIZE * sizeof(double);
      constexpr size_t outputBoolArrayByteSize = THREAD_SIZE * sizeof(bool);

      void runJpf52TestOnCpuSerial() {

        double* h_xs = nullptr;
        double* h_ys = nullptr;
        double* h_zs = nullptr;
        bool* h_outs = nullptr;

        auto cpuStartTime = system_clock::now(); // of type system_clock::time_point

        /* allocate space for host copies of a, b, c and setup input values */
        h_xs = static_cast<double*>(malloc(inputDoubleArrayByteSize));
        h_ys = static_cast<double*>(malloc(inputDoubleArrayByteSize));
        h_zs = static_cast<double*>(malloc(inputDoubleArrayByteSize));
        h_outs = static_cast<bool*>(malloc(outputBoolArrayByteSize));

        for (size_t i = 0; i < THREAD_SIZE; i++) {
          h_xs[i] = h_ys[i] = h_zs[i] = double(i);
        }

        auto cpuComputeStartTime = system_clock::now();
        for (size_t index = 0; index < THREAD_SIZE; index++) {
          h_outs[index] = jpf52TestForCpu(h_xs[index], h_ys[index], h_zs[index]);
        }
        auto cpuComputeEndTime = system_clock::now();

        /* clean up */
        free(h_xs);
        free(h_ys);
        free(h_zs);
        free(h_outs);

        auto cpuEndTime = system_clock::now();

        cout << "CPU use: " << chrono::duration_cast<chrono::microseconds>(cpuEndTime - cpuStartTime).count() << " microseconds\n";
        cout << "CPU compute use: " << chrono::duration_cast<chrono::microseconds>(cpuComputeEndTime - cpuComputeStartTime).count() << " microseconds\n";
      }

      void runJpf52TestOnCpuParallel()
      {
        
      }
    }
  }
}