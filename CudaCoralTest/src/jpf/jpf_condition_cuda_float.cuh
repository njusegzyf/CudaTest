#pragma once

#include <cmath>

#include "cuda_runtime.h"

namespace cudatest {
  namespace jpftest {

    #pragma region JPF benchmark52 conditions

    //x ^ tan(y) + z < x * Math.atan(z) && sin(y) + cos(y) + tan(y) >= x - z
    //public static void benchmark52(double x, double y, double z) {
    //  if (Math.pow(x, Math.tan(y)) + z < x * Math.atan(z) && Math.sin(y) + Math.cos(y) + Math.tan(y) >= x - z) {
    //    System.out.println("sucess");
    //  }

    __device__ inline bool jpf52Cond1Float(float x, float y, float z) {
      return std::pow(x, std::tan(y)) + z < x * std::atan(z);
    }

    __device__ inline bool jpf52Cond2Float(float x, float y, float z) {
      return std::sin(y) + std::cos(y) + std::tan(y) >= x - z;
    }

    __device__ inline bool jpf52Test(float x, float y, float z) {
      return jpf52Cond1Float(x, y, z) && jpf52Cond2Float(x, y, z);
    }

    #pragma endregion JPF benchmark52 conditions
  }
}
