#pragma once

#include <cmath>

namespace cudatest {
  namespace jpftest {

    #pragma region JPF benchmark52 conditions

    //x ^ tan(y) + z < x * Math.atan(z) && sin(y) + cos(y) + tan(y) >= x - z
    //public static void benchmark52(double x, double y, double z) {
    //  if (Math.pow(x, Math.tan(y)) + z < x * Math.atan(z) && Math.sin(y) + Math.cos(y) + Math.tan(y) >= x - z) {
    //    System.out.println("sucess");
    //  }

    inline bool jpf52Cond1ForCpu(double x, double y, double z) {
      return std::pow(x, std::tan(y)) + z < x * std::atan(z);
    }

    inline bool jpf52Cond2ForCpu(double x, double y, double z) {
      return std::sin(y) + std::cos(y) + std::tan(y) >= x - z;
    }

    inline bool jpf52TestForCpu(double x, double y, double z) {
      return jpf52Cond1ForCpu(x, y, z) && jpf52Cond2ForCpu(x, y, z);
    }

    #pragma endregion JPF benchmark52 conditions
  }
}