#pragma once

#include <cmath>

// allow device functions to be treated as regular functions
#define __device__

namespace cudatest {
  namespace jpftest {

    #pragma region JPF benchmark52 conditions

    //x ^ tan(y) + z < x * Math.atan(z) && sin(y) + cos(y) + tan(y) >= x - z
    //public static void benchmark52(double x, double y, double z) {
    //  if (Math.pow(x, Math.tan(y)) + z < x * Math.atan(z) && Math.sin(y) + Math.cos(y) + Math.tan(y) >= x - z) {
    //    System.out.println("sucess");
    //  }

    __device__ inline double jpf52Cond1(double x, double y, double z) {
      return (pow(x, tan(y)) + z) - (x * atan(z));
    }

    __device__ inline double jpf52Cond2(double x, double y, double z) {
      return (sin(y) + cos(y) + tan(y)) - (x - z);
    }

    __device__ inline bool jpf52Test(double x, double y, double z) {
      return jpf52Cond1(x, y, z) < 0 && jpf52Cond2(x, y, z) >= 0;
    }

    #pragma region Float version

    __device__ inline float jpf52Cond1(float x, float y, float z) {
      return (pow(x, tan(y)) + z) - (x * atan(z));
    }

    __device__ inline float jpf52Cond2(float x, float y, float z) {
      return (sin(y) + cos(y) + tan(y)) - (x - z);
    }

    __device__ inline bool jpf52Test(float x, float y, float z) {
      return jpf52Cond1(x, y, z) < 0 && jpf52Cond2(x, y, z) >= 0;
    }

    #pragma endregion

    #pragma endregion

    #pragma region JPF benchmark53 conditions

    ////x ^ Math.tan(y) + z < x * Math.atan(z) && Math.sin(y) + cos(y) + Math.tan(y) >= x - z && Math.atan(x) + Math.atan(y) > y
    //public static void benchmark53(double x, double y, double z) {
    //  if (Math.pow(x, Math.tan(y)) + z < x * Math.atan(z) && Math.sin(y) + Math.cos(y) + Math.tan(y) >= x - z && Math.atan(x) + Math.atan(y) > y) {
    //    System.out.println("sucess");
    //  }
    //}

    __device__ inline double jpf53Cond1(double x, double y, double z) {
      return (pow(x, tan(y)) + z) - (x * atan(z));
    }

    __device__ inline double jpf53Cond2(double x, double y, double z) {
      return (sin(y) + cos(y) + tan(y)) - (x - z);
    }

    __device__ inline double jpf53Cond3(double x, double y, double z) {
      return (atan(x) + atan(y)) - (y);
    }

    __device__ inline bool jpf53Test(double x, double y, double z) {
      return jpf53Cond1(x, y, z) < 0 && jpf53Cond2(x, y, z) >= 0 && jpf53Cond3(x, y, z) > 0;
    }

    #pragma region Float version

    __device__ inline float jpf53Cond1(float x, float y, float z) {
      return (pow(x, tan(y)) + z) - (x * atan(z));
    }

    __device__ inline float jpf53Cond2(float x, float y, float z) {
      return (sin(y) + cos(y) + tan(y)) - (x - z);
    }

    __device__ inline float jpf53Cond3(float x, float y, float z) {
      return (atan(x) + atan(y)) - (y);
    }

    __device__ inline bool jpf53Test(float x, float y, float z) {
      return jpf53Cond1(x, y, z) < 0 && jpf53Cond2(x, y, z) >= 0 && jpf53Cond3(x, y, z) > 0;
    }

    #pragma endregion

    #pragma endregion

    #pragma region JPF benchmark54 conditions

    ////x ^ Math.tan(y) + z < x * Math.atan(z) && Math.sin(y) + Math.cos(y) + Math.tan(y) >= x - z && Math.atan(x) + Math.atan(y) > y && Math.log(x^Math.tan(y)) < Math.log(z)
    //public static void benchmark54(double x, double y, double z) {
    //  if (Math.pow(x, Math.tan(y)) + z < x * Math.atan(z) && Math.sin(y) + Math.cos(y) + Math.tan(y) >= x - z && Math.atan(x) + Math.atan(y) > y && Math.log(Math.pow(x, Math.tan(y))) < Math.log(z)) {
    //    System.out.println("sucess");
    //  }
    //}

    __device__ inline double jpf54Cond1(double x, double y, double z) {
      return (pow(x, tan(y)) + z) - (x * atan(z));
    }

    __device__ inline double jpf54Cond2(double x, double y, double z) {
      return (sin(y) + cos(y) + tan(y)) - (x - z);
    }

    __device__ inline double jpf54Cond3(double x, double y, double z) {
      return (atan(x) + atan(y)) - (y);
    }

    __device__ inline double jpf54Cond4(double x, double y, double z) {
      return (log(pow(x, tan(y)))) - (log(z));
    }

    __device__ inline bool jpf54Test(double x, double y, double z) {
      return jpf54Cond1(x, y, z) < 0 && jpf54Cond2(x, y, z) >= 0 && jpf54Cond3(x, y, z) > 0 && jpf54Cond4(x, y, z) < 0;
    }

    #pragma endregion
  }
}
