#pragma once
#include <chrono>

namespace cudatest {
  
  using std::chrono::system_clock;
  using Timepoint = decltype(system_clock::now());
  using Duration = decltype(system_clock::now() - system_clock::now());

  class TestBase {
    public:

    protected:
    inline void stratTimer1() { timer1StartPoint = system_clock::now(); }
    inline void endTimer1() { timer1EndPoint = system_clock::now(); }
    inline void stratTimer2() { timer2StartPoint = system_clock::now(); }
    inline void endTimer2() { timer2EndPoint = system_clock::now(); }

    private:
    Timepoint timer1StartPoint;
    Timepoint timer1EndPoint;
    Timepoint timer2StartPoint;
    Timepoint timer2EndPoint;
  };
}