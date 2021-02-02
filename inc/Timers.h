#include <Kokkos_Core.hpp>
#include <bits/stdc++.h>

#include "DataTypes.h"

namespace kokkos_virtual {

/**
 * Simple class to manage timers for a run
 * 
 * @todo Allow instrumentation compatible with non-blocking CUDA kernels
 **/
class TimerManager {
 public:
  /**
   * Timer structure, holding a handle for a Kokkos timer and the total time elapsed
   * in this semantic unit
   **/
  struct Timer {
    Timer() : total_time(0.0) {};

    /**
     * Starts the timer
     **/
    void Start() {
      timer.reset();
    }

    /**
     * Stops the timer and records elapsed time
     **/
    void Stop() {
      total_time += timer.seconds();
    }

    Kokkos::Timer timer; //!< Timer handle using Kokkos timers
    real_t total_time;   //!< Total time elapsed
  };

  /**
   * Default constructor. Starts the total run timer.
   **/
  TimerManager() {
    total_time.Start();
  };

  /**
   * Accessing the timers by name
   * 
   * @param name the name of the timer
   * @result a reference to the corresponding timer
   **/
  Timer& operator[](std::string name) {
    return timers[name];
  }

  /**
   * Starts a given timer
   * 
   * @param the timer to start
   **/
  void Start(std::string name) {
    timers[name].Start();
  }

  /**
   * Stops a given timer
   * 
   * @param name the timer to stop
   **/
  void Stop(std::string name) {
    timers[name].Stop();
  }

  /**
   * Printing timers, stops the total run timer
   **/
  void PrintTimers() {
    total_time.Stop();
    std::cout << " == Timing information : " << std::endl;
    for (auto &[k, t]: timers) {
      real_t frac = t.total_time / total_time.total_time * 100.0;
      std::cout << "   . " << std::setw(15) << k << " : " << t.total_time << "s\t(" << frac << "%)" << std::endl;
    }

    std::cout << " == Total time : " << total_time.total_time << "s" << std::endl;
  }

  std::map<std::string, Timer> timers; //!< Map binding timers to a logical name
  Timer total_time; //!< Total time since init of the timer manager to printing the log information
};

}