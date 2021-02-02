#include <bits/stdc++.h>
#include <Kokkos_Core.hpp>

#include "inc/Solver.h"
#include "inc/DataTypes.h"
#include "inc/DataManager.h"
#include "inc/Timers.h"
#include "inc/Parameters.h"

#include "inc/ComputeAccelerationFunctor.h"

using namespace kokkos_virtual;

class EulerUpdateFunctor {
 public:
  EulerUpdateFunctor(ParticleArray particles, real_t dt) : particles(particles), dt(dt) {};

  static void apply(ParticleArray particles, real_t dt) {
    uint nbParticles = particles.extent(1);

    EulerUpdateFunctor functor(particles, dt);
    Kokkos::RangePolicy policy(0, nbParticles);
    Kokkos::parallel_for("Euler update", policy, functor);
    Kokkos::fence();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const uint iPart) const {
    // 1- Updating positions
    particles(IX, iPart) += dt * particles(IVX, iPart);
    particles(IY, iPart) += dt * particles(IVY, iPart);
    particles(IZ, iPart) += dt * particles(IVZ, iPart);

    // 2- Updating velocities
    particles(IVX, iPart) += dt * particles(IAX, iPart);
    particles(IVY, iPart) += dt * particles(IAY, iPart);
    particles(IVZ, iPart) += dt * particles(IAZ, iPart);
  }

  ParticleArray particles; //!< Particle array on device
  real_t dt;               //!< Time-step
};

class LeapfrogUpdateFunctor {
 public:
  LeapfrogUpdateFunctor(ParticleArray particles, real_t dt) : particles(particles), dt(dt) {};

  static void apply(ParticleArray particles, real_t dt) {
    uint nbParticles = particles.extent(1);

    LeapfrogUpdateFunctor functor(particles, dt);
    Kokkos::RangePolicy policy(0, nbParticles);
    Kokkos::parallel_for("Leapfrog update", policy, functor);
    Kokkos::fence();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const uint iPart) const {
    // 1- Updating velocities
    particles(IVX, iPart) += dt * particles(IAX, iPart);
    particles(IVY, iPart) += dt * particles(IAY, iPart);
    particles(IVZ, iPart) += dt * particles(IAZ, iPart);

    // 2- Updating positions
    particles(IX, iPart) += dt * particles(IVX, iPart);
    particles(IY, iPart) += dt * particles(IVY, iPart);
    particles(IZ, iPart) += dt * particles(IVZ, iPart);
  }

  ParticleArray particles; //!< Particle array on device
  real_t dt;               //!< Time-step
};

void usage(int argc, char **argv) {
  std::cerr << "ERROR: No parameter file provided !" << std::endl;
  std::cerr << "USAGE: " << argv[0] << " parameter_file" << std::endl;
  std::exit(1);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv); 
  {
    std::cout << " == Test #1: No specific effort made" << std::endl;
    if (argc < 2)
      usage(argc, argv);

    // Starting the timer
    TimerManager time_manager;

    // First we read the parameters
    std::string filename{argv[1]};
    Parameters params(filename);
    params.PrintParameters();

    // We start the timers

    // Initializing data
    DataManager data_manager;
    time_manager.Start("IOs");
    ParticleArray particles = data_manager.InitData(params.input_filename);
    time_manager.Stop("IOs");

    real_t t = 0.0;
    uint iteration = 0;

    std::cout << " == Saving initial condition " << std::endl;
    time_manager.Start("IOs");
    std::string output_filename = data_manager.BuildFilename(params, iteration);
    data_manager.SaveData(output_filename);
    time_manager.Stop("IOs");

    while (t < params.tmax) {
      iteration++;

      real_t dt = params.dt;

      // 1- Computing accelerations
      time_manager.Start("Accelerations");
      ComputeAccelerationsFunctor::apply(particles, params);
      time_manager.Stop("Accelerations");

      // 2- Updating particles
      time_manager.Start("Update");
      if (params.update_type == "euler")      
        EulerUpdateFunctor::apply(particles, params.dt);
      else if (params.update_type == "leapfrog")
        LeapfrogUpdateFunctor::apply(particles, params.dt);
      else {
        std::cout << "ERROR : Unknown update !" << std::endl;
        std::exit(4);
      }
      time_manager.Stop("Update");

      // 3- Updating time and saving
      t += dt;

      if (iteration % params.output_freq == 0) {
        std::cout << " == Saving solution at t=" << std::setprecision(3) << t << " (iteration #" << iteration << ")" << std::endl;
        time_manager.Start("IOs");
        output_filename = data_manager.BuildFilename(params, iteration);
        data_manager.SaveData(output_filename);
        time_manager.Stop("IOs");
      }      
    }

    // Printing end of run info
    std::cout << std::endl;
    std::cout << " ===== Run terminated =====" << std::endl;
    time_manager.PrintTimers();
    real_t parts_per_sec = iteration * particles.extent(1) / time_manager.total_time.total_time;
    std::cout << " = Updates " << parts_per_sec << " particles / second" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}