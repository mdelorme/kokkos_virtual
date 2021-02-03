#pragma once

#include <Kokkos_Core.hpp>

#include "Parameters.h"
#include "DataTypes.h"

/**
 *  TODO :
 *  . add a better way to switch between the implementations 
 *    of ComputeAccelerationsFunctor
 **/

/**
 * Note about implementation and experimentations
 * Calculation of accelerations is the O(N^2) part of the algorithm
 * 
 * The problem when dealing with multi-threaded implementation is the
 * race conditions happening when multiple threads try to update the 
 * accelerations of a given particle at the same time.
 * 
 * There is an additional complication which might impede the execution
 * speed. Using hierarchical parallelism with Kokkos requires the writing
 * of a special custom type for the reduction in the inner loop. This is 
 * a bit tedious (cf the last part of datatypes.h).
 * 
 * Here, I have experimented with four implementations, which can be
 * selected using the preprocessor (yeah ... I should think of something better)
 * 
 * 1- ATOMIC_UPDATE -> The update is made in concurrent using atomic_add inside
 *                     the inner loop. This might slow down execution because
 *                     of locking mechanisms
 * 
 * 2- SCRATCH_UPDATE -> We reserve scratch space of the size of 3 doubles times
 *                      the number of threads in a team. Each thread has its
 *                      own acceleration which is updated in a non-concurrent
 *                      way. At the end, a single Kokkos thread sums everything.
 * 
 * 3- CUSTOM_REDUCTION -> We implement a double[3] type that will be reduced
 *                        using a sum. We then replace the inner parallel_for
 *                        by a parallel_reduce
 * 
 * 4- TRIPLE_SCALAR_REDUCTION -> We use three scalar reductions along each 
 *                               axis to compute the final acceleration.
 * 
 * Preliminary results :
 * 
 * The setup is 10^4 iterations with N=2000
 * 
 * Method           Time computing acc          % total      Parts/sec
 * -------------------------------------------------------------------
 * ATOMIC_UPDATE                 31.6s            76.6%         4.86e5
 * SCRATCH_UPDATE                32.7s            77.7%         4.75e5
 * CUSTOM_REDUCTION              31.3s            75.8%         4.85e5
 * TRIPLE_SCALAR_REDUCTION       93.8s            88.8%         1.89e5
 * 
 * Preliminary conclusions :
 * 
 * The first three are equivalent. Atomic access must (maybe ?) be balanced
 * with coalescence loss in the custom reduction. What is happening in the 
 * case of the scratch update is not entirely clear. The final summation might
 * be taking a lot of time. Finally the triple scalar reduction does not 
 * balance the cost of the redundant calculations
 **/

// Tests here for the udpate
#define ATOMIC_UPDATE
//#define SCRATCH_UPDATE
//#define CUSTOM_REDUCTION
//#define TRIPLE_SCALAR_REDUCTION

namespace kokkos_virtual {
/**
 * Computes accelerations on a particle array
 **/
class ComputeAccelerationsFunctor {
 private:
  using Policy       = Kokkos::TeamPolicy<Kokkos::IndexType<uint32_t>>;
  using member_type  = Policy::member_type; 

 public:
  /**
   * Constructor
   * 
   * @param particles the particle array to be updated
   * @param params the parameter structure of the run
   **/
  ComputeAccelerationsFunctor(ParticleArray &particles,
                              Parameters    &params)
    : particles(particles), params(params) {
      nParticles = particles.extent(1);
    }

  /**
   * Static function to instantiate the functor and apply it on the
   * particle array.
   * 
   * @param particles the particle array to update
   * @param params the parameters of the run
   **/
  static void apply(ParticleArray &particles,
                    Parameters    &params) {
    // Create the functor
    ComputeAccelerationsFunctor functor(particles, params);

    uint nParticles = particles.extent(1);

    // First we reset the values of the accelerations
    // We do the reset operation apart to make it simpler
    // when we will switch to team policy
    Kokkos::parallel_for("ResetAccelerations", 
                         Kokkos::RangePolicy(0, nParticles),
                         KOKKOS_LAMBDA (const uint iPart) {
                           particles(IAX, iPart) = 0.0;
                           particles(IAY, iPart) = 0.0;
                           particles(IAZ, iPart) = 0.0;
                         });

    // Initialize team policy
    Kokkos::TeamPolicy policy(nParticles, Kokkos::AUTO);

#ifdef SCRATCH_UPDATE
    policy.set_scratch_size(0, Kokkos::PerThread(vec_size));
#endif    
    // Then, we apply the functor
    Kokkos::parallel_for("ComputeAccelerationFunctor", policy, functor);

    // Synchronization here. It is done so that the timers are accurate.
    Kokkos::fence();
  }

#ifdef ATOMIC_UPDATE
  /**
   * Operator calculating the accelerations for the current particle
   * 
   * @param thread the current element to be updated
   **/
  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & member) const {
    uint iPart = member.league_rank();
  
    while (iPart < nParticles) {
      // Reseting accelerations
      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        particles(IAX, iPart) = 0.0;
        particles(IAY, iPart) = 0.0;
        particles(IAZ, iPart) = 0.0;
      });
      
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nParticles), 
        [=] (const uint jPart) {
          if (jPart == iPart)
            return;

          const real_t dx = particles(IX, jPart) - particles(IX, iPart); 
          const real_t dy = particles(IY, jPart) - particles(IY, iPart);
          const real_t dz = particles(IZ, jPart) - particles(IZ, iPart);
          const real_t dist = sqrt(dx*dx+dy*dy+dz*dz);
          const real_t d3 = dist*dist*dist;

          // Calculating acceleration
          const real_t a = params.G * particles(IM, jPart) / d3;

          // And assigning accelerations 
          Kokkos::atomic_add(&particles(IAX, iPart), a*dx);
          Kokkos::atomic_add(&particles(IAY, iPart), a*dy);
          Kokkos::atomic_add(&particles(IAZ, iPart), a*dz);
        });

      iPart += member.league_size();
    }
  }
#endif

#ifdef SCRATCH_UPDATE
  /**
   * Operator calculating the accelerations for the current particle
   * 
   * @param thread the current element to be updated
   **/
  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & member) const {
    // Current particle is the id of the team
    // which is mapped on the id of the particle
    uint iPart = member.league_rank();
    
    // We allocate scratch memory once
    ScratchPadView acc(member.team_scratch(0), 3, member.team_size());

    uint iRank = member.team_rank();

    // Resetting scratch memory
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, member.team_size()),
      [=] (const uint i) {
        acc(IX, i) = 0.0;
        acc(IY, i) = 0.0;
        acc(IZ, i) = 0.0;
      });

      
    // Calculating memory in scratch
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nParticles), 
      [=] (const uint jPart) {
        if (jPart == iPart)
          return;

        const real_t dx   = particles(IX, jPart) - particles(IX, iPart); 
        const real_t dy   = particles(IY, jPart) - particles(IY, iPart);
        const real_t dz   = particles(IZ, jPart) - particles(IZ, iPart);
        const real_t dist = sqrt(dx*dx+dy*dy+dz*dz);
        const real_t d3   = dist*dist*dist;

        // Calculating acceleration
        const real_t a = params.G * particles(IM, jPart) / d3;

        // And assigning accelerations 
        acc(IX, iRank) += a*dx;
        acc(IY, iRank) += a*dy;
        acc(IZ, iRank) += a*dz;
      });

    // Waiting for all threads to be finished with the calculations
    //member.team_barrier();

    // Joining scratch using only one process per team
    Kokkos::single(Kokkos::PerTeam(member), [=]() {
      particles(IAX, iPart) = 0.0;
      particles(IAY, iPart) = 0.0;
      particles(IAZ, iPart) = 0.0;

      for (int i=0; i < member.team_size(); ++i) {
        particles(IAX, iPart) += acc(IX, i);
        particles(IAY, iPart) += acc(IY, i);
        particles(IAZ, iPart) += acc(IZ, i);
      }
    });

                      
  }
#endif

#ifdef CUSTOM_REDUCTION
  /**
   * Operator calculating the accelerations for the current particle
   * 
   * @param thread the current element to be updated
   **/
  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & member) const {
    // Current particle is the id of the team
    // which is mapped on the id of the particle
    uint iPart = member.league_rank();
    uint iRank = member.team_rank();

    VecSum acc{0.0, 0.0, 0.0};
      
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, nParticles), 
      [=] (const uint jPart, VecSum &sum) {
        if (jPart == iPart)
          return;

        const real_t dx   = particles(IX, jPart) - particles(IX, iPart); 
        const real_t dy   = particles(IY, jPart) - particles(IY, iPart);
        const real_t dz   = particles(IZ, jPart) - particles(IZ, iPart);
        const real_t dist = sqrt(dx*dx+dy*dy+dz*dz);
        const real_t d3   = dist*dist*dist;

        // Calculating acceleration
        const real_t a = params.G * particles(IM, jPart) / d3;

        // And assigning accelerations 
        sum[IX] += a*dx;
        sum[IY] += a*dy;
        sum[IZ] += a*dz;
      }, acc);

    // Waiting for all threads to be finished with the calculations
    //member.team_barrier();

    // Joining scratch using only one process per team
    Kokkos::single(Kokkos::PerTeam(member), [=]() {
      particles(IAX, iPart) = acc[IAX];
      particles(IAY, iPart) = acc[IAY];
      particles(IAZ, iPart) = acc[IAZ];
    });
  }
#endif

#ifdef TRIPLE_SCALAR_REDUCTION
  // /!\ NOT EFFICIENT AT ALL !
  /**
   * Operator calculating the accelerations for the current particle
   * 
   * @param thread the current element to be updated
   **/
  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & member) const {
    // Current particle is the id of the team
    // which is mapped on the id of the particle
    uint iPart = member.league_rank();
    uint iRank = member.team_rank();

    // Reset acceleration
    Kokkos::single(Kokkos::PerTeam(member), [=]() {
      particles(IAX, iPart) = 0.0;
      particles(IAY, iPart) = 0.0;
      particles(IAZ, iPart) = 0.0;
    });

    // Encapsulating the calculations of the acceleration term in a lambda
    auto compute_acc = [=](const uint iPart, const uint jPart, int dir) {
      const real_t dx   = particles(IX, jPart) - particles(IX, iPart); 
      const real_t dy   = particles(IY, jPart) - particles(IY, iPart);
      const real_t dz   = particles(IZ, jPart) - particles(IZ, iPart);
      const real_t dist = sqrt(dx*dx+dy*dy+dz*dz);
      const real_t d3   = dist*dist*dist;

      const real_t d[] {dx, dy, dz};

      return d[dir] * params.G * particles(IM, jPart) / d3;
    };
      
    // Using it on each acceleration component
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, nParticles), 
      [=] (const uint jPart, real_t &acc_x) {
        if (jPart == iPart)
          return;
        
        // And assigning accelerations 
        acc_x += compute_acc(iPart, jPart, IX);
      }, particles(IAX, iPart));

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, nParticles), 
      [=] (const uint jPart, real_t &acc_y) {
        if (jPart == iPart)
          return;
        
        // And assigning accelerations 
        acc_y += compute_acc(iPart, jPart, IY);
      }, particles(IAY, iPart));

    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, nParticles), 
      [=] (const uint jPart, real_t &acc_z) {
        if (jPart == iPart)
          return;
        
        // And assigning accelerations 
        acc_z += compute_acc(iPart, jPart, IZ);
      }, particles(IAZ, iPart));
  }
#endif

  ParticleArray particles;  //!< The particle array to update
  Parameters    params;     //!< The parameters of the run
  uint32_t      nParticles; //!< Number of particles to treat
};

}