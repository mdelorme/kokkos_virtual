#pragma once

#include <Kokkos_Core.hpp>

#include "Parameters.h"
#include "DataTypes.h"

/**
 *  TODO :
 *  . find a better way to switch between the implementations 
 *    of ComputeAccelerationsFunctor
 *  . Implement the custom reduction ... Pain ....
 **/
// Tests here for the udpate
//#define ATOMIC_UPDATE
#define SCRATCH_UPDATE
//#define CUSTOM_REDUCTION

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
      particles(IAX, iPart) = 0.0;
      particles(IAY, iPart) = 0.0;
      particles(IAZ, iPart) = 0.0;
      
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
    member.team_barrier();

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

#endif

  ParticleArray particles;  //!< The particle array to update
  Parameters    params;     //!< The parameters of the run
  uint32_t      nParticles; //!< Number of particles to treat
};

}