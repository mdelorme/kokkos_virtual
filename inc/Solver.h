#pragma once

#include "DataTypes.h"

namespace kokkos_virtual {

/**
 * Generic solver class
 **/
class Solver {
 public:
  Solver(ParticleArray &particles) : particles(particles) {};

  /**
   * Initializes the udpate by calculating accelerations
   * and every necessary quantities
   * 
   * @param s the system to prepare for update
   **/
  KOKKOS_INLINE_FUNCTION
  virtual void initUpdate(const uint32_t iPart) const = 0;

  /**
   * Updates the system
   * 
   * @param s the system to update
   **/
  virtual void updateSystem(const uint32_t iPart) const = 0;

  /**
   * Updates the time-step
   * 
   * @param dt the new time-step
   **/
  void updateDt(real_t dt) {
    this->dt = dt;
  }

  ParticleArray particles; //!< Particles to update
  real_t        dt;        //!< The time-step
};

}
