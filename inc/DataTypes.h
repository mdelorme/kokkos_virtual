#pragma once

#include <Kokkos_Core.hpp>

namespace kokkos_virtual {
  
using real_t = double;                               //!< Real type
using ParticleArray = Kokkos::View<real_t**>;        //!< Particle array living in device memory
using ParticleArrayHost = ParticleArray::HostMirror; //!< Particle array mirror living in host memory

//!< Scratch pad memory array
using ScratchPadView 
  = Kokkos::View<real_t**, 
                 Kokkos::DefaultExecutionSpace::scratch_memory_space, 
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/**
 * Fields held in Particle array
 **/
enum VarId : uint8_t {
  IX = 0,
  IY = 1,
  IZ = 2,
  IVX = 3,
  IVY = 4,
  IVZ = 5,
  IM  = 6,
  IAX = 7,
  IAY = 8,
  IAZ = 9,
};

constexpr uint NB_FIELDS = 10;                //!< Total number of fields in simulation arrays
constexpr uint NB_STORED_FIELDS = 7;          //!< Number of fields saved and loaded from files
constexpr size_t vec_size = 3*sizeof(real_t); //!< Size of a 3 vector in bytes

/**
 * Defining a custom reduction functor
 * for summing over all accelerations
 **/
struct VecSum {
  double values[3];

  KOKKOS_INLINE_FUNCTION
  double& operator[](int i) {
    return values[i];
  }
  
  KOKKOS_INLINE_FUNCTION
  double operator[](int i) const {
    return values[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(VecSum const& other) {
    for (int i = 0; i < 3; ++i) {
      values[i] += other.values[i];
    }
  }
  KOKKOS_INLINE_FUNCTION
  void operator+=(VecSum const volatile& other) volatile {
    for (int i = 0; i < 3; ++i) {
      values[i] += other.values[i];
    }
  }
};
}

/**
 * And corresponding identity
 **/
template <>
struct Kokkos::reduction_identity<kokkos_virtual::VecSum> {
KOKKOS_FORCEINLINE_FUNCTION constexpr static kokkos_virtual::VecSum sum() {
  return kokkos_virtual::VecSum{0.0, 0.0, 0.0};
}
};