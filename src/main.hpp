// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef MAIN_HPP_
#define MAIN_HPP_

#include <limits> // numeric limits

#include "basic_types.hpp" // Real
#include <parthenon/package.hpp>

// TODO(pgrete) There's a compiler bug in nvcc < 11.2 that precludes the use
// of C++17 with relaxed-constexpr in Kokkos,
// see https://github.com/kokkos/kokkos/issues/3496
// This also precludes our downstream use of constexpr int here.
// Update once nvcc/cuda >= 11.2 is more widely available on machine.
enum {
  IDN = 0,
  IM1 = 1,
  IM2 = 2,
  IM3 = 3,
  IEN = 4,
  NHYDRO = 5,
  IB1 = 5,
  IB2 = 6,
  IB3 = 7,
  IA1 = 8, // Magnetic Potential
  IA2 = 9, // for 
  IA3 = 10 // Unstaggered Constrained Transport
};

// array indices for 1D primitives: velocity and pressure
enum { IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4 };

enum class Hst { idx, ekin, emag, divb };
enum class Reconstruction { undefined, none, weno3, weno5 };
enum class Integrator { undefined, rk1, rk2, vl2, rk3 };
enum class Fluid { undefined, euler, mhd };

constexpr parthenon::Real float_min{std::numeric_limits<float>::min()};

using InitPackageDataFun_t =
    std::function<void(parthenon::ParameterInput *pin, parthenon::StateDescriptor *pkg)>;

#endif // MAIN_HPP_
