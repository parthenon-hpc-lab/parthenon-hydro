//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recon.hpp
//  \brief Lax Friedrichs Reconnection Methods 

#ifndef RECON_HPP_
#define RECON_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

// First declare general template
template <Fluid fluid, Reconstruction recon>
struct Reconstruct;

// now include the specializations
#include "hydro_weno3.hpp"
#include "hydro_weno5.hpp"
// #include "mhd_weno3.hpp"
#include "mhd_weno5.hpp"

// "none" solvers for runs/testing without fluid evolution, i.e., just reset fluxes
template <>
struct Reconstruct<Fluid::euler, Reconstruction::none> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
                const int iu, const int ivx, const parthenon::VariablePack<Real> &q,
                VariableFluxPack<Real> &cons, const AdiabaticHydroEOS &eos) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      for (size_t v = 0; v < NHYDRO; v++) {
        cons.flux(ivx, v, k, j, i) = 0.0;
      }
    });
  }
};

// "none" solvers for runs/testing without fluid evolution, i.e., just reset fluxes
template <>
struct Reconstruct<Fluid::mhd, Reconstruction::none> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
                const int iu, const int ivx, const parthenon::VariablePack<Real> &q,
                VariableFluxPack<Real> &cons, const AdiabaticMHDEOS &eos) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      for (size_t v = 0; v < NHYDRO; v++) {
        cons.flux(ivx, v, k, j, i) = 0.0;
      }
    });
  }
};

#endif // RECON_HPP_