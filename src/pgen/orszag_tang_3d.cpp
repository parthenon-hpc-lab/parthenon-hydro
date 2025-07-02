
//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file orszag_tang_3d.cpp
//! \brief Problem generator for the Orszag Tang vortex.
//!
//! REFERENCE: Orszag & Tang (J. Fluid Mech., 90, 129, 1998) and
//! https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace orszag_tang_3d {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real nghost = pin->GetOrAddReal("parthenon/mesh", "nghost", 3);

  int il, iu, jl, ju, kl, ku;
  il = ib.s - nghost, iu = ib.e + nghost;
  jl = jb.s - nghost, ju = jb.e + nghost;
  kl = kb.s - nghost, ku = kb.e + nghost;

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  Real gamma = pin->GetReal("hydro", "gamma");

  Real gm1 =  gamma - 1.0;
  Real B0 = 1.0;
  Real d0 = gamma * gamma;
  Real v0 = 1.0;
  Real p0 = gamma;

  Real ep = 0.2;

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Orszag-Tang 3d", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        u(IDN, k, j, i) = d0;

        Real vex = -v0 * std::sin(coords.Xc<2>(j)) * (1 + ep * std::sin(coords.Xc<3>(k)));
        Real vey =  v0 * std::sin(coords.Xc<1>(i)) * (1 + ep * std::sin(coords.Xc<3>(k)));
        Real vez =  v0 * ep * std::sin(coords.Xc<3>(k));
        Real Bx = -B0 * std::sin(coords.Xc<2>(j));
        Real By =  B0 * std::sin(2.0 * coords.Xc<1>(i));
        Real Bz = 0.0;

        // Note the different signs in this pgen compared to the the eqn mentioned in the
        // original paper (and other codes).
        // They are related to our domain going from -0.5 to 0.5 (for symmetry reason)
        // rather than 0  to 2pi (i.e., the sign for single wave sinus is flipped).
        u(IM1, k, j, i) = d0 * vex;
        u(IM2, k, j, i) = d0 * vey;
        u(IM3, k, j, i) = d0 * vez;

        u(IB1, k, j, i) = Bx;
        u(IB2, k, j, i) = By;
        u(IB3, k, j, i) = Bz;

        u(IEN, k, j, i) = p0 / gm1 + 0.5 * d0 * (vex * vex + vey * vey + vez * vez) + 0.5 * (Bx * Bx + By * By + Bz * Bz);

        u(IA3, k, j, i) = 0.5 * std::cos(2.0 * coords.Xc<1>(i)) + std::cos(coords.Xc<2>(j));
        
      });
}
} // namespace orszag_tang_3d
