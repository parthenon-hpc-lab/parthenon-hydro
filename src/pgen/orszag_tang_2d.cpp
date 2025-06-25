
//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file orszag_tang_2d.cpp
//! \brief Problem generator for the Orszag Tang vortex.
//!
//! REFERENCE: Orszag & Tang (J. Fluid Mech., 90, 129, 1998) and
//! https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
//========================================================================================

#include <iomanip>   // For std::setprecision

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace orszag_tang_2d {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  int il, iu, jl, ju, kl, ku;
  il = ib.s - 3, iu = ib.e + 3, jl = jb.s - 3, ju = jb.e + 3, kl = kb.s, ku = kb.e;

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  Real gamma = pin->GetReal("hydro", "gamma");

  Real gm1 =  gamma - 1.0;
  Real B0 = 1.0;
  Real d0 = gamma * gamma;
  Real v0 = 1.0;
  Real p0 = gamma;

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Orszag-Tang", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        u(IDN, k, j, i) = d0;

        Real vex = -v0 * std::sin(coords.Xc<2>(j));
        Real vey =  v0 * std::sin(coords.Xc<1>(i));
        Real Bx  = -B0 * std::sin(coords.Xc<2>(j));
        Real By  =  B0 * std::sin(2.0 * coords.Xc<1>(i));
        
        // Note the different signs in this pgen compared to the the eqn mentioned in the
        // original paper (and other codes).
        // They are related to our domain going from -0.5 to 0.5 (for symmetry reason)
        // rather than 0  to 2pi (i.e., the sign for single wave sinus is flipped).
        u(IM1, k, j, i) = d0 * vex;
        u(IM2, k, j, i) = d0 * vey; 
        u(IM3, k, j, i) = 0.0;

        u(IB1, k, j, i) = Bx;
        u(IB2, k, j, i) = By; 
        u(IB3, k, j, i) = 0.0;

        u(IEN, k, j, i) = p0 / gm1 + 0.5 * d0 * (vex * vex + vey * vey) + 0.5 * (Bx * Bx + By * By);

        u(IA3, k, j, i) = 0.5 * std::cos(2.0 * coords.Xc<1>(i)) + std::cos(coords.Xc<2>(j)); 

      });
}
} // namespace orszag_tang_2d
