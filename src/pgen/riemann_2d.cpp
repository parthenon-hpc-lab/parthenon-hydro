
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"

namespace riemann_2d {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real nghost = pin->GetOrAddReal("parthenon/mesh", "nghost", 3);

  int il, iu, jl, ju, kl, ku;
  il = ib.s - nghost, iu = ib.e + nghost;
  jl = jb.s - nghost, ju = jb.e + nghost;
  kl = kb.s, ku = kb.e;

  // Top Right
  Real rho_tr = pin->GetOrAddReal("problem/riemann_2d", "rho_tr", 1.5);
  Real vex_tr = pin->GetOrAddReal("problem/riemann_2d", "vex_tr", 0.0);
  Real vey_tr = pin->GetOrAddReal("problem/riemann_2d", "vey_tr", 0.0);
  Real pre_tr = pin->GetOrAddReal("problem/riemann_2d", "pre_tr", 1.5);

  // Top Left
  Real rho_tl = pin->GetOrAddReal("problem/riemann_2d", "rho_tl", 0.5323);
  Real vex_tl = pin->GetOrAddReal("problem/riemann_2d", "vex_tl", 1.206);
  Real vey_tl = pin->GetOrAddReal("problem/riemann_2d", "vey_tl", 0.0);
  Real pre_tl = pin->GetOrAddReal("problem/riemann_2d", "pre_tl", 0.3);

  // Bottom Left
  Real rho_bl = pin->GetOrAddReal("problem/riemann_2d", "rho_bl", 0.138);
  Real vex_bl = pin->GetOrAddReal("problem/riemann_2d", "vex_bl", 1.206);
  Real vey_bl = pin->GetOrAddReal("problem/riemann_2d", "vey_bl", 1.206);
  Real pre_bl = pin->GetOrAddReal("problem/riemann_2d", "pre_bl", 0.029);

  // Bottom Right
  Real rho_br = pin->GetOrAddReal("problem/riemann_2d", "rho_br", 0.5323);
  Real vex_br = pin->GetOrAddReal("problem/riemann_2d", "vex_br", 0.0);
  Real vey_br = pin->GetOrAddReal("problem/riemann_2d", "vey_br", 1.206);
  Real pre_br = pin->GetOrAddReal("problem/riemann_2d", "pre_br", 0.3);

  Real x_discont = pin->GetOrAddReal("problem/riemann_2d", "x_discont", 0.5);
  Real y_discont = pin->GetOrAddReal("problem/riemann_2d", "y_discont", 0.5);

  Real gamma = pin->GetReal("hydro", "gamma");

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &cons = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator Riemann", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {

        if (coords.Xc<1>(i) < x_discont) {  // Left half
          if (coords.Xc<2>(j) < y_discont) {  // Bottom-left quadrant
            cons(IDN, k, j, i) = rho_bl;
            cons(IM1, k, j, i) = rho_bl * vex_bl;
            cons(IM2, k, j, i) = rho_bl * vey_bl;
            cons(IEN, k, j, i) = 0.5 * rho_bl * (vex_bl * vex_bl + vey_bl * vey_bl) + pre_bl / (gamma - 1.0);
          } else {  // Top-left quadrant
            cons(IDN, k, j, i) = rho_tl;
            cons(IM1, k, j, i) = rho_tl * vex_tl;
            cons(IM2, k, j, i) = rho_tl * vey_tl;
            cons(IEN, k, j, i) = 0.5 * rho_tl * (vex_tl * vex_tl + vey_tl * vey_tl) + pre_tl / (gamma - 1.0);
          }
        } else {  // Right half
          if (coords.Xc<2>(j) < y_discont) {  // Bottom-right quadrant
            cons(IDN, k, j, i) = rho_br;
            cons(IM1, k, j, i) = rho_br * vex_br;
            cons(IM2, k, j, i) = rho_br * vey_br;
            cons(IEN, k, j, i) = 0.5 * rho_br * (vex_br * vex_br + vey_br * vey_br) + pre_br / (gamma - 1.0);
          } else {  // Top-right quadrant
            cons(IDN, k, j, i) = rho_tr;
            cons(IM1, k, j, i) = rho_tr * vex_tr;
            cons(IM2, k, j, i) = rho_tr * vey_tr;
            cons(IEN, k, j, i) = 0.5 * rho_tr * (vex_tr * vex_tr + vey_tr * vey_tr) + pre_tr / (gamma - 1.0);
          }
        }

      });
}
} // namespace riemann_2d