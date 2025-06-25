
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"

namespace brio_wu {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real rho_l = pin->GetOrAddReal("problem/brio_wu", "rho_l",  1.0);
  Real vex_l = pin->GetOrAddReal("problem/brio_wu", "vex_l", -0.4);
  Real vey_l = pin->GetOrAddReal("problem/brio_wu", "vey_l",  0.0);
  Real vez_l = pin->GetOrAddReal("problem/brio_wu", "vez_l",  0.0);
  Real pre_l = pin->GetOrAddReal("problem/brio_wu", "pre_l",  1.0);
  Real Bx_l  = pin->GetOrAddReal("problem/brio_wu", "Bx_l",  0.75);
  Real By_l  = pin->GetOrAddReal("problem/brio_wu", "By_l",   1.0);
  Real Bz_l  = pin->GetOrAddReal("problem/brio_wu", "Bz_l",   0.0);

  Real rho_r = pin->GetOrAddReal("problem/brio_wu", "rho_r",  0.2);
  Real vex_r = pin->GetOrAddReal("problem/brio_wu", "vex_r", -0.4);
  Real vey_r = pin->GetOrAddReal("problem/brio_wu", "vey_r",  0.0);
  Real vez_r = pin->GetOrAddReal("problem/brio_wu", "vez_r",  0.0);
  Real pre_r = pin->GetOrAddReal("problem/brio_wu", "pre_r",  0.1);
  Real Bx_r  = pin->GetOrAddReal("problem/brio_wu", "Bx_r",  0.75);
  Real By_r  = pin->GetOrAddReal("problem/brio_wu", "By_r",  -1.0);
  Real Bz_r  = pin->GetOrAddReal("problem/brio_wu", "Bz_r",   0.0);

  Real x_discont = pin->GetOrAddReal("problem/brio_wu", "x_discont", 0.0);

  Real gamma = pin->GetReal("hydro", "gamma");

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &cons = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "Init briowu", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (coords.Xc<1>(i) < x_discont) {
          cons(IDN, k, j, i) = rho_l;
          cons(IM1, k, j, i) = rho_l * vex_l;
          cons(IM2, k, j, i) = rho_l * vey_l;
          cons(IM3, k, j, i) = rho_l * vez_l;
          cons(IEN, k, j, i) = 0.5 * rho_l * (vex_l * vex_l + vey_l * vey_l + vez_l * vez_l) + 0.5 * (Bx_l * Bx_l + By_l * By_l + Bz_l * Bz_l) + pre_l / (gamma - 1.0); 
          cons(IB1, k, j, i) = Bx_l;
          cons(IB2, k, j, i) = By_l;
          cons(IB3, k, j, i) = Bz_l;

        } else {
          cons(IDN, k, j, i) = rho_r;
          cons(IM1, k, j, i) = rho_r * vex_r;
          cons(IM2, k, j, i) = rho_r * vey_r;
          cons(IM3, k, j, i) = rho_r * vez_r;
          cons(IEN, k, j, i) = 0.5 * rho_r * (vex_r * vex_r + vey_r * vey_r + vez_r * vez_r) + 0.5 * (Bx_r * Bx_r + By_r * By_r + Bz_r * Bz_r) + pre_r / (gamma - 1.0);
          cons(IB1, k, j, i) = Bx_r;
          cons(IB2, k, j, i) = By_r;
          cons(IB3, k, j, i) = Bz_r;
        }
      });
}
} // namespace briowu