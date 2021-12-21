//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file blast.cpp
//  \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//         cylindrical, and spherical coordinates.  Contains post-processing code
//         to check whether blast is spherical for regression tests
//
// REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

// Parthenon headers
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../main.hpp"
#include "parthenon/prelude.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

namespace blast {

//========================================================================================
//! \fn void ProblemGenerator(MeshBlock &pmb, ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rout = pin->GetReal("problem", "radius_outer");
  Real rin = rout - pin->GetOrAddReal("problem", "radius_inner", rout);
  Real pa = pin->GetOrAddReal("problem", "pamb", 1.0);
  Real da = pin->GetOrAddReal("problem", "damb", 1.0);
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5 / 3);
  Real gm1 = gamma - 1.0;

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x1_0 = pin->GetOrAddReal("problem", "x1_0", 0.0);
  Real x2_0 = pin->GetOrAddReal("problem", "x2_0", 0.0);
  Real x3_0 = pin->GetOrAddReal("problem", "x3_0", 0.0);
  Real x0, y0, z0;
  x0 = x1_0;
  y0 = x2_0;
  z0 = x3_0;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;
  // setup uniform ambient medium with spherical over-pressured region
  pmb->par_for(
      "ProblemGenerator Blast", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real den = da;
        Real pres = pa;
        Real x = coords.x1v(i);
        Real y = coords.x2v(j);
        Real z = coords.x3v(k);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));

        if (rad < rout) {
          if (rad < rin) {
            den = drat * da;
            pres = prat * pa;
          } else { // add smooth ramp in pressure
            Real f = (rad - rin) / (rout - rin);
            Real log_den = (1.0 - f) * std::log(drat * da) + f * std::log(da);
            den = std::exp(log_den);
            Real log_pres = (1.0 - f) * std::log(prat * pa) + f * std::log(pa);
            pres = std::exp(log_pres);
          }
        }
        u(IDN, k, j, i) = den;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = pres / gm1;
      });
}

} // namespace blast
