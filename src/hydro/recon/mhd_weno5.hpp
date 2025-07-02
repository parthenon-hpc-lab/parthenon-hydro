//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mhd_weno5.hpp
//  \brief Lax-Friedrichs flux splitting for hydrodynamics
//
// Computes 1D Fluxes using a Finite Difference Lax-Friedrichs Flux Vector Splitting method
// Following Procedure 2.10 from the following paper:
// "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws"
// By: Chi-Wang Shu
// https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf


#ifndef MHD_WENO5_HPP_
#define MHD_WENO5_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()
#include <iomanip>   // For std::setprecision

// Athena headers
#include "../../main.hpp"
// #include "weno_helpers.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

//----------------------------------------------------------------------------------------
//! \fn void MHD::LaxFriedrichsFlux
//  \brief The Lax-Friedrichs Flux Vector Splitting solver for magnetohydrodynamics (adiabatic)

template <>
struct Reconstruct<Fluid::mhd, Reconstruction::weno5> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const parthenon::VariablePack<Real> &q,
        VariableFluxPack<Real> &cons, const AdiabaticMHDEOS &eos) {

    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;

    constexpr int NMHD = 8;

    const auto gamma = eos.GetGamma();
    const auto gm1 = gamma - 1.0;
    const auto igm1 = 1.0 / gm1;
    
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
   
      Real w0[(NMHD)], w1[(NMHD)], w2[(NMHD)], w3[(NMHD)], w4[(NMHD)], w5[(NMHD)];
      Real q0[(NMHD)], q1[(NMHD)], q2[(NMHD)], q3[(NMHD)], q4[(NMHD)], q5[(NMHD)];
      Real f0[(NMHD)], f1[(NMHD)], f2[(NMHD)], f3[(NMHD)], f4[(NMHD)], f5[(NMHD)];
      Real rr[(NMHD)][(NMHD)], ru[(NMHD)][(NMHD)], lu[(NMHD)][(NMHD)], lq[(NMHD)][(NMHD)];
      Real vj0[NMHD], vj1[NMHD], vj2[NMHD], vj3[NMHD], vj4[NMHD], vj5[NMHD];
      Real gj0[NMHD], gj1[NMHD], gj2[NMHD], gj3[NMHD], gj4[NMHD], gj5[NMHD];
      Real g_p0[NMHD], g_p1[NMHD], g_p2[NMHD], g_p3[NMHD], g_p4[NMHD], g_p5[NMHD];
      Real g_m0[NMHD], g_m1[NMHD], g_m2[NMHD], g_m3[NMHD], g_m4[NMHD], g_m5[NMHD];
      Real weno1[NMHD], weno2[NMHD], weno_sum[NMHD];
      Real f_half[NMHD];

      //--- Step 0.  Load states into local variables:
        
      if (ivx == IV1){
        w0[IDN] = q(IDN, k, j, i - 3); 
        w0[IV1] = q(ivx, k, j, i - 3);
        w0[IV2] = q(ivy, k, j, i - 3);
        w0[IV3] = q(ivz, k, j, i - 3);
        w0[IPR] = q(IPR, k, j, i - 3);
        w0[IB1] = q(iBx, k, j, i - 3);
        w0[IB2] = q(iBy, k, j, i - 3);
        w0[IB3] = q(iBz, k, j, i - 3);

        q0[IDN] = cons(IDN, k, j, i - 3);
        q0[IM1] = cons(ivx, k, j, i - 3);
        q0[IM2] = cons(ivy, k, j, i - 3);
        q0[IM3] = cons(ivz, k, j, i - 3);
        q0[IEN] = cons(IEN, k, j, i - 3);
        q0[IB1] = cons(iBx, k, j, i - 3);
        q0[IB2] = cons(iBy, k, j, i - 3);
        q0[IB3] = cons(iBz, k, j, i - 3);

        w1[IDN] = q(IDN, k, j, i - 2);
        w1[IV1] = q(ivx, k, j, i - 2);
        w1[IV2] = q(ivy, k, j, i - 2);
        w1[IV3] = q(ivz, k, j, i - 2);
        w1[IPR] = q(IPR, k, j, i - 2);
        w1[IB1] = q(iBx, k, j, i - 2);
        w1[IB2] = q(iBy, k, j, i - 2);
        w1[IB3] = q(iBz, k, j, i - 2);

        q1[IDN] = cons(IDN, k, j, i - 2);
        q1[IM1] = cons(ivx, k, j, i - 2);
        q1[IM2] = cons(ivy, k, j, i - 2);
        q1[IM3] = cons(ivz, k, j, i - 2);
        q1[IEN] = cons(IEN, k, j, i - 2);
        q1[IB1] = cons(iBx, k, j, i - 2);
        q1[IB2] = cons(iBy, k, j, i - 2);
        q1[IB3] = cons(iBz, k, j, i - 2);

        w2[IDN] = q(IDN, k, j, i - 1);
        w2[IV1] = q(ivx, k, j, i - 1);
        w2[IV2] = q(ivy, k, j, i - 1);
        w2[IV3] = q(ivz, k, j, i - 1);
        w2[IPR] = q(IPR, k, j, i - 1);
        w2[IB1] = q(iBx, k, j, i - 1);
        w2[IB2] = q(iBy, k, j, i - 1);
        w2[IB3] = q(iBz, k, j, i - 1);

        q2[IDN] = cons(IDN, k, j, i - 1);
        q2[IM1] = cons(ivx, k, j, i - 1);
        q2[IM2] = cons(ivy, k, j, i - 1);
        q2[IM3] = cons(ivz, k, j, i - 1);
        q2[IEN] = cons(IEN, k, j, i - 1);
        q2[IB1] = cons(iBx, k, j, i - 1);
        q2[IB2] = cons(iBy, k, j, i - 1);
        q2[IB3] = cons(iBz, k, j, i - 1);

        w3[IDN] = q(IDN, k, j, i);
        w3[IV1] = q(ivx, k, j, i);
        w3[IV2] = q(ivy, k, j, i);
        w3[IV3] = q(ivz, k, j, i);
        w3[IPR] = q(IPR, k, j, i);
        w3[IB1] = q(iBx, k, j, i);
        w3[IB2] = q(iBy, k, j, i);
        w3[IB3] = q(iBz, k, j, i);

        q3[IDN] = cons(IDN, k, j, i);
        q3[IM1] = cons(ivx, k, j, i);
        q3[IM2] = cons(ivy, k, j, i);
        q3[IM3] = cons(ivz, k, j, i);
        q3[IEN] = cons(IEN, k, j, i);
        q3[IB1] = cons(iBx, k, j, i);
        q3[IB2] = cons(iBy, k, j, i);
        q3[IB3] = cons(iBz, k, j, i);

        w4[IDN] = q(IDN, k, j, i + 1);
        w4[IV1] = q(ivx, k, j, i + 1);
        w4[IV2] = q(ivy, k, j, i + 1);
        w4[IV3] = q(ivz, k, j, i + 1);
        w4[IPR] = q(IPR, k, j, i + 1);
        w4[IB1] = q(iBx, k, j, i + 1);
        w4[IB2] = q(iBy, k, j, i + 1);
        w4[IB3] = q(iBz, k, j, i + 1);

        q4[IDN] = cons(IDN, k, j, i + 1);
        q4[IM1] = cons(ivx, k, j, i + 1);
        q4[IM2] = cons(ivy, k, j, i + 1);
        q4[IM3] = cons(ivz, k, j, i + 1);
        q4[IEN] = cons(IEN, k, j, i + 1);
        q4[IB1] = cons(iBx, k, j, i + 1);
        q4[IB2] = cons(iBy, k, j, i + 1);
        q4[IB3] = cons(iBz, k, j, i + 1);

        w5[IDN] = q(IDN, k, j, i + 2);
        w5[IV1] = q(ivx, k, j, i + 2);
        w5[IV2] = q(ivy, k, j, i + 2);
        w5[IV3] = q(ivz, k, j, i + 2);
        w5[IPR] = q(IPR, k, j, i + 2);
        w5[IB1] = q(iBx, k, j, i + 2);
        w5[IB2] = q(iBy, k, j, i + 2);
        w5[IB3] = q(iBz, k, j, i + 2);        
        
        q5[IDN] = cons(IDN, k, j, i + 2);
        q5[IM1] = cons(ivx, k, j, i + 2);
        q5[IM2] = cons(ivy, k, j, i + 2);
        q5[IM3] = cons(ivz, k, j, i + 2);
        q5[IEN] = cons(IEN, k, j, i + 2);
        q5[IB1] = cons(iBx, k, j, i + 2);
        q5[IB2] = cons(iBy, k, j, i + 2);
        q5[IB3] = cons(iBz, k, j, i + 2);
      }
        
      if (ivx == IV2){
        w0[IDN] = q(IDN, k, j - 3, i);
        w0[IV1] = q(ivx, k, j - 3, i);
        w0[IV2] = q(ivy, k, j - 3, i);
        w0[IV3] = q(ivz, k, j - 3, i);
        w0[IPR] = q(IPR, k, j - 3, i);
        w0[IB1] = q(iBx, k, j - 3, i);
        w0[IB2] = q(iBy, k, j - 3, i);
        w0[IB3] = q(iBz, k, j - 3, i);

        q0[IDN] = cons(IDN, k, j - 3, i);
        q0[IM1] = cons(ivx, k, j - 3, i);
        q0[IM2] = cons(ivy, k, j - 3, i);
        q0[IM3] = cons(ivz, k, j - 3, i);
        q0[IEN] = cons(IEN, k, j - 3, i);
        q0[IB1] = cons(iBx, k, j - 3, i);
        q0[IB2] = cons(iBy, k, j - 3, i);
        q0[IB3] = cons(iBz, k, j - 3, i); 

        w1[IDN] = q(IDN, k, j - 2, i);
        w1[IV1] = q(ivx, k, j - 2, i);
        w1[IV2] = q(ivy, k, j - 2, i);
        w1[IV3] = q(ivz, k, j - 2, i);
        w1[IPR] = q(IPR, k, j - 2, i);
        w1[IB1] = q(iBx, k, j - 2, i);
        w1[IB2] = q(iBy, k, j - 2, i);
        w1[IB3] = q(iBz, k, j - 2, i);

        q1[IDN] = cons(IDN, k, j - 2, i);
        q1[IM1] = cons(ivx, k, j - 2, i);
        q1[IM2] = cons(ivy, k, j - 2, i);
        q1[IM3] = cons(ivz, k, j - 2, i);
        q1[IEN] = cons(IEN, k, j - 2, i);
        q1[IB1] = cons(iBx, k, j - 2, i);
        q1[IB2] = cons(iBy, k, j - 2, i);
        q1[IB3] = cons(iBz, k, j - 2, i);

        w2[IDN] = q(IDN, k, j - 1, i);
        w2[IV1] = q(ivx, k, j - 1, i);
        w2[IV2] = q(ivy, k, j - 1, i);
        w2[IV3] = q(ivz, k, j - 1, i);
        w2[IPR] = q(IPR, k, j - 1, i);
        w2[IB1] = q(iBx, k, j - 1, i);
        w2[IB2] = q(iBy, k, j - 1, i);
        w2[IB3] = q(iBz, k, j - 1, i);

        q2[IDN] = cons(IDN, k, j - 1, i);
        q2[IM1] = cons(ivx, k, j - 1, i);
        q2[IM2] = cons(ivy, k, j - 1, i);
        q2[IM3] = cons(ivz, k, j - 1, i);
        q2[IEN] = cons(IEN, k, j - 1, i );
        q2[IB1] = cons(iBx, k, j - 1, i);
        q2[IB2] = cons(iBy, k, j - 1, i);
        q2[IB3] = cons(iBz, k, j - 1, i);

        w3[IDN] = q(IDN, k, j, i);
        w3[IV1] = q(ivx, k, j, i);
        w3[IV2] = q(ivy, k, j, i);
        w3[IV3] = q(ivz, k, j, i);
        w3[IPR] = q(IPR, k, j, i);
        w3[IB1] = q(iBx, k, j, i);
        w3[IB2] = q(iBy, k, j, i);
        w3[IB3] = q(iBz, k, j, i);

        q3[IDN] = cons(IDN, k, j, i);
        q3[IM1] = cons(ivx, k, j, i);
        q3[IM2] = cons(ivy, k, j, i);
        q3[IM3] = cons(ivz, k, j, i);
        q3[IEN] = cons(IEN, k, j, i);
        q3[IB1] = cons(iBx, k, j, i);
        q3[IB2] = cons(iBy, k, j, i);
        q3[IB3] = cons(iBz, k, j, i);

        w4[IDN] = q(IDN, k, j + 1, i);
        w4[IV1] = q(ivx, k, j + 1, i);
        w4[IV2] = q(ivy, k, j + 1, i);
        w4[IV3] = q(ivz, k, j + 1, i);
        w4[IPR] = q(IPR, k, j + 1, i);
        w4[IB1] = q(iBx, k, j + 1, i);
        w4[IB2] = q(iBy, k, j + 1, i);
        w4[IB3] = q(iBz, k, j + 1, i);

        q4[IDN] = cons(IDN, k, j + 1, i);
        q4[IM1] = cons(ivx, k, j + 1, i);
        q4[IM2] = cons(ivy, k, j + 1, i);
        q4[IM3] = cons(ivz, k, j + 1, i);
        q4[IEN] = cons(IEN, k, j + 1, i);
        q4[IB1] = cons(iBx, k, j + 1, i);
        q4[IB2] = cons(iBy, k, j + 1, i);
        q4[IB3] = cons(iBz, k, j + 1, i);

        w5[IDN] = q(IDN, k, j + 2, i);
        w5[IV1] = q(ivx, k, j + 2, i);
        w5[IV2] = q(ivy, k, j + 2, i);
        w5[IV3] = q(ivz, k, j + 2, i);
        w5[IPR] = q(IPR, k, j + 2, i);
        w5[IB1] = q(iBx, k, j + 2, i);
        w5[IB2] = q(iBy, k, j + 2, i);
        w5[IB3] = q(iBz, k, j + 2, i);

        q5[IDN] = cons(IDN, k, j + 2, i);
        q5[IM1] = cons(ivx, k, j + 2, i);
        q5[IM2] = cons(ivy, k, j + 2, i);
        q5[IM3] = cons(ivz, k, j + 2, i);
        q5[IEN] = cons(IEN, k, j + 2, i);
        q5[IB1] = cons(iBx, k, j + 2, i);
        q5[IB2] = cons(iBy, k, j + 2, i);
        q5[IB3] = cons(iBz, k, j + 2, i);
      }

      if (ivx == IV3){
        w0[IDN] = q(IDN, k - 3, j, i);
        w0[IV1] = q(ivx, k - 3, j, i);
        w0[IV2] = q(ivy, k - 3, j, i);
        w0[IV3] = q(ivz, k - 3, j, i);
        w0[IPR] = q(IPR, k - 3, j, i);
        w0[IB1] = q(iBx, k - 3, j, i);
        w0[IB2] = q(iBy, k - 3, j, i);
        w0[IB3] = q(iBz, k - 3, j, i);

        q0[IDN] = cons(IDN, k - 3, j, i);
        q0[IM1] = cons(ivx, k - 3, j, i);
        q0[IM2] = cons(ivy, k - 3, j, i);
        q0[IM3] = cons(ivz, k - 3, j, i);
        q0[IEN] = cons(IEN, k - 3, j, i);
        q0[IB1] = cons(iBx, k - 3, j, i);
        q0[IB2] = cons(iBy, k - 3, j, i);
        q0[IB3] = cons(iBz, k - 3, j, i);

        w1[IDN] = q(IDN, k - 2, j, i);
        w1[IV1] = q(ivx, k - 2, j, i);
        w1[IV2] = q(ivy, k - 2, j, i);
        w1[IV3] = q(ivz, k - 2, j, i);
        w1[IPR] = q(IPR, k - 2, j, i);
        w1[IB1] = q(iBx, k - 2, j, i);
        w1[IB2] = q(iBy, k - 2, j, i);
        w1[IB3] = q(iBz, k - 2, j, i);

        q1[IDN] = cons(IDN, k - 2, j, i);
        q1[IM1] = cons(ivx, k - 2, j, i);
        q1[IM2] = cons(ivy, k - 2, j, i);
        q1[IM3] = cons(ivz, k - 2, j, i);
        q1[IEN] = cons(IEN, k - 2, j, i);
        q1[IB1] = cons(iBx, k - 2, j, i);
        q1[IB2] = cons(iBy, k - 2, j, i);
        q1[IB3] = cons(iBz, k - 2, j, i);

        w2[IDN] = q(IDN, k - 1, j, i);
        w2[IV1] = q(ivx, k - 1, j, i);
        w2[IV2] = q(ivy, k - 1, j, i);
        w2[IV3] = q(ivz, k - 1, j, i);
        w2[IPR] = q(IPR, k - 1, j, i);
        w2[IB1] = q(iBx, k - 1, j, i);
        w2[IB2] = q(iBy, k - 1, j, i);
        w2[IB3] = q(iBz, k - 1, j, i);

        q2[IDN] = cons(IDN, k - 1, j, i);
        q2[IM1] = cons(ivx, k - 1, j, i);
        q2[IM2] = cons(ivy, k - 1, j, i);
        q2[IM3] = cons(ivz, k - 1, j, i);
        q2[IEN] = cons(IEN, k - 1, j, i);
        q2[IB1] = cons(iBx, k - 1, j, i);
        q2[IB2] = cons(iBy, k - 1, j, i);
        q2[IB3] = cons(iBz, k - 1, j, i);
        
        w3[IDN] = q(IDN, k, j, i);
        w3[IV1] = q(ivx, k, j, i);
        w3[IV2] = q(ivy, k, j, i);
        w3[IV3] = q(ivz, k, j, i);
        w3[IPR] = q(IPR, k, j, i);
        w3[IB1] = q(iBx, k, j, i);
        w3[IB2] = q(iBy, k, j, i);
        w3[IB3] = q(iBz, k, j, i);

        q3[IDN] = cons(IDN, k, j, i);
        q3[IM1] = cons(ivx, k, j, i);
        q3[IM2] = cons(ivy, k, j, i);
        q3[IM3] = cons(ivz, k, j, i);
        q3[IEN] = cons(IEN, k, j, i);
        q3[IB1] = cons(iBx, k, j, i);
        q3[IB2] = cons(iBy, k, j, i);
        q3[IB3] = cons(iBz, k, j, i);

        w4[IDN] = q(IDN, k + 1, j, i);
        w4[IV1] = q(ivx, k + 1, j, i);
        w4[IV2] = q(ivy, k + 1, j, i);
        w4[IV3] = q(ivz, k + 1, j, i);
        w4[IPR] = q(IPR, k + 1, j, i);
        w4[IB1] = q(iBx, k + 1, j, i);
        w4[IB2] = q(iBy, k + 1, j, i);
        w4[IB3] = q(iBz, k + 1, j, i);

        q4[IDN] = cons(IDN, k + 1, j, i);
        q4[IM1] = cons(ivx, k + 1, j, i);
        q4[IM2] = cons(ivy, k + 1, j, i);
        q4[IM3] = cons(ivz, k + 1, j, i);
        q4[IEN] = cons(IEN, k + 1, j, i);
        q4[IB1] = cons(iBx, k + 1, j, i);
        q4[IB2] = cons(iBy, k + 1, j, i);
        q4[IB3] = cons(iBz, k + 1, j, i);

        w5[IDN] = q(IDN, k + 2, j, i);
        w5[IV1] = q(ivx, k + 2, j, i);
        w5[IV2] = q(ivy, k + 2, j, i);
        w5[IV3] = q(ivz, k + 2, j, i);
        w5[IPR] = q(IPR, k + 2, j, i);
        w5[IB1] = q(iBx, k + 2, j, i);
        w5[IB2] = q(iBy, k + 2, j, i);
        w5[IB3] = q(iBz, k + 2, j, i);

        q5[IDN] = cons(IDN, k + 2, j, i);
        q5[IM1] = cons(ivx, k + 2, j, i);
        q5[IM2] = cons(ivy, k + 2, j, i);
        q5[IM3] = cons(ivz, k + 2, j, i);
        q5[IEN] = cons(IEN, k + 2, j, i);
        q5[IB1] = cons(iBx, k + 2, j, i);
        q5[IB2] = cons(iBy, k + 2, j, i);
        q5[IB3] = cons(iBz, k + 2, j, i);
      }

      //--- Step 1.  Compute the physical flux at each grid point:
      Real Bnm0 = w0[IB1] * w0[IB1] + w0[IB2] * w0[IB2] + w0[IB3] * w0[IB3];
      f0[IDN] = q0[IM1];
      f0[IM1] = q0[IM1] * w0[IV1] + w0[IPR] + 0.5 * Bnm0 - w0[IB1] * w0[IB1];
      f0[IM2] = q0[IM1] * w0[IV2] - w0[IB1] * w0[IB2];
      f0[IM3] = q0[IM1] * w0[IV3] - w0[IB1] * w0[IB3];
      f0[IEN] = w0[IV1] * (q0[IEN] + w0[IPR] + 0.5 * Bnm0) - w0[IB1] * (w0[IV1] * w0[IB1] + w0[IV2] * w0[IB2] + w0[IV3] * w0[IB3]);
      f0[IB1] = 0.0;
      f0[IB2] = w0[IV1] * w0[IB2] - w0[IV2] * w0[IB1];
      f0[IB3] = w0[IV1] * w0[IB3] - w0[IV3] * w0[IB1];

      Real Bnm1 = w1[IB1] * w1[IB1] + w1[IB2] * w1[IB2] + w1[IB3] * w1[IB3];
      f1[IDN] = q1[IM1];
      f1[IM1] = q1[IM1] * w1[IV1] + w1[IPR] + 0.5 * Bnm1 - w1[IB1] * w1[IB1];
      f1[IM2] = q1[IM1] * w1[IV2] - w1[IB1] * w1[IB2];
      f1[IM3] = q1[IM1] * w1[IV3] - w1[IB1] * w1[IB3];
      f1[IEN] = w1[IV1] * (q1[IEN] + w1[IPR] + 0.5 * Bnm1) - w1[IB1] * (w1[IV1] * w1[IB1] + w1[IV2] * w1[IB2] + w1[IV3] * w1[IB3]);
      f1[IB1] = 0.0;
      f1[IB2] = w1[IV1] * w1[IB2] - w1[IV2] * w1[IB1];
      f1[IB3] = w1[IV1] * w1[IB3] - w1[IV3] * w1[IB1];

      Real Bnm2 = w2[IB1] * w2[IB1] + w2[IB2] * w2[IB2] + w2[IB3] * w2[IB3];
      f2[IDN] = q2[IM1];
      f2[IM1] = q2[IM1] * w2[IV1] + w2[IPR] + 0.5 * Bnm2 - w2[IB1] * w2[IB1];
      f2[IM2] = q2[IM1] * w2[IV2] - w2[IB1] * w2[IB2];
      f2[IM3] = q2[IM1] * w2[IV3] - w2[IB1] * w2[IB3];
      f2[IEN] = w2[IV1] * (q2[IEN] + w2[IPR] + 0.5 * Bnm2) - w2[IB1] * (w2[IV1] * w2[IB1] + w2[IV2] * w2[IB2] + w2[IV3] * w2[IB3]);
      f2[IB1] = 0.0;
      f2[IB2] = w2[IV1] * w2[IB2] - w2[IV2] * w2[IB1];
      f2[IB3] = w2[IV1] * w2[IB3] - w2[IV3] * w2[IB1];

      Real Bnm3 = w3[IB1] * w3[IB1] + w3[IB2] * w3[IB2] + w3[IB3] * w3[IB3];
      f3[IDN] = q3[IM1];
      f3[IM1] = q3[IM1] * w3[IV1] + w3[IPR] + 0.5 * Bnm3 - w3[IB1] * w3[IB1];
      f3[IM2] = q3[IM1] * w3[IV2] - w3[IB1] * w3[IB2];
      f3[IM3] = q3[IM1] * w3[IV3] - w3[IB1] * w3[IB3];
      f3[IEN] = w3[IV1] * (q3[IEN] + w3[IPR] + 0.5 * Bnm3) - w3[IB1] * (w3[IV1] * w3[IB1] + w3[IV2] * w3[IB2] + w3[IV3] * w3[IB3]);
      f3[IB1] = 0.0;
      f3[IB2] = w3[IV1] * w3[IB2] - w3[IV2] * w3[IB1];
      f3[IB3] = w3[IV1] * w3[IB3] - w3[IV3] * w3[IB1];

      Real Bnm4 = w4[IB1] * w4[IB1] + w4[IB2] * w4[IB2] + w4[IB3] * w4[IB3];
      f4[IDN] = q4[IM1];
      f4[IM1] = q4[IM1] * w4[IV1] + w4[IPR] + 0.5 * Bnm4 - w4[IB1] * w4[IB1];
      f4[IM2] = q4[IM1] * w4[IV2] - w4[IB1] * w4[IB2];
      f4[IM3] = q4[IM1] * w4[IV3] - w4[IB1] * w4[IB3];
      f4[IEN] = w4[IV1] * (q4[IEN] + w4[IPR] + 0.5 * Bnm4) - w4[IB1] * (w4[IV1] * w4[IB1] + w4[IV2] * w4[IB2] + w4[IV3] * w4[IB3]);
      f4[IB1] = 0.0;
      f4[IB2] = w4[IV1] * w4[IB2] - w4[IV2] * w4[IB1];
      f4[IB3] = w4[IV1] * w4[IB3] - w4[IV3] * w4[IB1];

      Real Bnm5 = w5[IB1] * w5[IB1] + w5[IB2] * w5[IB2] + w5[IB3] * w5[IB3];
      f5[IDN] = q5[IM1];
      f5[IM1] = q5[IM1] * w5[IV1] + w5[IPR] + 0.5 * Bnm5 - w5[IB1] * w5[IB1];
      f5[IM2] = q5[IM1] * w5[IV2] - w5[IB1] * w5[IB2];
      f5[IM3] = q5[IM1] * w5[IV3] - w5[IB1] * w5[IB3];
      f5[IEN] = w5[IV1] * (q5[IEN] + w5[IPR] + 0.5 * Bnm5) - w5[IB1] * (w5[IV1] * w5[IB1] + w5[IV2] * w5[IB2] + w5[IV3] * w5[IB3]);
      f5[IB1] = 0.0;
      f5[IB2] = w5[IV1] * w5[IB2] - w5[IV2] * w5[IB1];
      f5[IB3] = w5[IV1] * w5[IB3] - w5[IV3] * w5[IB1];

      //--- Step 2.  At each x_{i+1/2,j,k}:
      //--- (a) Compute the average state w_{i+1/2,j,k} in the primitive variables:

      Real half_den = 0.5 * (w2[IDN] + w3[IDN]);
      Real half_vex = 0.5 * (w2[IV1] + w3[IV1]);
      Real half_vey = 0.5 * (w2[IV2] + w3[IV2]);
      Real half_vez = 0.5 * (w2[IV3] + w3[IV3]);
      Real half_pre = 0.5 * (w2[IPR] + w3[IPR]);
      Real half_Bx  = 0.5 * (w2[IB1] + w3[IB1]);
      Real half_By  = 0.5 * (w2[IB2] + w3[IB2]);
      Real half_Bz  = 0.5 * (w2[IB3] + w3[IB3]);


      //--- (b) Compute the right and left eigenvectors of the flux Jacobian matrix, ∂f/∂x, at x = x_{i+1/2,j,k}:

      // See Section 1.5 (Scaling Theorem Example: Magnetohydrodynamic Equations) of:
      // "Numerical Methods for Gasdynamic Systems on Unstructured Meshes"
      // By Timothy J. Barth
      // Entropy Scaled Eigenvectors of the Modified MHD Equations
      // Equations (54) - (57)

      Real t1 = 0.0;
      Real half_Bnm = std::sqrt(half_Bz * half_Bz + half_By * half_By);
      Real t2, t3;

      if (half_Bnm != 0) {
        t2 = half_By / half_Bnm;
        t3 = half_Bz / half_Bnm;
      } else {
        t2 = std::sin(M_PI / 4.0);
        t3 = std::cos(M_PI / 4.0);
      }

      Real n1 = 1.0;
      Real n2 = 0.0;
      Real n3 = 0.0;

      Real rhosq = std::sqrt(half_den); 
      Real presq = std::sqrt(half_pre); 
      Real a2 = (gamma * half_pre) / half_den; 
      Real a = std::sqrt(a2); 
      Real sqg2 = std::sqrt(1.0 / (2.0 * gamma)); 
      Real sq12 = std::sqrt(0.5); 
      Real twosq = std::sqrt(2.0); 
      Real sqpr = std::sqrt(half_pre) / half_den; 
      Real sqpor = std::sqrt(half_pre / half_den); 
      Real sq1og = std::sqrt(1.0 / gamma); 
      Real sqgam = std::sqrt(gm1 / gamma);
      Real b1s = half_Bx / rhosq; 
      Real b2s = half_By / rhosq; 
      Real b3s = half_Bz / rhosq; 
      Real BNs = b1s * n1 + b2s * n2 + b3s * n3; 
      Real BN = half_Bx * n1 + half_By * n2 + half_Bz * n3; 
      Real d = a2 + (b1s * b1s + b2s * b2s + b3s * b3s); 
      Real cf = std::sqrt(0.5 * std::abs(d + std::sqrt(d * d - 4.0 * a2 * (BNs * BNs)))); 
      Real cs = std::sqrt(0.5 * std::abs(d - std::sqrt(d * d - 4.0 * a2 * (BNs * BNs)))); 
      Real cf2 = 0.5 * std::abs(d + std::sqrt(d * d - 4.0 * a2 * (BNs * BNs))); 
      Real cs2 = 0.5 * std::abs(d - std::sqrt(d * d - 4.0 * a2 * (BNs * BNs)));       
      Real beta1 = ((b1s * n1 + b2s * n2 + b3s * n3) >= 0.0) ? 1.0 : -1.0;

      Real alphaf, alphas;
      if (std::abs(cf * cf - cs * cs) <= 1.0e-12) {
        alphaf = std::sin(std::atan(1.0) / 2.0);
        alphas = std::cos(std::atan(1.0) / 2.0);
      } else {
        alphaf = std::sqrt(std::abs(a2 - cs * cs)) / std::sqrt(std::abs(cf * cf - cs * cs));
        alphas = std::sqrt(std::abs(cf * cf - a2)) / std::sqrt(std::abs(cf * cf - cs * cs));
      }

      Real TxN1 = n3 * t2 - n2 * t3; 
      Real TxN2 = n1 * t3 - n3 * t1; 
      Real TxN3 = n2 * t1 - n1 * t2; 

      Real BT = half_Bx * t1 + half_By * t2 + half_Bz * t3; 

      //--- Right Eigenvectors 

      // 1 - Right Eigenvector Entropy Wave
      ru[0][0] = std::sqrt(gm1 / gamma) * rhosq;
      ru[1][0] = 0.0;
      ru[2][0] = 0.0;
      ru[3][0] = 0.0;
      ru[4][0] = 0.0;
      ru[5][0] = 0.0;
      ru[6][0] = 0.0;
      ru[7][0] = 0.0;

      // 2 - Right Eigenvector Divergence Wave
      ru[0][1] = 0.0;
      ru[1][1] = 0.0;
      ru[2][1] = 0.0;
      ru[3][1] = 0.0;
      ru[4][1] = 0.0;
      ru[5][1] = sq1og * a * n1;
      ru[6][1] = sq1og * a * n2;
      ru[7][1] = sq1og * a * n3;

      // 3 - Right Eigenvector Alfven Wave
      ru[0][2] = 0.0;
      ru[1][2] = -sq12 * (sqpr * TxN1);
      ru[2][2] = -sq12 * (sqpr * TxN2);
      ru[3][2] = -sq12 * (sqpr * TxN3);
      ru[4][2] = 0.0;
      ru[5][2] = sq12 * sqpor * TxN1;
      ru[6][2] = sq12 * sqpor * TxN2;
      ru[7][2] = sq12 * sqpor * TxN3;

      // 4 - Right Eigenvector Alfven Wave
      ru[0][3] =  ru[0][2];
      ru[1][3] = -ru[1][2];
      ru[2][3] = -ru[2][2];
      ru[3][3] = -ru[3][2];
      ru[4][3] =  ru[4][2];
      ru[5][3] =  ru[5][2];
      ru[6][3] =  ru[6][2];
      ru[7][3] =  ru[7][2];

      // 5 - Right Eigenvector Fast Magneto-acoustic Wave
      Real bst = b1s * t1 + b2s * t2 + b3s * t3;
      ru[0][4] = sqg2 * alphaf * rhosq;
      ru[1][4] = sqg2 * ((alphaf * a2 * n1 + alphas * a * ((bst) * n1 - (BNs) * t1))) / (rhosq * cf);
      ru[2][4] = sqg2 * ((alphaf * a2 * n2 + alphas * a * ((bst) * n2 - (BNs) * t2))) / (rhosq * cf);
      ru[3][4] = sqg2 * ((alphaf * a2 * n3 + alphas * a * ((bst) * n3 - (BNs) * t3))) / (rhosq * cf);
      ru[4][4] = sqg2 * alphaf * rhosq * a2;
      ru[5][4] = sqg2 * alphas * a * t1;
      ru[6][4] = sqg2 * alphas * a * t2;
      ru[7][4] = sqg2 * alphas * a * t3;

      // 6 - Right Eigenvector Fast Magneto-acoustic Wave
      ru[0][5] =  ru[0][4];
      ru[1][5] = -ru[1][4];
      ru[2][5] = -ru[2][4];
      ru[3][5] = -ru[3][4];
      ru[4][5] =  ru[4][4];
      ru[5][5] =  ru[5][4];
      ru[6][5] =  ru[6][4];
      ru[7][5] =  ru[7][4];

      // 7 - Right Eigenvector Slow Magneto-acoustic Wave
      ru[0][6] = sqg2 * alphas * rhosq;
      ru[1][6] = sqg2 * beta1 * (alphaf * cf * cf * t1 + alphas * a * (BNs) * n1) / (rhosq * cf);
      ru[2][6] = sqg2 * beta1 * (alphaf * cf * cf * t2 + alphas * a * (BNs) * n2) / (rhosq * cf);
      ru[3][6] = sqg2 * beta1 * (alphaf * cf * cf * t3 + alphas * a * (BNs) * n3) / (rhosq * cf);
      ru[4][6] = sqg2 * alphas * rhosq * a2;
      ru[5][6] = -sqg2 * alphaf * a * t1;
      ru[6][6] = -sqg2 * alphaf * a * t2;
      ru[7][6] = -sqg2 * alphaf * a * t3;

      // 8 - Right Eigenvector Slow Magneto-acoustic Wave
      ru[0][7] =  ru[0][6];
      ru[1][7] = -ru[1][6];
      ru[2][7] = -ru[2][6];
      ru[3][7] = -ru[3][6];
      ru[4][7] =  ru[4][6];
      ru[5][7] =  ru[5][6];
      ru[6][7] =  ru[6][6];
      ru[7][7] =  ru[7][6];

      for (int m = 0; m < 8; ++m) {
        rr[0][m] = ru[0][m] / gm1;
        rr[1][m] = (ru[0][m] * half_vex + ru[1][m] * half_den) / gm1;
        rr[2][m] = (ru[0][m] * half_vey + ru[2][m] * half_den) / gm1;
        rr[3][m] = (ru[0][m] * half_vez + ru[3][m] * half_den) / gm1;
        rr[4][m] = (ru[4][m] / gm1 + half_Bx * ru[5][m] + half_By * ru[6][m] + half_Bz * ru[7][m] + 0.5 * ru[0][m] * (half_vex * half_vex + half_vey * half_vey + half_vez * half_vez) + ru[1][m] * half_vex * half_den + ru[2][m] * half_vey * half_den + ru[3][m] * half_vez * half_den) / gm1;
        rr[5][m] = ru[5][m] / gm1;
        rr[6][m] = ru[6][m] / gm1;
        rr[7][m] = ru[7][m] / gm1;
      }

      //--- Left Eigenvectors 
      
      // 1 - Left Eigenvector
      lu[0][0] = 1.0 / (sqgam * rhosq);
      lu[0][1] = 0.0;
      lu[0][2] = 0.0;
      lu[0][3] = 0.0;
      lu[0][4] = -1.0 / (a2 * sqgam * rhosq);
      lu[0][5] = 0.0;
      lu[0][6] = 0.0;
      lu[0][7] = 0.0;
        
      // 2 - Left Eigenvector
      Real nen = n1*n1*(t2*t2+t3*t3) + n2*n2*(t1*t1+t3*t3) + n3*n3*(t1*t1+t2*t2) - 2.0*n2*n3*t2*t3 - 2.0*n1*n3*t1*t3 - 2.0*n1*n2*t1*t2;
      Real nen2 = a * sq1og * nen;
      lu[1][0] = 0.0;
      lu[1][1] = 0.0;
      lu[1][2] = 0.0;
      lu[1][3] = 0.0;
      lu[1][4] = 0.0;
      lu[1][5] = (n1*(t2*t2 + t3*t3) - t1*(n2*t2 + n3*t3))/nen2;
      lu[1][6] = (n2*(t1*t1 + t3*t3) - t2*(n1*t1 + n3*t3))/nen2;
      lu[1][7] = (n3*(t1*t1 + t2*t2) - t3*(n1*t1 + n2*t2))/nen2;

      // 3 - Left Eigenvector
      Real nen3 = twosq * presq * nen;
      Real nen31 = nen3 / rhosq;
      lu[2][0] = 0.0;
      lu[2][1] = half_den * (-TxN1) / nen3;
      lu[2][2] = half_den * (-TxN2) / nen3;
      lu[2][3] = half_den * (-TxN3) / nen3;    
      lu[2][4] = 0.0;
      lu[2][5] = TxN1 / nen31;
      lu[2][6] = TxN2 / nen31;
      lu[2][7] = TxN3 / nen31; 

      // 4 - Left Eigenvector
      lu[3][0] =  lu[2][0];
      lu[3][1] = -lu[2][1];
      lu[3][2] = -lu[2][2];
      lu[3][3] = -lu[2][3];
      lu[3][4] =  lu[2][4];
      lu[3][5] =  lu[2][5];
      lu[3][6] =  lu[2][6];
      lu[3][7] =  lu[2][7];

      // 5 - Left Eigenvector
      Real Term51 = half_den*cf*( rhosq*cf*cf*alphaf*( -t1*(n2*t2 + n3*t3) + n1*(t2*t2+t3*t3) ) - a*BN*alphas*( n2*TxN3 - n3*TxN2 ) );
      Real Term52 = half_den*cf*( rhosq*cf*cf*alphaf*( -t2*(n1*t1 + n3*t3) + n2*(t1*t1+t3*t3) ) - a*BN*alphas*( n3*TxN1 - n1*TxN3 ) );
      Real Term53 = half_den*cf*( rhosq*cf*cf*alphaf*( -t3*(n1*t1 + n2*t2) + n3*(t1*t1+t2*t2) ) - a*BN*alphas*( n1*TxN2 - n2*TxN1 ) );
      
      Real Term54 = alphaf / (twosq * a * a * sq1og * rhosq * (alphaf * alphaf + alphas * alphas));

      Real Term55 = alphas*( n2*TxN3 - n3*TxN2 );
      Real Term56 = alphas*( n3*TxN1 - n1*TxN3 );
      Real Term57 = alphas*( n1*TxN2 - n2*TxN1 );

      Real nen51 = twosq * a * nen * sq1og * (a * BN * BN * alphas * alphas + rhosq * cf * cf * alphaf * (a * rhosq * alphaf + BT * alphas));

      Real nen52 = twosq * a * sq1og * (alphaf * alphaf + alphas * alphas) * nen;

      lu[4][0] = 0.0;
      lu[4][1] = Term51 / nen51;
      lu[4][2] = Term52 / nen51;
      lu[4][3] = Term53 / nen51;
      lu[4][4] = Term54;
      lu[4][5] = Term55 / nen52;
      lu[4][6] = Term56 / nen52;
      lu[4][7] = Term57 / nen52;

      // 6 - Left Eigenvector
      lu[5][0] =  lu[4][0];
      lu[5][1] = -lu[4][1];
      lu[5][2] = -lu[4][2];
      lu[5][3] = -lu[4][3];
      lu[5][4] =  lu[4][4];
      lu[5][5] =  lu[4][5];
      lu[5][6] =  lu[4][6];
      lu[5][7] =  lu[4][7];

      // 7 - Left Eigenvector
      Real Term71 = half_den*cf*( a*rhosq*alphaf*( n2*TxN3 - n3*TxN2 ) + alphas*( ( half_By*TxN2 + half_Bz*TxN3 )*(-TxN1) + half_Bx*( n1*n1*(t2*t2 + t3*t3) + t1*t1*(n2*n2 + n3*n3) - 2.0*n1*t1*(n2*t2 + n3*t3) ) ) );
      Real Term72 = half_den*cf*( a*rhosq*alphaf*( n3*TxN1 - n1*TxN3 ) + alphas*( ( half_Bz*TxN3 + half_Bx*TxN1 )*(-TxN2) + half_By*( n2*n2*(t3*t3 + t1*t1) + t2*t2*(n3*n3 + n1*n1) - 2.0*n2*t2*(n3*t3 + n1*t1) ) ) );
      Real Term73 = half_den*cf*( a*rhosq*alphaf*( n1*TxN2 - n2*TxN1 ) + alphas*( ( half_Bx*TxN1 + half_By*TxN2 )*(-TxN3) + half_Bz*( n3*n3*(t1*t1 + t2*t2) + t3*t3*(n1*n1 + n2*n2) - 2.0*n3*t3*(n1*t1 + n2*t2) ) ) );

      Real Term74 =  alphas / (twosq * a * a * sq1og * rhosq * (alphaf * alphaf + alphas * alphas));

      Real Term75 = -alphaf*( n2*TxN3 - n3*TxN2 );
      Real Term76 = -alphaf*( n3*TxN1 - n1*TxN3 );
      Real Term77 = -alphaf*( n1*TxN2 - n2*TxN1 );

      Real nen71 = twosq * beta1 * nen * sq1og * (a * BN * BN * alphas * alphas + rhosq * cf * cf * alphaf * (a * rhosq * alphaf + BT * alphas));

      Real nen72 = nen52;

      lu[6][0] = 0.0;
      lu[6][1] = Term71 / nen71;
      lu[6][2] = Term72 / nen71;
      lu[6][3] = Term73 / nen71;
      lu[6][4] = Term74;
      lu[6][5] = Term75 / nen72;
      lu[6][6] = Term76 / nen72;
      lu[6][7] = Term77 / nen72;

      // 8 - Left Eigenvector
      lu[7][0] =  lu[6][0];
      lu[7][1] = -lu[6][1];
      lu[7][2] = -lu[6][2];
      lu[7][3] = -lu[6][3];
      lu[7][4] =  lu[6][4];
      lu[7][5] =  lu[6][5];
      lu[7][6] =  lu[6][6];
      lu[7][7] =  lu[6][7];

      for (int mm = 0; mm < 8; ++mm) {
        lq[mm][0] =  lu[mm][0] * gm1 - lu[mm][1] * half_vex * gm1 / half_den - lu[mm][2] * half_vey * gm1 / half_den - lu[mm][3] * half_vez * gm1 / half_den + lu[mm][4] * gm1 * gm1 * (half_vex * half_vex + half_vey * half_vey + half_vez * half_vez) * 0.5;
        lq[mm][1] = -lu[mm][4] * half_vex * gm1 * gm1 + lu[mm][1] * gm1 / half_den;
        lq[mm][2] = -lu[mm][4] * half_vey * gm1 * gm1 + lu[mm][2] * gm1 / half_den;
        lq[mm][3] = -lu[mm][4] * half_vez * gm1 * gm1 + lu[mm][3] * gm1 / half_den;
        lq[mm][4] =  lu[mm][4] * gm1 * gm1;
        lq[mm][5] =  lu[mm][5] * gm1 - half_Bx * lu[mm][4] * gm1 * gm1;
        lq[mm][6] =  lu[mm][6] * gm1 - half_By * lu[mm][4] * gm1 * gm1;
        lq[mm][7] =  lu[mm][7] * gm1 - half_Bz * lu[mm][4] * gm1 * gm1;
      }

      //--- (c) Project the solution and physical flux into the right eigenvector space:

      for (int ii = 0; ii < NMHD; ++ii) {

        vj0[ii] = 0.0;
        vj1[ii] = 0.0;
        vj2[ii] = 0.0;
        vj3[ii] = 0.0;
        vj4[ii] = 0.0;
        vj5[ii] = 0.0;

        gj0[ii] = 0.0;
        gj1[ii] = 0.0;
        gj2[ii] = 0.0;
        gj3[ii] = 0.0;
        gj4[ii] = 0.0;
        gj5[ii] = 0.0;

        for (int jj = 0; jj < NMHD; ++jj) {

          vj0[ii] += lq[ii][jj] * q0[jj];
          vj1[ii] += lq[ii][jj] * q1[jj];
          vj2[ii] += lq[ii][jj] * q2[jj];
          vj3[ii] += lq[ii][jj] * q3[jj];
          vj4[ii] += lq[ii][jj] * q4[jj];
          vj5[ii] += lq[ii][jj] * q5[jj];

          gj0[ii] += lq[ii][jj] * f0[jj];
          gj1[ii] += lq[ii][jj] * f1[jj];
          gj2[ii] += lq[ii][jj] * f2[jj];
          gj3[ii] += lq[ii][jj] * f3[jj];
          gj4[ii] += lq[ii][jj] * f4[jj];
          gj5[ii] += lq[ii][jj] * f5[jj];

        }
      }

      //--- (d) Perform a Lax-Friedrichs flux vector splitting for each component of the characteristic variables:
      // Specifically, assume that the mth components of Vj and Gj are vj and gj, respectively, then compute
      // g^{±}_{j}= 0.5 * (g_j ± α^{m} v_j) where α(m) = max_k | λ^{m} q_k | 
      // is the maximal wave speed of the m^{th} component of characteristic variables over all grid points

      Real aa0 = std::sqrt((gamma * w0[IPR]) / w0[IDN]);
      Real aa1 = std::sqrt((gamma * w1[IPR]) / w1[IDN]);
      Real aa2 = std::sqrt((gamma * w2[IPR]) / w2[IDN]);
      Real aa3 = std::sqrt((gamma * w3[IPR]) / w3[IDN]);
      Real aa4 = std::sqrt((gamma * w4[IPR]) / w4[IDN]);
      Real aa5 = std::sqrt((gamma * w5[IPR]) / w5[IDN]);

      Real ca0 = std::sqrt((w0[IB1] * w0[IB1] + w0[IB2] * w0[IB2] + w0[IB3] * w0[IB3]) / w0[IDN]);
      Real ca1 = std::sqrt((w1[IB1] * w1[IB1] + w1[IB2] * w1[IB2] + w1[IB3] * w1[IB3]) / w1[IDN]);
      Real ca2 = std::sqrt((w2[IB1] * w2[IB1] + w2[IB2] * w2[IB2] + w2[IB3] * w2[IB3]) / w2[IDN]);
      Real ca3 = std::sqrt((w3[IB1] * w3[IB1] + w3[IB2] * w3[IB2] + w3[IB3] * w3[IB3]) / w3[IDN]);
      Real ca4 = std::sqrt((w4[IB1] * w4[IB1] + w4[IB2] * w4[IB2] + w4[IB3] * w4[IB3]) / w4[IDN]);
      Real ca5 = std::sqrt((w5[IB1] * w5[IB1] + w5[IB2] * w5[IB2] + w5[IB3] * w5[IB3]) / w5[IDN]);

      Real cax0 = std::sqrt((w0[IB1] * w0[IB1]) / w0[IDN]);
      Real cax1 = std::sqrt((w1[IB1] * w1[IB1]) / w1[IDN]);
      Real cax2 = std::sqrt((w2[IB1] * w2[IB1]) / w2[IDN]);
      Real cax3 = std::sqrt((w3[IB1] * w3[IB1]) / w3[IDN]);
      Real cax4 = std::sqrt((w4[IB1] * w4[IB1]) / w4[IDN]);
      Real cax5 = std::sqrt((w5[IB1] * w5[IB1]) / w5[IDN]);

      Real cfx0 = std::sqrt(0.5 * std::abs(aa0 * aa0 + ca0 * ca0 + std::sqrt((aa0 * aa0 + ca0 * ca0) * (aa0 * aa0 + ca0 * ca0) - (4 * aa0 * aa0 * cax0 * cax0))));
      Real cfx1 = std::sqrt(0.5 * std::abs(aa1 * aa1 + ca1 * ca1 + std::sqrt((aa1 * aa1 + ca1 * ca1) * (aa1 * aa1 + ca1 * ca1) - (4 * aa1 * aa1 * cax1 * cax1))));
      Real cfx2 = std::sqrt(0.5 * std::abs(aa2 * aa2 + ca2 * ca2 + std::sqrt((aa2 * aa2 + ca2 * ca2) * (aa2 * aa2 + ca2 * ca2) - (4 * aa2 * aa2 * cax2 * cax2))));
      Real cfx3 = std::sqrt(0.5 * std::abs(aa3 * aa3 + ca3 * ca3 + std::sqrt((aa3 * aa3 + ca3 * ca3) * (aa3 * aa3 + ca3 * ca3) - (4 * aa3 * aa3 * cax3 * cax3))));
      Real cfx4 = std::sqrt(0.5 * std::abs(aa4 * aa4 + ca4 * ca4 + std::sqrt((aa4 * aa4 + ca4 * ca4) * (aa4 * aa4 + ca4 * ca4) - (4 * aa4 * aa4 * cax4 * cax4))));
      Real cfx5 = std::sqrt(0.5 * std::abs(aa5 * aa5 + ca5 * ca5 + std::sqrt((aa5 * aa5 + ca5 * ca5) * (aa5 * aa5 + ca5 * ca5) - (4 * aa5 * aa5 * cax5 * cax5))));
      
      Real csx0 = std::sqrt(0.5 * std::abs(aa0 * aa0 + ca0 * ca0 - std::sqrt((aa0 * aa0 + ca0 * ca0) * (aa0 * aa0 + ca0 * ca0) - (4 * aa0 * aa0 * cax0 * cax0))));
      Real csx1 = std::sqrt(0.5 * std::abs(aa1 * aa1 + ca1 * ca1 - std::sqrt((aa1 * aa1 + ca1 * ca1) * (aa1 * aa1 + ca1 * ca1) - (4 * aa1 * aa1 * cax1 * cax1))));
      Real csx2 = std::sqrt(0.5 * std::abs(aa2 * aa2 + ca2 * ca2 - std::sqrt((aa2 * aa2 + ca2 * ca2) * (aa2 * aa2 + ca2 * ca2) - (4 * aa2 * aa2 * cax2 * cax2))));
      Real csx3 = std::sqrt(0.5 * std::abs(aa3 * aa3 + ca3 * ca3 - std::sqrt((aa3 * aa3 + ca3 * ca3) * (aa3 * aa3 + ca3 * ca3) - (4 * aa3 * aa3 * cax3 * cax3))));
      Real csx4 = std::sqrt(0.5 * std::abs(aa4 * aa4 + ca4 * ca4 - std::sqrt((aa4 * aa4 + ca4 * ca4) * (aa4 * aa4 + ca4 * ca4) - (4 * aa4 * aa4 * cax4 * cax4))));
      Real csx5 = std::sqrt(0.5 * std::abs(aa5 * aa5 + ca5 * ca5 - std::sqrt((aa5 * aa5 + ca5 * ca5) * (aa5 * aa5 + ca5 * ca5) - (4 * aa5 * aa5 * cax5 * cax5))));

      Real em = 1.0e-15;

      Real max_eig_1 = std::max({em, std::abs(w0[IV1]),      std::abs(w1[IV1]),      std::abs(w2[IV1]),      std::abs(w3[IV1]),      std::abs(w4[IV1]),      std::abs(w5[IV1])});
      Real max_eig_2 = std::max({em, std::abs(w0[IV1]),      std::abs(w1[IV1]),      std::abs(w2[IV1]),      std::abs(w3[IV1]),      std::abs(w4[IV1]),      std::abs(w5[IV1])});
      Real max_eig_3 = std::max({em, std::abs(w0[IV1]+cax0), std::abs(w1[IV1]+cax1), std::abs(w2[IV1]+cax2), std::abs(w3[IV1]+cax3), std::abs(w4[IV1]+cax4), std::abs(w5[IV1]+cax5)});
      Real max_eig_4 = std::max({em, std::abs(w0[IV1]-cax0), std::abs(w1[IV1]-cax1), std::abs(w2[IV1]-cax2), std::abs(w3[IV1]-cax3), std::abs(w4[IV1]-cax4), std::abs(w5[IV1]-cax5)});
      Real max_eig_5 = std::max({em, std::abs(w0[IV1]+cfx0), std::abs(w1[IV1]+cfx1), std::abs(w2[IV1]+cfx2), std::abs(w3[IV1]+cfx3), std::abs(w4[IV1]+cfx4), std::abs(w5[IV1]+cfx5)});
      Real max_eig_6 = std::max({em, std::abs(w0[IV1]-cfx0), std::abs(w1[IV1]-cfx1), std::abs(w2[IV1]-cfx2), std::abs(w3[IV1]-cfx3), std::abs(w4[IV1]-cfx4), std::abs(w5[IV1]-cfx5)});
      Real max_eig_7 = std::max({em, std::abs(w0[IV1]+csx0), std::abs(w1[IV1]+csx1), std::abs(w2[IV1]+csx2), std::abs(w3[IV1]+csx3), std::abs(w4[IV1]+csx4), std::abs(w5[IV1]+csx5)});
      Real max_eig_8 = std::max({em, std::abs(w0[IV1]-csx0), std::abs(w1[IV1]-csx1), std::abs(w2[IV1]-csx2), std::abs(w3[IV1]-csx3), std::abs(w4[IV1]-csx4), std::abs(w5[IV1]-csx5)});

      Real alpha[NMHD] = {max_eig_1, max_eig_2, max_eig_3, max_eig_4, max_eig_5, max_eig_6, max_eig_7, max_eig_8};


      for (int iii = 0; iii < NMHD; ++iii) {

        g_p0[iii] = 0.5 * (gj0[iii] + 1.1 * alpha[iii] * vj0[iii]);
        g_p1[iii] = 0.5 * (gj1[iii] + 1.1 * alpha[iii] * vj1[iii]);  
        g_p2[iii] = 0.5 * (gj2[iii] + 1.1 * alpha[iii] * vj2[iii]);  
        g_p3[iii] = 0.5 * (gj3[iii] + 1.1 * alpha[iii] * vj3[iii]);  
        g_p4[iii] = 0.5 * (gj4[iii] + 1.1 * alpha[iii] * vj4[iii]);  
        // gp5[iii] = 0.5 * (gj5[iii] + 1.1 * alpha[iii] * vj5[iii]);   // not needed for WENO

        // gm0[iii] = 0.5 * (gj0[iii] - 1.1 * alpha[iii] * vj0[iii]);   // not needed for WENO 
        g_m1[iii] = 0.5 * (gj1[iii] - 1.1 * alpha[iii] * vj1[iii]);  
        g_m2[iii] = 0.5 * (gj2[iii] - 1.1 * alpha[iii] * vj2[iii]);  
        g_m3[iii] = 0.5 * (gj3[iii] - 1.1 * alpha[iii] * vj3[iii]);  
        g_m4[iii] = 0.5 * (gj4[iii] - 1.1 * alpha[iii] * vj4[iii]);  
        g_m5[iii] = 0.5 * (gj5[iii] - 1.1 * alpha[iii] * vj5[iii]);  

      }

      //--- (e) Perform a WENO reconstruction on each of the computed flux components gj± to obtain 
      // the corresponding component of the numerical flux

      Real epsilon = 1E-6;

      // TODO(wendelnc) Make WENO5 a function in weno_helpers.hpp

      // gp0,gp1,gp2,gp3,gp4 are vmm,vm,v,vp,vpp
      for (int jjj = 0; jjj < NMHD; ++jjj) {     

        Real q_im2 = g_p0[jjj];
        Real q_im1 = g_p1[jjj];
        Real q_i   = g_p2[jjj];
        Real q_ip1 = g_p3[jjj];
        Real q_ip2 = g_p4[jjj];

        Real beta[3]; // (2.63) 
        beta[0] = (13.0/12.0)*SQR(q_i - 2*q_ip1 + q_ip2) + (1.0/4.0)*SQR(3*q_i - 4*q_ip1 + q_ip2);
        beta[1] = (13.0/12.0)*SQR(q_im1 - 2*q_i + q_ip1) + (1.0/4.0)*SQR(q_im1 - q_ip1);
        beta[2] = (13.0/12.0)*SQR(q_im2 - 2*q_im1 + q_i) + (1.0/4.0)*SQR(q_im2 - 4*q_im1 + 3*q_i);

        Real indicator[3]; // fraction part of (2.59) 
        indicator[0] = 1 / SQR(epsilon + beta[0]);
        indicator[1] = 1 / SQR(epsilon + beta[1]);
        indicator[2] = 1 / SQR(epsilon + beta[2]);

        // compute qL_ip1
        Real f[3]; // polynomial based on constants c_{r,j} in Table 2.1 
        // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
        f[0] = 2*q_i + 5*q_ip1 - q_ip2;
        f[1] = -1*q_im1 + 5*q_i + 2*q_ip1;
        f[2] = 2*q_im2 - 7*q_im1 + 11*q_i;

        Real alpha[3]; // (2.59) & below (2.54)
        alpha[0] = indicator[0] * 3.0 / 10.0;
        alpha[1] = indicator[1] * 6.0 / 10.0;
        alpha[2] = indicator[2] * 1.0 / 10.0;
        Real alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

        weno1[jjj] = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2]) / alpha_sum; // (2.52) 

      }

      // gm5,gm4,gm3,gm2,gm1 are vmm,vm,v,vp,vpp
      for (int jjjj = 0; jjjj < NMHD; ++jjjj) {

        Real q_im2 = g_m5[jjjj];
        Real q_im1 = g_m4[jjjj];
        Real q_i   = g_m3[jjjj];
        Real q_ip1 = g_m2[jjjj];
        Real q_ip2 = g_m1[jjjj];

        Real beta[3]; // (2.63) 
        beta[0] = (13.0/12.0)*SQR(q_i - 2*q_ip1 + q_ip2) + (1.0/4.0)*SQR(3*q_i - 4*q_ip1 + q_ip2);
        beta[1] = (13.0/12.0)*SQR(q_im1 - 2*q_i + q_ip1) + (1.0/4.0)*SQR(q_im1 - q_ip1);
        beta[2] = (13.0/12.0)*SQR(q_im2 - 2*q_im1 + q_i) + (1.0/4.0)*SQR(q_im2 - 4*q_im1 + 3*q_i);

        Real indicator[3]; // fraction part of (2.59) 
        indicator[0] = 1 / SQR(epsilon + beta[0]);
        indicator[1] = 1 / SQR(epsilon + beta[1]);
        indicator[2] = 1 / SQR(epsilon + beta[2]);

        // compute qL_ip1
        Real f[3]; // polynomial based on constants c_{r,j} in Table 2.1 
        // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
        f[0] = 2*q_i + 5*q_ip1 - q_ip2;
        f[1] = -1*q_im1 + 5*q_i + 2*q_ip1;
        f[2] = 2*q_im2 - 7*q_im1 + 11*q_i;

        Real alpha[3]; // (2.59) & below (2.54)
        alpha[0] = indicator[0] * 3.0 / 10.0;
        alpha[1] = indicator[1] * 6.0 / 10.0;
        alpha[2] = indicator[2] * 1.0 / 10.0;
        Real alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

        weno2[jjjj] = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2]) / alpha_sum; // (2.52) 
        
        weno_sum[jjjj] = weno1[jjjj] + weno2[jjjj];

      }

      //--- (f) Project the numerical flux back to the conserved variables
        
      for (int iiii = 0; iiii < NMHD; ++iiii) {
        f_half[iiii] = 0.0;
        for (int jjjjj = 0; jjjjj < NMHD; ++jjjjj) {
          f_half[iiii] += rr[iiii][jjjjj] * weno_sum[jjjjj];
        }
      }

      //--- Step 3.  Update flux at each x_{i+1/2,j,k}:

      cons.flux(ivx, IDN, k, j, i) = f_half[IDN];
      cons.flux(ivx, ivx, k, j, i) = f_half[IV1];
      cons.flux(ivx, ivy, k, j, i) = f_half[IV2];
      cons.flux(ivx, ivz, k, j, i) = f_half[IV3];
      cons.flux(ivx, IEN, k, j, i) = f_half[IEN];
      cons.flux(ivx, iBx, k, j, i) = f_half[IB1];
      cons.flux(ivx, iBy, k, j, i) = f_half[IB2];
      cons.flux(ivx, iBz, k, j, i) = f_half[IB3];

    });
  }
};

#endif // MHD_WENO5_HPP_            