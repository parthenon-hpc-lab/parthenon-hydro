//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro_weno5.hpp
//  \brief Lax-Friedrichs flux splitting for hydrodynamics
//
// Computes 1D Fluxes using a Finite Difference Lax-Friedrichs Flux Vector Splitting method
// Following Procedure 2.10 from the following paper:
// "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws"
// By: Chi-Wang Shu
// https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf


#ifndef HYDRO_WENO5_HPP_
#define HYDRO_WENO5_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()
#include <iomanip>   // For std::setprecision
#include <fstream>   // For output

// Athena headers
#include "../../main.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

//----------------------------------------------------------------------------------------
//! \fn void Hydro::LaxFriedrichsFlux
//  \brief The Lax-Friedrichs Flux Vector Splitting solver for hydrodynamics (adiabatic)

template <>
struct Reconstruct<Fluid::euler, Reconstruction::weno5> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const parthenon::VariablePack<Real> &q,
        VariableFluxPack<Real> &cons, const AdiabaticHydroEOS &eos) {

    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    Real gamma;
    gamma = eos.GetGamma();
    Real gm1 = gamma - 1.0;
    Real igm1 = 1.0 / gm1;
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
    
      Real w0[(NHYDRO)], w1[(NHYDRO)], w2[(NHYDRO)], w3[(NHYDRO)], w4[(NHYDRO)], w5[(NHYDRO)];
      Real q0[(NHYDRO)], q1[(NHYDRO)], q2[(NHYDRO)], q3[(NHYDRO)], q4[(NHYDRO)], q5[(NHYDRO)];
      Real f0[(NHYDRO)], f1[(NHYDRO)], f2[(NHYDRO)], f3[(NHYDRO)], f4[(NHYDRO)], f5[(NHYDRO)];
      Real eigenvalues[(NHYDRO)], right_eigenmatrix[(NHYDRO)][(NHYDRO)], left_eigenmatrix[(NHYDRO)][(NHYDRO)];
      Real vj0[NHYDRO], vj1[NHYDRO], vj2[NHYDRO], vj3[NHYDRO], vj4[NHYDRO], vj5[NHYDRO];
      Real gj0[NHYDRO], gj1[NHYDRO], gj2[NHYDRO], gj3[NHYDRO], gj4[NHYDRO], gj5[NHYDRO];
      Real g_p0[NHYDRO], g_p1[NHYDRO], g_p2[NHYDRO], g_p3[NHYDRO], g_p4[NHYDRO], g_p5[NHYDRO];
      Real g_m0[NHYDRO], g_m1[NHYDRO], g_m2[NHYDRO], g_m3[NHYDRO], g_m4[NHYDRO], g_m5[NHYDRO];
      Real weno1[NHYDRO], weno2[NHYDRO], weno_sum[NHYDRO];
      Real f_half[NHYDRO]; 
   
      //--- Step 0.  Load states into local variables:
        
      if (ivx == IV1){
        w0[IDN] = q(IDN, k, j, i - 3);
        w0[IV1] = q(ivx, k, j, i - 3);
        w0[IV2] = q(ivy, k, j, i - 3);
        w0[IV3] = q(ivz, k, j, i - 3);
        w0[IPR] = q(IPR, k, j, i - 3);

        q0[IDN] = cons(IDN, k, j, i - 3);
        q0[IM1] = cons(ivx, k, j, i - 3);
        q0[IM2] = cons(ivy, k, j, i - 3);
        q0[IM3] = cons(ivz, k, j, i - 3);
        q0[IEN] = cons(IEN, k, j, i - 3);

        w1[IDN] = q(IDN, k, j, i - 2);
        w1[IV1] = q(ivx, k, j, i - 2);
        w1[IV2] = q(ivy, k, j, i - 2);
        w1[IV3] = q(ivz, k, j, i - 2);
        w1[IPR] = q(IPR, k, j, i - 2);

        q1[IDN] = cons(IDN, k, j, i - 2);
        q1[IM1] = cons(ivx, k, j, i - 2);
        q1[IM2] = cons(ivy, k, j, i - 2);
        q1[IM3] = cons(ivz, k, j, i - 2);
        q1[IEN] = cons(IEN, k, j, i - 2);

        w2[IDN] = q(IDN, k, j, i - 1);
        w2[IV1] = q(ivx, k, j, i - 1);
        w2[IV2] = q(ivy, k, j, i - 1);
        w2[IV3] = q(ivz, k, j, i - 1);
        w2[IPR] = q(IPR, k, j, i - 1);

        q2[IDN] = cons(IDN, k, j, i - 1);
        q2[IM1] = cons(ivx, k, j, i - 1);
        q2[IM2] = cons(ivy, k, j, i - 1);
        q2[IM3] = cons(ivz, k, j, i - 1);
        q2[IEN] = cons(IEN, k, j, i - 1);
    
        w3[IDN] = q(IDN, k, j, i);
        w3[IV1] = q(ivx, k, j, i);
        w3[IV2] = q(ivy, k, j, i);
        w3[IV3] = q(ivz, k, j, i);
        w3[IPR] = q(IPR, k, j, i);

        q3[IDN] = cons(IDN, k, j, i);
        q3[IM1] = cons(ivx, k, j, i);
        q3[IM2] = cons(ivy, k, j, i);
        q3[IM3] = cons(ivz, k, j, i);
        q3[IEN] = cons(IEN, k, j, i);

        w4[IDN] = q(IDN, k, j, i + 1);
        w4[IV1] = q(ivx, k, j, i + 1);
        w4[IV2] = q(ivy, k, j, i + 1);
        w4[IV3] = q(ivz, k, j, i + 1);
        w4[IPR] = q(IPR, k, j, i + 1);

        q4[IDN] = cons(IDN, k, j, i + 1);
        q4[IM1] = cons(ivx, k, j, i + 1);
        q4[IM2] = cons(ivy, k, j, i + 1);
        q4[IM3] = cons(ivz, k, j, i + 1);
        q4[IEN] = cons(IEN, k, j, i + 1);

        w5[IDN] = q(IDN, k, j, i + 2);
        w5[IV1] = q(ivx, k, j, i + 2);
        w5[IV2] = q(ivy, k, j, i + 2);
        w5[IV3] = q(ivz, k, j, i + 2);
        w5[IPR] = q(IPR, k, j, i + 2);
        
        q5[IDN] = cons(IDN, k, j, i + 2);
        q5[IM1] = cons(ivx, k, j, i + 2);
        q5[IM2] = cons(ivy, k, j, i + 2);
        q5[IM3] = cons(ivz, k, j, i + 2);
        q5[IEN] = cons(IEN, k, j, i + 2);
      }
        
      if (ivx == IV2){
        w0[IDN] = q(IDN, k, j - 3, i);
        w0[IV1] = q(ivx, k, j - 3, i);
        w0[IV2] = q(ivy, k, j - 3, i);
        w0[IV3] = q(ivz, k, j - 3, i);
        w0[IPR] = q(IPR, k, j - 3, i);

        q0[IDN] = cons(IDN, k, j - 3, i);
        q0[IM1] = cons(ivx, k, j - 3, i);
        q0[IM2] = cons(ivy, k, j - 3, i);
        q0[IM3] = cons(ivz, k, j - 3, i);
        q0[IEN] = cons(IEN, k, j - 3, i);

        w1[IDN] = q(IDN, k, j - 2, i);
        w1[IV1] = q(ivx, k, j - 2, i);
        w1[IV2] = q(ivy, k, j - 2, i);
        w1[IV3] = q(ivz, k, j - 2, i);
        w1[IPR] = q(IPR, k, j - 2, i);

        q1[IDN] = cons(IDN, k, j - 2, i);
        q1[IM1] = cons(ivx, k, j - 2, i);
        q1[IM2] = cons(ivy, k, j - 2, i);
        q1[IM3] = cons(ivz, k, j - 2, i);
        q1[IEN] = cons(IEN, k, j - 2, i);

        w2[IDN] = q(IDN, k, j - 1, i);
        w2[IV1] = q(ivx, k, j - 1, i);
        w2[IV2] = q(ivy, k, j - 1, i);
        w2[IV3] = q(ivz, k, j - 1, i);
        w2[IPR] = q(IPR, k, j - 1, i);

        q2[IDN] = cons(IDN, k, j - 1, i);
        q2[IM1] = cons(ivx, k, j - 1, i);
        q2[IM2] = cons(ivy, k, j - 1, i);
        q2[IM3] = cons(ivz, k, j - 1, i);
        q2[IEN] = cons(IEN, k, j - 1, i );
    
        w3[IDN] = q(IDN, k, j, i);
        w3[IV1] = q(ivx, k, j, i);
        w3[IV2] = q(ivy, k, j, i);
        w3[IV3] = q(ivz, k, j, i);
        w3[IPR] = q(IPR, k, j, i);

        q3[IDN] = cons(IDN, k, j, i);
        q3[IM1] = cons(ivx, k, j, i);
        q3[IM2] = cons(ivy, k, j, i);
        q3[IM3] = cons(ivz, k, j, i);
        q3[IEN] = cons(IEN, k, j, i);

        w4[IDN] = q(IDN, k, j + 1, i);
        w4[IV1] = q(ivx, k, j + 1, i);
        w4[IV2] = q(ivy, k, j + 1, i);
        w4[IV3] = q(ivz, k, j + 1, i);
        w4[IPR] = q(IPR, k, j + 1, i);

        q4[IDN] = cons(IDN, k, j + 1, i);
        q4[IM1] = cons(ivx, k, j + 1, i);
        q4[IM2] = cons(ivy, k, j + 1, i);
        q4[IM3] = cons(ivz, k, j + 1, i);
        q4[IEN] = cons(IEN, k, j + 1, i);

        w5[IDN] = q(IDN, k, j + 2, i);
        w5[IV1] = q(ivx, k, j + 2, i);
        w5[IV2] = q(ivy, k, j + 2, i);
        w5[IV3] = q(ivz, k, j + 2, i);
        w5[IPR] = q(IPR, k, j + 2, i);
        
        q5[IDN] = cons(IDN, k, j + 2, i);
        q5[IM1] = cons(ivx, k, j + 2, i);
        q5[IM2] = cons(ivy, k, j + 2, i);
        q5[IM3] = cons(ivz, k, j + 2, i);
        q5[IEN] = cons(IEN, k, j + 2, i);
      }

      if (ivx == IV3){
        w0[IDN] = q(IDN, k - 3, j, i);
        w0[IV1] = q(ivx, k - 3, j, i);
        w0[IV2] = q(ivy, k - 3, j, i);
        w0[IV3] = q(ivz, k - 3, j, i);
        w0[IPR] = q(IPR, k - 3, j, i);

        q0[IDN] = cons(IDN, k - 3, j, i);
        q0[IM1] = cons(ivx, k - 3, j, i);
        q0[IM2] = cons(ivy, k - 3, j, i);
        q0[IM3] = cons(ivz, k - 3, j, i);
        q0[IEN] = cons(IEN, k - 3, j, i);

        w1[IDN] = q(IDN, k - 2, j, i);
        w1[IV1] = q(ivx, k - 2, j, i);
        w1[IV2] = q(ivy, k - 2, j, i);
        w1[IV3] = q(ivz, k - 2, j, i);
        w1[IPR] = q(IPR, k - 2, j, i);

        q1[IDN] = cons(IDN, k - 2, j, i);
        q1[IM1] = cons(ivx, k - 2, j, i);
        q1[IM2] = cons(ivy, k - 2, j, i);
        q1[IM3] = cons(ivz, k - 2, j, i);
        q1[IEN] = cons(IEN, k - 2, j, i);

        w2[IDN] = q(IDN, k - 1, j, i);
        w2[IV1] = q(ivx, k - 1, j, i);
        w2[IV2] = q(ivy, k - 1, j, i);
        w2[IV3] = q(ivz, k - 1, j, i);
        w2[IPR] = q(IPR, k - 1, j, i);

        q2[IDN] = cons(IDN, k - 1, j, i);
        q2[IM1] = cons(ivx, k - 1, j, i);
        q2[IM2] = cons(ivy, k - 1, j, i);
        q2[IM3] = cons(ivz, k - 1, j, i);
        q2[IEN] = cons(IEN, k - 1, j, i );
    
        w3[IDN] = q(IDN, k, j, i);
        w3[IV1] = q(ivx, k, j, i);
        w3[IV2] = q(ivy, k, j, i);
        w3[IV3] = q(ivz, k, j, i);
        w3[IPR] = q(IPR, k, j, i);

        q3[IDN] = cons(IDN, k, j, i);
        q3[IM1] = cons(ivx, k, j, i);
        q3[IM2] = cons(ivy, k, j, i);
        q3[IM3] = cons(ivz, k, j, i);
        q3[IEN] = cons(IEN, k, j, i);

        w4[IDN] = q(IDN, k + 1, j, i);
        w4[IV1] = q(ivx, k + 1, j, i);
        w4[IV2] = q(ivy, k + 1, j, i);
        w4[IV3] = q(ivz, k + 1, j, i);
        w4[IPR] = q(IPR, k + 1, j, i);

        q4[IDN] = cons(IDN, k + 1, j, i);
        q4[IM1] = cons(ivx, k + 1, j, i);
        q4[IM2] = cons(ivy, k + 1, j, i);
        q4[IM3] = cons(ivz, k + 1, j, i);
        q4[IEN] = cons(IEN, k + 1, j, i);

        w5[IDN] = q(IDN, k + 2, j, i);
        w5[IV1] = q(ivx, k + 2, j, i);
        w5[IV2] = q(ivy, k + 2, j, i);
        w5[IV3] = q(ivz, k + 2, j, i);
        w5[IPR] = q(IPR, k + 2, j, i);
            
        q5[IDN] = cons(IDN, k + 2, j, i);
        q5[IM1] = cons(ivx, k + 2, j, i);
        q5[IM2] = cons(ivy, k + 2, j, i);
        q5[IM3] = cons(ivz, k + 2, j, i);
        q5[IEN] = cons(IEN, k + 2, j, i);
      }

      //--- Step 1.  Compute the physical flux at each grid point:

      f0[IDN] = q0[IM1];
      f0[IM1] = q0[IM1] * w0[IV1] + w0[IPR];
      f0[IM2] = q0[IM1] * w0[IV2];
      f0[IM3] = q0[IM1] * w0[IV3];
      f0[IEN] = w0[IV1] * (q0[IEN] + w0[IPR]);

      f1[IDN] = q1[IM1];
      f1[IM1] = q1[IM1] * w1[IV1] + w1[IPR];
      f1[IM2] = q1[IM1] * w1[IV2];
      f1[IM3] = q1[IM1] * w1[IV3];
      f1[IEN] = w1[IV1] * (q1[IEN] + w1[IPR]);

      f2[IDN] = q2[IM1];
      f2[IM1] = q2[IM1] * w2[IV1] + w2[IPR];
      f2[IM2] = q2[IM1] * w2[IV2];
      f2[IM3] = q2[IM1] * w2[IV3];
      f2[IEN] = w2[IV1] * (q2[IEN] + w2[IPR]);

      f3[IDN] = q3[IM1];
      f3[IM1] = q3[IM1] * w3[IV1] + w3[IPR];
      f3[IM2] = q3[IM1] * w3[IV2];
      f3[IM3] = q3[IM1] * w3[IV3];
      f3[IEN] = w3[IV1] * (q3[IEN] + w3[IPR]);

      f4[IDN] = q4[IM1];
      f4[IM1] = q4[IM1] * w4[IV1] + w4[IPR];
      f4[IM2] = q4[IM1] * w4[IV2];
      f4[IM3] = q4[IM1] * w4[IV3];
      f4[IEN] = w4[IV1] * (q4[IEN] + w4[IPR]);

      f5[IDN] = q5[IM1];
      f5[IM1] = q5[IM1] * w5[IV1] + w5[IPR];
      f5[IM2] = q5[IM1] * w5[IV2];
      f5[IM3] = q5[IM1] * w5[IV3];
      f5[IEN] = w5[IV1] * (q5[IEN] + w5[IPR]);

      //--- Step 2.  At each x_{i+1/2,j,k}:
      //--- (a) Compute the average state w_{i+1/2,j,k} in the primitive variables:

      Real half_den = 0.5 * (w2[IDN] + w3[IDN]);
      Real half_vex = 0.5 * (w2[IV1] + w3[IV1]);
      Real half_vey = 0.5 * (w2[IV2] + w3[IV2]);
      Real half_vez = 0.5 * (w2[IV3] + w3[IV3]);
      Real half_pre = 0.5 * (w2[IPR] + w3[IPR]);
      Real half_vsq = half_vex * half_vex + half_vey * half_vey + half_vez * half_vez;
      Real half_E = half_pre / (gm1) + (0.5 * half_den * half_vsq);
      Real half_H = (half_E + half_pre) / half_den;
      Real half_a = std::sqrt((gamma * half_pre) / half_den);

      //--- (b) Compute the right and left eigenvectors of the flux Jacobian matrix, ∂f/∂x, at x = x_{i+1/2,j,k}:

      // Right-eigenvectors, stored as COLUMNS (eq. B3)
      right_eigenmatrix[0][0] = 1.0;
      right_eigenmatrix[1][0] = half_vex - half_a;
      right_eigenmatrix[2][0] = half_vey;
      right_eigenmatrix[3][0] = half_vez;
      right_eigenmatrix[4][0] = half_H - half_vex * half_a;

      right_eigenmatrix[0][1] = 0.0;
      right_eigenmatrix[1][1] = 0.0;
      right_eigenmatrix[2][1] = 1.0;
      right_eigenmatrix[3][1] = 0.0;
      right_eigenmatrix[4][1] = half_vey;

      right_eigenmatrix[0][2] = 0.0;
      right_eigenmatrix[1][2] = 0.0;
      right_eigenmatrix[2][2] = 0.0;
      right_eigenmatrix[3][2] = 1.0;
      right_eigenmatrix[4][2] = half_vez;

      right_eigenmatrix[0][3] = 1.0;
      right_eigenmatrix[1][3] = half_vex;
      right_eigenmatrix[2][3] = half_vey;
      right_eigenmatrix[3][3] = half_vez;
      right_eigenmatrix[4][3] = 0.5 * half_vsq;

      right_eigenmatrix[0][4] = 1.0;
      right_eigenmatrix[1][4] = half_vex + half_a;
      right_eigenmatrix[2][4] = half_vey;
      right_eigenmatrix[3][4] = half_vez;
      right_eigenmatrix[4][4] = half_H + half_vex * half_a;

      // Left-eigenvectors, stored as ROWS (eq. B4)
      Real na = 0.5 / (half_a*half_a);
      left_eigenmatrix[0][0] = na * (0.5 * gm1 * half_vsq + half_vex * half_a);
      left_eigenmatrix[0][1] = -na * (gm1 * half_vex + half_a);
      left_eigenmatrix[0][2] = -na * gm1 * half_vey;
      left_eigenmatrix[0][3] = -na * gm1 * half_vez;
      left_eigenmatrix[0][4] = na * gm1;

      left_eigenmatrix[1][0] = -half_vey;
      left_eigenmatrix[1][1] = 0.0;
      left_eigenmatrix[1][2] = 1.0;
      left_eigenmatrix[1][3] = 0.0;
      left_eigenmatrix[1][4] = 0.0;

      left_eigenmatrix[2][0] = -half_vez;
      left_eigenmatrix[2][1] = 0.0;
      left_eigenmatrix[2][2] = 0.0;
      left_eigenmatrix[2][3] = 1.0;
      left_eigenmatrix[2][4] = 0.0;
        
      Real qa = gm1 / (half_a*half_a);
      left_eigenmatrix[3][0] = 1.0 - na * gm1 * half_vsq;
      left_eigenmatrix[3][1] = qa * half_vex;
      left_eigenmatrix[3][2] = qa * half_vey;
      left_eigenmatrix[3][3] = qa * half_vez;
      left_eigenmatrix[3][4] = -qa;

      left_eigenmatrix[4][0] = na * (0.5 * gm1 * half_vsq - half_vex * half_a);
      left_eigenmatrix[4][1] = -na * (gm1 * half_vex - half_a);
      left_eigenmatrix[4][2] = left_eigenmatrix[0][2];
      left_eigenmatrix[4][3] = left_eigenmatrix[0][3];
      left_eigenmatrix[4][4] = left_eigenmatrix[0][4];

      //--- (c) Project the solution and physical flux into the right eigenvector space:

      for (int ii = 0; ii < NHYDRO; ++ii) {

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

        for (int jj = 0; jj < NHYDRO; ++jj) {

          vj0[ii] += left_eigenmatrix[ii][jj] * q0[jj];
          vj1[ii] += left_eigenmatrix[ii][jj] * q1[jj];
          vj2[ii] += left_eigenmatrix[ii][jj] * q2[jj];
          vj3[ii] += left_eigenmatrix[ii][jj] * q3[jj];
          vj4[ii] += left_eigenmatrix[ii][jj] * q4[jj];
          vj5[ii] += left_eigenmatrix[ii][jj] * q5[jj];

          gj0[ii] += left_eigenmatrix[ii][jj] * f0[jj];
          gj1[ii] += left_eigenmatrix[ii][jj] * f1[jj];
          gj2[ii] += left_eigenmatrix[ii][jj] * f2[jj];
          gj3[ii] += left_eigenmatrix[ii][jj] * f3[jj];
          gj4[ii] += left_eigenmatrix[ii][jj] * f4[jj];
          gj5[ii] += left_eigenmatrix[ii][jj] * f5[jj];

        }
      }


      //--- (d) Perform a Lax-Friedrichs flux vector splitting for each component of the characteristic variables:
      // Specifically, assume that the mth components of Vj and Gj are vj and gj, respectively, then compute
      // g^{±}_{j}= 0.5 * (g_j ± α^{m} v_j) where α(m) = max_k | λ^{m} q_k | 
      // is the maximal wave speed of the m^{th} component of characteristic variables over all grid points

      Real a0 = std::sqrt((gamma * w0[IPR]) / w0[IDN]);
      Real a1 = std::sqrt((gamma * w1[IPR]) / w1[IDN]);
      Real a2 = std::sqrt((gamma * w2[IPR]) / w2[IDN]);
      Real a3 = std::sqrt((gamma * w3[IPR]) / w3[IDN]);
      Real a4 = std::sqrt((gamma * w4[IPR]) / w4[IDN]);
      Real a5 = std::sqrt((gamma * w5[IPR]) / w5[IDN]);

      Real max_eig_0 = std::max({std::abs(w0[IV1]-a0), std::abs(w1[IV1]-a1), std::abs(w2[IV1]-a2), std::abs(w3[IV1]-a3), std::abs(w4[IV1]-a4), std::abs(w5[IV1]-a5)});
      Real max_eig_1 = std::max({std::abs(w0[IV1]),    std::abs(w1[IV1]),    std::abs(w2[IV1]),    std::abs(w3[IV1]),    std::abs(w4[IV1]),    std::abs(w5[IV1])});
      Real max_eig_2 = std::max({std::abs(w0[IV1]+a0), std::abs(w1[IV1]+a1), std::abs(w2[IV1]+a2), std::abs(w3[IV1]+a3), std::abs(w4[IV1]+a4), std::abs(w5[IV1]+a5)});

      Real alpha[NHYDRO] = {max_eig_0, max_eig_1, max_eig_1, max_eig_1, max_eig_2};

      for (int iii = 0; iii < NHYDRO; ++iii) {

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
      for (int jjj = 0; jjj < NHYDRO; ++jjj) {     

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
      for (int jjjj = 0; jjjj < NHYDRO; ++jjjj) {

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
        
      for (int iiii = 0; iiii < NHYDRO; ++iiii) {
        f_half[iiii] = 0.0;
        for (int jjjjj = 0; jjjjj < NHYDRO; ++jjjjj) {
          f_half[iiii] += right_eigenmatrix[iiii][jjjjj] * weno_sum[jjjjj];
        }
      }

      //--- Step 3.  Update flux at each x_{i+1/2,j,k}:

      cons.flux(ivx, IDN, k, j, i) = f_half[IDN];
      cons.flux(ivx, ivx, k, j, i) = f_half[IV1];
      cons.flux(ivx, ivy, k, j, i) = f_half[IV2];
      cons.flux(ivx, ivz, k, j, i) = f_half[IV3];
      cons.flux(ivx, IEN, k, j, i) = f_half[IEN];
      
    });
  }
};

#endif // HYDRO_WENO5_HPP_            