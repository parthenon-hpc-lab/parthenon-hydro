#ifndef MP_WENO5_3D_HPP
#define MP_WENO5_3D_HPP


#include <iostream> 
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>  // for std::setw, std::setprecision

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "weno_helpers.hpp"

TaskStatus CalculateHJFluxes3D(std::shared_ptr<MeshData<Real>> &md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");

  auto prim_list = std::vector<std::string>({"prim"});
  auto const &prim_pack = md->PackVariables(prim_list);

  Real epsilon = 1.0E-8;
  Real nu = 0.1;
  Real delta  = 0.0;

  pmb->par_for(
  "hj flux", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
  KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
  
    const auto &coords = cons_pack.GetCoords(b);
    const auto &prim = prim_pack(b);
    auto &cons = cons_pack(b);

    auto dx = cons.GetCoords().Dxc<X1DIR>(k, j, i);
    auto dy = cons.GetCoords().Dxc<X2DIR>(k, j, i);
    auto dz = cons.GetCoords().Dxc<X3DIR>(k, j, i);

    Real max_vex = 1e-15;
    Real max_vey = 1e-15;
    Real max_vez = 1e-15;

    for (int dk = -3; dk <= 3; ++dk) {
      for (int dj = -3; dj <= 3; ++dj) {
        for (int di = -3; di <= 3; ++di) {
          const int ii = i + di;
          const int jj = j + dj;
          const int kk = k + dk;
          max_vex = std::max(max_vex, std::abs(prim(IV1, kk, jj, ii)));
          max_vey = std::max(max_vey, std::abs(prim(IV2, kk, jj, ii)));
          max_vez = std::max(max_vez, std::abs(prim(IV3, kk, jj, ii)));
        }
      }
    }

    max_vex = 1.1 * max_vex;
    max_vey = 1.1 * max_vey;
    max_vez = 1.1 * max_vez;

    //--------------------------------------------------------------------------------------
    // x direction

    // Real max_vex = std::max({std::abs(prim(IV1,k,j,i-3)),std::abs(prim(IV1,k,j,i-2)),std::abs(prim(IV1,k,j,i-1)),std::abs(prim(IV1,k,j,i)),std::abs(prim(IV1,k,j,i+1)),std::abs(prim(IV1,k,j,i+2)),std::abs(prim(IV1,k,j,i+3))});

    Real Ax_xp, Ax_xm; // A^{x+}_{xijk} and A^{x-}_{xijk}
    HJ_FLUX(Ax_xp,Ax_xm,cons(IA1,k,j,i-3),cons(IA1,k,j,i-2),cons(IA1,k,j,i-1),cons(IA1,k,j,i),cons(IA1,k,j,i+1),cons(IA1,k,j,i+2),cons(IA1,k,j,i+3),dx);

    Real Ay_xp, Ay_xm; // A^{y+}_{xijk} and A^{y-}_{xijk}
    HJ_FLUX(Ay_xp,Ay_xm,cons(IA2,k,j,i-3),cons(IA2,k,j,i-2),cons(IA2,k,j,i-1),cons(IA2,k,j,i),cons(IA2,k,j,i+1),cons(IA2,k,j,i+2),cons(IA2,k,j,i+3),dx);

    Real Az_xp, Az_xm; // A^{z+}_{xijk} and A^{z-}_{xijk}
    HJ_FLUX(Az_xp,Az_xm,cons(IA3,k,j,i-3),cons(IA3,k,j,i-2),cons(IA3,k,j,i-1),cons(IA3,k,j,i),cons(IA3,k,j,i+1),cons(IA3,k,j,i+2),cons(IA3,k,j,i+3),dx);

    Real axm = pow(epsilon + pow(dx * Ax_xm, 2.0), -2.0);
    Real axp = pow(epsilon + pow(dx * Ax_xp, 2.0), -2.0);
    Real gamma_1 = std::abs((axm / (axm + axp)) - 0.5);

    //--------------------------------------------------------------------------------------
    // y direction   
    
    // Real max_vey = std::max({std::abs(prim(IV2,k,j-3,i)),std::abs(prim(IV2,k,j-2,i)),std::abs(prim(IV2,k,j-1,i)),std::abs(prim(IV2,k,j,i)),std::abs(prim(IV2,k,j+1,i)),std::abs(prim(IV2,k,j+2,i)),std::abs(prim(IV2,k,j+3,i))});
  
    Real Ax_yp, Ax_ym; // A^{x+}_{yijk} and A^{x-}_{yijk}
    HJ_FLUX(Ax_yp, Ax_ym,cons(IA1,k,j-3,i),cons(IA1,k,j-2,i),cons(IA1,k,j-1,i),cons(IA1,k,j,i),cons(IA1,k,j+1,i),cons(IA1,k,j+2,i),cons(IA1,k,j+3,i),dy);

    Real Ay_yp, Ay_ym; // A^{y+}_{yijk} and A^{y-}_{yijk}
    HJ_FLUX(Ay_yp, Ay_ym,cons(IA2,k,j-3,i),cons(IA2,k,j-2,i),cons(IA2,k,j-1,i),cons(IA2,k,j,i),cons(IA2,k,j+1,i),cons(IA2,k,j+2,i),cons(IA2,k,j+3,i),dy);

    Real Az_yp, Az_ym; // A^{z+}_{yijk} and A^{z-}_{yijk}
    HJ_FLUX(Az_yp, Az_ym,cons(IA3,k,j-3,i),cons(IA3,k,j-2,i),cons(IA3,k,j-1,i),cons(IA3,k,j,i),cons(IA3,k,j+1,i),cons(IA3,k,j+2,i),cons(IA3,k,j+3,i),dy);

    Real aym = pow(epsilon + pow(dy * Ay_ym, 2.0), -2.0);
    Real ayp = pow(epsilon + pow(dy * Ay_yp, 2.0), -2.0);
    Real gamma_2 = std::abs((aym / (aym + ayp)) - 0.5);

    //--------------------------------------------------------------------------------------
    // z direction 

    // Real max_vez = std::max({std::abs(prim(IV3,k-3,j,i)),std::abs(prim(IV3,k-2,j,i)),std::abs(prim(IV3,k-1,j,i)),std::abs(prim(IV3,k,j,i)),std::abs(prim(IV3,k+1,j,i)),std::abs(prim(IV3,k+2,j,i)),std::abs(prim(IV3,k+3,j,i))});

    Real Ax_zp, Ax_zm; // A^{x+}_{zijk} and A^{x-}_{zijk}
    HJ_FLUX(Ax_zp, Ax_zm,cons(IA1,k-3,j,i),cons(IA1,k-2,j,i),cons(IA1,k-1,j,i),cons(IA1,k,j,i),cons(IA1,k+1,j,i),cons(IA1,k+2,j,i),cons(IA1,k+3,j,i),dz);

    Real Ay_zp, Ay_zm; // A^{y+}_{zijk} and A^{y-}_{zijk}
    HJ_FLUX(Ay_zp, Ay_zm,cons(IA2,k-3,j,i),cons(IA2,k-2,j,i),cons(IA2,k-1,j,i),cons(IA2,k,j,i),cons(IA2,k+1,j,i),cons(IA2,k+2,j,i),cons(IA2,k+3,j,i),dz);

    Real Az_zp, Az_zm; // A^{z+}_{zijk} and A^{z-}_{zijk}
    HJ_FLUX(Az_zp, Az_zm,cons(IA3,k-3,j,i),cons(IA3,k-2,j,i),cons(IA3,k-1,j,i),cons(IA3,k,j,i),cons(IA3,k+1,j,i),cons(IA3,k+2,j,i),cons(IA3,k+3,j,i),dz);

    Real azm = pow(epsilon + pow(dz * Az_zm, 2.0), -2.0);
    Real azp = pow(epsilon + pow(dz * Az_zp, 2.0), -2.0);
    Real gamma_3 = std::abs((azm / (azm + azp)) - 0.5);

    //--------------------------------------------------------------------------------------
    // Reconstruction

    Real A_x = (0.5 * prim(IV2,k,j,i) * (Ax_ym + Ax_yp)) + (0.5 * prim(IV3,k,j,i) * (Ax_zm + Ax_zp)) - 
               (0.5 * max_vey * (Ax_yp - Ax_ym)) - (0.5 * max_vez * (Ax_zp - Ax_zm)) -
               (0.5 * prim(IV2,k,j,i) * (Ay_xm + Ay_xp)) - (0.5 * prim(IV3,k,j,i) * (Az_xm + Az_xp)) -
              //  (2 * nu)*((cons(IA1,k,j,i-1) - 2*cons(IA1,k,j,i) + cons(IA1,k,j,i+1))/(delta+dt));
               (2 * nu * gamma_1)*((cons(IA1,k,j,i-1) - 2*cons(IA1,k,j,i) + cons(IA1,k,j,i+1))/(delta+dt));

    Real A_y = (0.5 * prim(IV1,k,j,i) * (Ay_xm + Ay_xp)) + (0.5 * prim(IV3,k,j,i) * (Ay_zm + Ay_zp)) - 
               (0.5 * max_vex * (Ay_xp - Ay_xm)) - (0.5 * max_vez * (Ay_zp - Ay_zm)) -
               (0.5 * prim(IV1,k,j,i) * (Ax_ym + Ax_yp)) - (0.5 * prim(IV3,k,j,i) * (Az_ym + Az_yp)) -
              //  (2 * nu)*((cons(IA2,k,j-1,i) - 2*cons(IA2,k,j,i) + cons(IA2,k,j+1,i))/(delta+dt));
               (2 * nu * gamma_2)*((cons(IA2,k,j-1,i) - 2*cons(IA2,k,j,i) + cons(IA2,k,j+1,i))/(delta+dt));
    
    Real A_z = (0.5 * prim(IV1,k,j,i) * (Az_xm + Az_xp)) + (0.5 * prim(IV2,k,j,i) * (Az_ym + Az_yp)) - 
               (0.5 * max_vex * (Az_xp - Az_xm)) - (0.5 * max_vey * (Az_yp - Az_ym)) -
               (0.5 * prim(IV1,k,j,i) * (Ax_zm + Ax_zp)) - (0.5 * prim(IV2,k,j,i) * (Ay_zm + Ay_zp)) -
              //  (2 * nu)*((cons(IA3,k-1,j,i) - 2*cons(IA3,k,j,i) + cons(IA3,k+1,j,i))/(delta+dt));  
               (2 * nu * gamma_3)*((cons(IA3,k-1,j,i) - 2*cons(IA3,k,j,i) + cons(IA3,k+1,j,i))/(delta+dt));  
    
    cons.flux(IV1, IA1, k, j, i) = A_x;
    cons.flux(IV1, IA2, k, j, i) = A_y;
    cons.flux(IV1, IA3, k, j, i) = A_z;

  });

  return TaskStatus::complete;

}


TaskStatus HJAfterstep3D(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");

  auto prim_list = std::vector<std::string>({"prim"});
  auto const &prim_pack = md->PackVariables(prim_list);

  auto cons_new = cons_pack; 


  pmb->par_for(
  "hj afterstep", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
  KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
  
    const auto &coords = cons_pack.GetCoords(b);
    const auto &prim = prim_pack(b);
    auto &cons = cons_pack(b);

    auto dx = cons.GetCoords().Dxc<X1DIR>(k, j, i);
    auto dy = cons.GetCoords().Dxc<X2DIR>(k, j, i);
    auto dz = cons.GetCoords().Dxc<X3DIR>(k, j, i);

    // 4th Order Stencil
    Real B_x = (1 / (12 * dy)) * (cons(IA3,k,j-2,i) - 8.0 * cons(IA3,k,j-1,i) + 8.0 * cons(IA3,k,j+1,i) - cons(IA3,k,j+2,i)) -
               (1 / (12 * dz)) * (cons(IA2,k-2,j,i) - 8.0 * cons(IA2,k-1,j,i) + 8.0 * cons(IA2,k+1,j,i) - cons(IA2,k+2,j,i));

    Real B_y = (1 / (12 * dz)) * (cons(IA1,k-2,j,i) - 8.0 * cons(IA1,k-1,j,i) + 8.0 * cons(IA1,k+1,j,i) - cons(IA1,k+2,j,i)) -
               (1 / (12 * dx)) * (cons(IA3,k,j,i-2) - 8.0 * cons(IA3,k,j,i-1) + 8.0 * cons(IA3,k,j,i+1) - cons(IA3,k,j,i+2));
    
    Real B_z = (1 / (12 * dx)) * (cons(IA2,k,j,i-2) - 8.0 * cons(IA2,k,j,i-1) + 8.0 * cons(IA2,k,j,i+1) - cons(IA2,k,j,i+2)) -
               (1 / (12 * dy)) * (cons(IA1,k,j-2,i) - 8.0 * cons(IA1,k,j-1,i) + 8.0 * cons(IA1,k,j+1,i) - cons(IA1,k,j+2,i));

    // 6th Order Stencil
    // Real B_xx = (1 / (60 * dy)) * (cons(IA3,k,j+3,i) - 9.0 * cons(IA3,k,j+2,i) + 45.0 * cons(IA3,k,j+1,i) - 45.0 * cons(IA3,k,j-1,i) + 9.0 * cons(IA3,k,j-2,i) - cons(IA3,k,j-3,i)) -
    //             (1 / (60 * dz)) * (cons(IA2,k+3,j,i) - 9.0 * cons(IA2,k+2,j,i) + 45.0 * cons(IA2,k+1,j,i) - 45.0 * cons(IA2,k-1,j,i) + 9.0 * cons(IA2,k-2,j,i) - cons(IA2,k-3,j,i));

    // Real B_yy = (1 / (60 * dz)) * (cons(IA1,k+3,j,i) - 9.0 * cons(IA1,k+2,j,i) + 45.0 * cons(IA1,k+1,j,i) - 45.0 * cons(IA1,k-1,j,i) + 9.0 * cons(IA1,k-2,j,i) - cons(IA1,k-3,j,i)) -
    //             (1 / (60 * dx)) * (cons(IA3,k,j,i+3) - 9.0 * cons(IA3,k,j,i+2) + 45.0 * cons(IA3,k,j,i+1) - 45.0 * cons(IA3,k,j,i-1) + 9.0 * cons(IA3,k,j,i-2) - cons(IA3,k,j,i-3));
    
    // Real B_zz = (1 / (60 * dy)) * (cons(IA2,k,j,i+3) - 9.0 * cons(IA2,k,j,i+2) + 45.0 * cons(IA2,k,j,i+1) - 45.0 * cons(IA2,k,j,i-1) + 9.0 * cons(IA2,k,j,i-2) - cons(IA2,k,j,i-3)) -
    //             (1 / (60 * dz)) * (cons(IA1,k,j+3,i) - 9.0 * cons(IA1,k,j+2,i) + 45.0 * cons(IA1,k,j+1,i) - 45.0 * cons(IA1,k,j-1,i) + 9.0 * cons(IA1,k,j-2,i) - cons(IA1,k,j-3,i));

    // 2nd Order Stencil
    // Real B_xxx = (1 / (2 * dy)) * (cons(IA3,k,j+1,i) - cons(IA3,k,j-1,i)) -
    //              (1 / (2 * dz)) * (cons(IA2,k+1,j,i) - cons(IA2,k-1,j,i));

    // Real B_yyy = (1 / (2 * dz)) * (cons(IA1,k+1,j,i) - cons(IA1,k-1,j,i)) -
    //              (1 / (2 * dx)) * (cons(IA3,k,j,i+1) - cons(IA3,k,j,i-1));
    
    // Real B_zzz = (1 / (2 * dx)) * (cons(IA2,k,j,i+1) - cons(IA2,k,j,i-1)) -
    //              (1 / (2 * dy)) * (cons(IA1,k,j+1,i) - cons(IA1,k,j-1,i));

    cons(IB1,k,j,i) = B_x;
    cons(IB2,k,j,i) = B_y;
    cons(IB3,k,j,i) = B_z;

  });

  return TaskStatus::complete;

}



#endif // MP_WENO5_3D_HPP
