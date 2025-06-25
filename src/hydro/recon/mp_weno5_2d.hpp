#ifndef MP_WENO5_2D_HPP
#define MP_WENO5_2D_HPP


#include <iostream> 
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>   // For std::setprecision

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "weno_helpers.hpp"

TaskStatus CalculateHJFluxes2D(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");

  auto prim_list = std::vector<std::string>({"prim"});
  auto const &prim_pack = md->PackVariables(prim_list);

  pmb->par_for(
  "hj flux", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
  KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
  
    const auto &coords = cons_pack.GetCoords(b);
    const auto &prim = prim_pack(b);
    auto &cons = cons_pack(b);

    auto dx = cons.GetCoords().Dxc<X1DIR>(k, j, i);
    auto dy = cons.GetCoords().Dxc<X2DIR>(k, j, i);
    auto dz = cons.GetCoords().Dxc<X3DIR>(k, j, i);

    //--------------------------------------------------------------------------------------
    // A_z

    Real Az_xp, Az_xm; // A^{z+}_{xijk} and A^{z-}_{xijk}
    HJ_FLUX(Az_xp,Az_xm,cons(IA3,k,j,i-3),cons(IA3,k,j,i-2),cons(IA3,k,j,i-1),cons(IA3,k,j,i),cons(IA3,k,j,i+1),cons(IA3,k,j,i+2),cons(IA3,k,j,i+3),dx);

    Real Az_yp, Az_ym; // A^{z+}_{yijk} and A^{z-}_{yijk}
    HJ_FLUX(Az_yp, Az_ym,cons(IA3,k,j-3,i),cons(IA3,k,j-2,i),cons(IA3,k,j-1,i),cons(IA3,k,j,i),cons(IA3,k,j+1,i),cons(IA3,k,j+2,i),cons(IA3,k,j+3,i),dy);

    Real max_vex = std::max({std::abs(prim(IV1,k,j,i-3)),std::abs(prim(IV1,k,j,i-2)),std::abs(prim(IV1,k,j,i-1)),std::abs(prim(IV1,k,j,i)),std::abs(prim(IV1,k,j,i+1)),std::abs(prim(IV1,k,j,i+2)),std::abs(prim(IV1,k,j,i+3))});
    Real max_vey = std::max({std::abs(prim(IV2,k,j-3,i)),std::abs(prim(IV2,k,j-2,i)),std::abs(prim(IV2,k,j-1,i)),std::abs(prim(IV2,k,j,i)),std::abs(prim(IV2,k,j+1,i)),std::abs(prim(IV2,k,j+2,i)),std::abs(prim(IV2,k,j+3,i))});

    Real A_z = (0.5 * prim(IV1,k,j,i) * (Az_xp + Az_xm)) + (0.5 * prim(IV2,k,j,i) * (Az_yp + Az_ym)) - (0.5 * max_vex * (Az_xp - Az_xm)) - (0.5 * max_vey * (Az_yp - Az_ym));

    cons.flux(IV1, IA3, k, j, i) = A_z;

  });

  return TaskStatus::complete;

}

TaskStatus HJAfterstep2D(std::shared_ptr<MeshData<Real>> &md) {
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

    //--------------------------------------------------------------------------------------
    // B_x

    Real B_x = (1 / (12 * dy)) * (cons(IA3,k,j-2,i) - 8.0 * cons(IA3,k,j-1,i) + 8.0 * cons(IA3,k,j+1,i) - cons(IA3,k,j+2,i));

    //--------------------------------------------------------------------------------------
    // B_y 

    Real B_y = (1 / (12 * dx)) * (cons(IA3,k,j,i+2) - 8.0 * cons(IA3,k,j,i+1) + 8.0 * cons(IA3,k,j,i-1) - cons(IA3,k,j,i-2));
    
    cons(IB1,k,j,i) = B_x;
    cons(IB2,k,j,i) = B_y;

  });

  return TaskStatus::complete;

}



#endif // MP_WENO5_2D_HPP
