//========================================================================================
// parthenon-hydro - a performance portable block-structured AMR compr. hydro miniapp
// Copyright (c) 2020-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon header
#include <amr_criteria/refinement_package.hpp>
#include <parthenon/parthenon.hpp>
#include <prolong_restrict/prolong_restrict.hpp>

// parthenon-hydro headers
#include "../eos/adiabatic_hydro.hpp"
#include "../eos/adiabatic_mhd.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"
#include "recon/mp_weno5_2d.hpp"
#include "recon/mp_weno5_3d.hpp"

using namespace parthenon::driver::prelude;

namespace Hydro {

HydroDriver::HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("hydro", "gamma");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/time", "cfl");
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection HydroDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  const auto &stage_name = integrator->stage_name;
  auto hydro_pkg = blocks[0]->packages.Get("Hydro");

  TaskID none(0);
  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();

  TaskRegion &async_region_1 = tc.AddRegion(num_task_lists_executed_independently);
  // std::cout << "async_region_1" << std::endl;
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_1[i];
    // Using "base" as u0, which already exists (and returned by using plain Get())
    auto &u0 = pmb->meshblock_data.Get();

    // Create meshblock data for register u1. This is a no-op if u1 already exists.
    if (stage == 1) {
      pmb->meshblock_data.Add("u1", u0);

      // init u1, see (11) in Athena++ method paper
      auto &u1 = pmb->meshblock_data.Get("u1");
      auto init_u1 = tl.AddTask(
          none,
          [](MeshBlockData<Real> *u0, MeshBlockData<Real> *u1) {
            u1->Get("cons").data.DeepCopy(u0->Get("cons").data);
            return TaskStatus::complete;
          },
          u0.get(), u1.get());
    }
  }
  const int num_partitions = pmesh->DefaultNumPartitions();

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  // std::cout << "single_tasklist_per_pack_region" << std::endl;
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    const auto any = parthenon::BoundaryType::any;
  
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mu0);
    auto start_flxcor_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mu0);

    // Calculate fluxes (will be stored in the x1, x2, x3 flux arrays of each var)
    const auto flux_str = (stage == 1) ? "flux_first_stage" : "flux_other_stage";
    FluxFun_t *calc_flux_fun = hydro_pkg->Param<FluxFun_t *>(flux_str);
    auto calc_flux = tl.AddTask(none, calc_flux_fun, mu0);

    // Unstaggered Constrained Transport Step
    Fluid fluid = hydro_pkg->Param<Fluid>("fluid");
    auto pmb = mu0->GetBlockData(0)->GetBlockPointer();
    auto calc_mp_flux = calc_flux;
    if (fluid == Fluid::mhd) {
      if (pmb->pmy_mesh->ndim == 2) {
        auto calc_mp_flux = tl.AddTask(calc_flux, CalculateHJFluxes2D, mu0); // 2D Magnetic Potential
      } else if (pmb->pmy_mesh->ndim == 3) {
        auto calc_mp_flux = tl.AddTask(calc_flux, CalculateHJFluxes3D, mu0, integrator->dt); // 3D Magnetic Potential
      }
    }

    // Correct for fluxes across levels (to maintain conservative nature of update)
    auto send_flx = tl.AddTask(calc_mp_flux, parthenon::LoadAndSendFluxCorrections, mu0);
    auto recv_flx = tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, mu0);
    auto set_flx  = tl.AddTask(recv_flx | calc_mp_flux, parthenon::SetFluxCorrections, mu0);

    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);
    // Compute the divergence of fluxes of conserved variables
    auto update = tl.AddTask(
        set_flx, parthenon::Update::UpdateWithFluxDivergenceCA<MeshData<Real>>, mu0.get(),
        mu1.get(), integrator->gam0[stage - 1], integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);

    // Turbulence Driver Force
    auto source_split_first_order = update;
    if (stage == integrator->nstages) {
      source_split_first_order =
          tl.AddTask(update, AddSplitSourcesFirstOrder, mu0.get(), tm);
    } 

    // Update ghost cells (local and non local)
    // Note that Parthenon also support to add those tasks manually for more fine-grained
    // control.
    auto bcstep1 = parthenon::AddBoundaryExchangeTasks(source_split_first_order | start_bnd, tl, mu0, pmesh->multilevel); 


    // Unstaggered Constrained Transport Afterstep
    if (fluid == Fluid::mhd) {
      if (pmb->pmy_mesh->ndim == 2) {
        
        auto afterstep_flux = tl.AddTask(bcstep1, HJAfterstep2D, mu0);; // 2D Magnetic Potential
        parthenon::AddBoundaryExchangeTasks(afterstep_flux | start_bnd, tl, mu0, pmesh->multilevel); 

      } else if (pmb->pmy_mesh->ndim == 3) {
        
        auto afterstep_flux = tl.AddTask(bcstep1, HJAfterstep3D, mu0); // 3D Magnetic Potential
        parthenon::AddBoundaryExchangeTasks(afterstep_flux | start_bnd, tl, mu0, pmesh->multilevel); 

      }
    } 


  } // single_tasklist_per_pack_region
  
  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  // std::cout << "single_tasklist_per_pack_region_3" << std::endl;
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
  }
  
  if (stage == integrator->nstages) {
    TaskRegion &tr = tc.AddRegion(num_partitions);
    // std::cout << "tr" << std::endl;
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_dt = tl.AddTask(none, parthenon::Update::EstimateTimestep<MeshData<Real>>,
                               mu0.get());
    }
  }

  

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
    // std::cout << "async_region_4" << std::endl;
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &u0 = blocks[i]->meshblock_data.Get("base");
      auto tag_refine =
          tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<Real>>, u0.get());
    }
  }

  return tc;
}
} // namespace Hydro
