//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2022, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon header
#include <mesh/refinement_cc_in_one.hpp>
#include <parthenon/parthenon.hpp>
#include <refinement/refinement.hpp>

// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"

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
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    const auto any = parthenon::BoundaryType::any;
    tl.AddTask(none, parthenon::cell_centered_bvars::StartReceiveBoundBufs<any>, mu0);
    tl.AddTask(none, parthenon::cell_centered_bvars::StartReceiveFluxCorrections, mu0);

    // Calculate fluxes (will be stored in the x1, x2, x3 flux arrays of each var)
    auto calc_flux = tl.AddTask(none, CalculateFluxes, mu0);

    // Correct for fluxes across levels (to maintain conservative nature of update)
    auto send_flx = tl.AddTask(
        calc_flux, parthenon::cell_centered_bvars::LoadAndSendFluxCorrections, mu0);
    auto recv_flx = tl.AddTask(
        calc_flux, parthenon::cell_centered_bvars::ReceiveFluxCorrections, mu0);
    auto set_flx =
        tl.AddTask(recv_flx, parthenon::cell_centered_bvars::SetFluxCorrections, mu0);

    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);
    // Compute the divergence of fluxes of conserved variables
    auto update = tl.AddTask(
        set_flx, parthenon::Update::UpdateWithFluxDivergence<MeshData<Real>>, mu0.get(),
        mu1.get(), integrator->gam0[stage - 1], integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);

    // Note the difference between local and non-local buffers.
    // The best performing (at scale) combination/order is still tbd.
    // update ghost cells (non local)
    const auto nonlocal = parthenon::BoundaryType::nonlocal;
    auto send_nonlocal =
        tl.AddTask(update, parthenon::cell_centered_bvars::SendBoundBufs<nonlocal>, mu0);
  }

  TaskRegion &sendrecv_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = sendrecv_region[i];

    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);

    // update ghost cells (local)
    const auto local = parthenon::BoundaryType::local;
    auto send_local =
        tl.AddTask(none, parthenon::cell_centered_bvars::SendBoundBufs<local>, mu0);
    auto recv_local =
        tl.AddTask(none, parthenon::cell_centered_bvars::ReceiveBoundBufs<local>, mu0);
    auto set_local =
        tl.AddTask(recv_local, parthenon::cell_centered_bvars::SetBounds<local>, mu0);

    // update ghost cells (non-local buffers) hoping that messages arrived while the local
    // buffer were handled
    const auto nonlocal = parthenon::BoundaryType::nonlocal;
    auto recv_nonlocal =
        tl.AddTask(none, parthenon::cell_centered_bvars::ReceiveBoundBufs<nonlocal>, mu0);
    auto set_nonlocal = tl.AddTask(
        recv_nonlocal, parthenon::cell_centered_bvars::SetBounds<nonlocal>, mu0);

    if (pmesh->multilevel) {
      tl.AddTask(set_nonlocal | set_local,
                 parthenon::cell_centered_refinement::RestrictPhysicalBounds, mu0.get());
    }
  }

  TaskRegion &async_region_3 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_3[i];
    auto &u0 = blocks[i]->meshblock_data.Get("base");
    auto prolongBound = none;
    if (pmesh->multilevel) {
      prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, u0);
    }

    // set physical boundaries
    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, u0);
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, parthenon::Update::EstimateTimestep<MeshData<Real>>, mu0.get());
    }
  }

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
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
