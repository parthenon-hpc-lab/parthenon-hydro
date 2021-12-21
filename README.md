# Parthenon-Hydro

`Parthenon-Hydro` is a finite volume, compressible hydrodynamics sample implementation using the
performance portable adaptive mesh framework [Parthenon](https://github.com/lanl/parthenon) and [Kokkos](https://github.com/kokkos/kokkos).
It is effectively a simplified version of [AthenaPK](https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenapk)
demonstrating the use of the Parthenon interfaces (including the testing framework) with a use-case that is more complex
than the examples included in Parthenon itself.

## Features/algorithms

- Compressible hydrodynamics
  - All integrators currently supported by Parthenon: RK1, RK2, RK3, and VL2
  - Piecewise linear (PLM) reconstruction
  - HLLE Riemann solver
  - Adiabatic equation of state
- Static and adaptive mesh refinement
- Problem generators for
  - a linear wave
  - blast wave
  - Kelvin-Helmholtz instability

## Getting in touch

If you
* encounter a bug or problem,
* have a feature request,
* would like to contribute, or
* have a general question or comment

please either
- open an issue/merge request, or
- contact us in the Parthenon channel on matrix.org [#parthenon-general:matrix.org](https://app.element.io/#/room/#parthenon-general:matrix.org)

## Getting started

### Installation

#### Dependencies

##### Required

* CMake 3.13 or greater
* C++17 compatible compiler
* Parthenon (using the submodule version provided by AthenaPK)
* Kokkos (using the submodule version provided by AthenaPK)

##### Optional

* MPI
* OpenMP (for host parallelism. Note that MPI is the recommended option for on-node parallelism.)
* HDF5 (for outputs)

#### Building Parthenon-Hydro

`Parthenon-Hydro` is also used for integration testing and therefore closely tracks the `develop` branch of Parthenon.
For this reason, it is highly recommended to only use `Parthenon-Hydro` with the Kokkos and Parthenon versions that are
provided by the submodules and to build everything together from source.
Neither other versions or nor using preinstalled Parthenon/Kokkos libraries have been tested.

Obtain all (Parthenon-Hydro, Parthenon, and Kokkos) sources

    git clone https://github.com/pgrete/parthenon-hydro
    cd parthenon-hydro

    # get submodules (mainly Kokkos and Parthenon)
    git submodule init
    git submodule update

Most of the general build instructions and options for Parthenon (see [here](https://github.com/lanl/parthenon/blob/develop/docs/building.md)) also apply to `Parthenon-Hydro`.
The following examples are a few standard cases.

Most simple configuration (only CPU, no MPI, no HDF5)

    # enabling Broadwell architecture (AVX2) instructions
    cmake -S. -Bbuild-host -DKokkos_ARCH_BDW=ON -DPARTHENON_DISABLE_MPI=ON -DPARTHENON_DISABLE_HDF5=ON ../
    cd build-host && make

An Intel Skylake system (AVX512 instructions) with NVidia Volta V100 GPUs and with MPI and HDF5 enabled (the latter is the default option, so they don't need to be specified)

    cmake -S. -Bbuild-gpu -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ../
    cd build-gpu && make

#### Run Parthenon-Hydro

Some example input files are provided in the [inputs](inputs/) folder.

    # for a simple linear wave test run
    ./bin/parthenon-hydro -i ../inputs/linear_wave3d.in

    # to run a convergence test:
    for M in 16 32 64 128; do
      export N=$M;
      ./bin/parthenon-hydro -i ../inputs/linear_wave3d.in parthenon/meshblock/nx1=$((2*N)) parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N parthenon/mesh/nx1=$((2*M)) parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M
    done

    # and check the resulting errors
    cat linearwave-errors.dat

