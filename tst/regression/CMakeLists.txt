# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

# import Parthenon setup_test_serial and setup_test_parallel
include(${PROJECT_SOURCE_DIR}/external/parthenon/cmake/TestSetup.cmake)

setup_test_serial("convergence" "--driver ${PROJECT_BINARY_DIR}/bin/parthenon-hydro \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 3" "convergence")

  # ${NUM_MPI_PROC_TESTING} is reused from Parthenon
setup_test_parallel(${NUM_MPI_PROC_TESTING} "convergence" "--driver ${PROJECT_BINARY_DIR}/bin/parthenon-hydro \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 3" "convergence")
