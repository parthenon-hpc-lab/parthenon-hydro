# Copyright (c) 2020-2021, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    parthenon-hydro
        main.cpp
        eos/adiabatic_hydro.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        recon/plm_simple.hpp
        refinement/gradient.cpp
        refinement/other.cpp
)

add_subdirectory(pgen)

target_link_libraries(parthenon-hydro PRIVATE parthenon)
