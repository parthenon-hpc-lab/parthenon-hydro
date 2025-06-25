# Copyright (c) 2020-2021, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    parthenon-hydro
        main.cpp
        eos/adiabatic_hydro.cpp
        eos/adiabatic_mhd.cpp
        units.hpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        refinement/gradient.cpp
        refinement/other.cpp
        utils/few_modes_ft.cpp
)

add_subdirectory(pgen)

target_link_libraries(parthenon-hydro PRIVATE parthenon)
