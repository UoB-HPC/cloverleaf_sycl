cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cloverleaf_sycl)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(ComputeCpp REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 14)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif ()


include_directories(SYSTEM ${MPI_INCLUDE_PATH})

#include_directories(src)

SET(SOURCES
        src/accelerate.cpp
        src/advec_cell.cpp
        src/advec_mom.cpp
        src/advection.cpp
        src/build_field.cpp
        src/calc_dt.cpp
        src/clover_leaf.cpp
        src/comms.cpp
        src/field_summary.cpp
        src/flux_calc.cpp
        src/generate_chunk.cpp
        src/hydro.cpp
        src/ideal_gas.cpp
        src/initialise_chunk.cpp
        src/initialise.cpp
        src/pack_kernel.cpp
        src/PdV.cpp
        src/read_input.cpp
        src/report.cpp
        src/reset_field.cpp
        src/revert.cpp
        src/start.cpp
        src/timer.cpp
        src/timestep.cpp
        src/update_halo.cpp
        src/update_tile_halo.cpp
        src/update_tile_halo_kernel.cpp
        src/viscosity.cpp
        src/visit.cpp

#        src/_main.cpp
        )

add_executable(clover_leaf ${SOURCES})
target_link_libraries(clover_leaf ${MPI_CXX_LIBRARIES})


add_sycl_to_target(
        TARGET clover_leaf
        SOURCES src/_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR})
