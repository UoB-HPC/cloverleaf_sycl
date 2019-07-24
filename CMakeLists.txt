cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cloverleaf_sycl)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(ComputeCpp REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 11)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

#include_directories(src)

SET(SOURCES
        src/cxx14_compat.hpp

        src/sycl_reduction.hpp
        src/sycl_utils.hpp

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

#                src/_main.cpp
        #        src/_main_nosycl.cpp
        )


add_executable(clover_leaf ${SOURCES})


target_compile_options(clover_leaf
        PUBLIC
        -Wall
        -Wextra
        -Wcast-align
        -Wpedantic
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-ignored-attributes
        )

# -fsanitize=address
#set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer -fsanitize=address -march=native -g)
set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer -march=native -g)
set(RELEASE_OPTIONS -O3 -march=native -g)

#target_link_libraries(clover_leaf PUBLIC $<$<CONFIG:DEBUG>:-Wl,-lasan>) # must come before any other libraries
target_link_libraries(clover_leaf PRIVATE ${MPI_CXX_LIBRARIES})
target_compile_options(clover_leaf PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

#set_property(SOURCE ${SOURCES} PROPERTY COMPUTECPP_SOURCE_FLAGS -Rno-sycl-kernel-naming)

add_sycl_to_target(
        TARGET clover_leaf
        SOURCES ${SOURCES}
        ${CMAKE_CURRENT_BINARY_DIR}) # must be the last