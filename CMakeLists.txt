cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cloverleaf_sycl)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (SYCL_RUNTIME)

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

    if (${SYCL_RUNTIME} STREQUAL "HIPSYCL")

        if (NOT HIPSYCL_INSTALL_DIR)
            message(FATAL_ERROR "HIPSYCL_INSTALL_DIR is undefined")
        endif ()

        set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake)
        find_package(hipSYCL CONFIG REQUIRED)
        set(EXTRA_FLAGS -Wno-sign-compare -Wno-stringop-truncation)
    elseif (${SYCL_RUNTIME} STREQUAL "COMPUTECPP")

        if (NOT ComputeCpp_DIR)
            message(FATAL_ERROR "ComputeCpp_DIR is undefined")
        endif ()
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        set(COMPUTECPP_USER_FLAGS -O3 -fsycl-split-modules=20 -mllvm -inline-threshold=10000 -no-serial-memop)
        find_package(ComputeCpp REQUIRED)
        #        set(EXTRA_FLAGS  -pedantic)
    elseif (${SYCL_RUNTIME} STREQUAL "DPCPP")

        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_COMPILER "dpcpp")
        set(EXTRA_FLAGS -pedantic)
    else ()
        message(FATAL_ERROR "SYCL_RUNTIME unsupported, must be one of HIPSYCL|COMPUTECPP|DPCPP, got ${SYCL_RUNTIME}")
    endif ()
else ()
    message(FATAL_ERROR "SYCL_RUNTIME not defined, must be one of HIPSYCL|COMPUTECPP|DPCPP")
endif ()


if (MPI_AS_LIBRARY)

    if (NOT DEFINED MPI_C_LIB_DIR)
        message(FATAL_ERROR "MPI_C_LIB_DIR must be specified, typically <mpi_root_dir>/lib")
    endif ()

    if (NOT DEFINED MPI_C_INCLUDE_DIR)
        message(FATAL_ERROR "MPI_C_INCLUDE_DIR must be specified, typically <mpi_root_dir>/include")
    endif ()

    if (NOT DEFINED MPI_C_LIB)
        message(FATAL_ERROR "MPI_C_LIB must be specified, for example: mpich for libmpich.so in MPI_C_LIB_DIR")
    endif ()

    message(STATUS "Using MPI as a library (${MPI_C_LIB})")
    message(STATUS "MPI include dir: ${MPI_C_INCLUDE_DIR}")
    message(STATUS "MPI library dir: ${MPI_C_LIB_DIR}")

    include_directories(${MPI_C_INCLUDE_DIR})
    link_directories(${MPI_C_LIB_DIR})
else ()
    find_package(MPI REQUIRED)
    set(MPI_C_LIB MPI::MPI_C)
endif ()


set(SOURCES
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
        src/visit.cpp)

include_directories(src)
add_executable(clover_leaf ${SOURCES})

target_compile_options(clover_leaf
        PUBLIC
        -Wall
        -Wextra
        -Wcast-align
        -Wfatal-errors
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-ignored-attributes

        ${EXTRA_FLAGS}
        )

set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer ${CXX_EXTRA_FLAGS})
set(RELEASE_OPTIONS -O3 ${CXX_EXTRA_FLAGS})

target_link_libraries(clover_leaf PUBLIC ${MPI_C_LIB})

target_compile_options(clover_leaf PUBLIC "$<$<CONFIG:RelWithDebInfo>:${RELEASE_OPTIONS}>")
target_compile_options(clover_leaf PUBLIC "$<$<CONFIG:Release>:${RELEASE_OPTIONS}>")
target_compile_options(clover_leaf PUBLIC "$<$<CONFIG:Debug>:${DEBUG_OPTIONS}>")

target_link_options(clover_leaf PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})

if (NOT ${SYCL_RUNTIME} STREQUAL "DPCPP")
    add_sycl_to_target(
            TARGET clover_leaf
            SOURCES ${SOURCES}) # must be the last
endif ()

