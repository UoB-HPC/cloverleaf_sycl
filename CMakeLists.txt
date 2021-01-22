cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(cloverleaf_sycl)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (SYCL_RUNTIME)

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

    if (${SYCL_RUNTIME} STREQUAL "HIPSYCL")

        if (NOT HIPSYCL_INSTALL_DIR)
            message(FATAL_ERROR "HIPSYCL_INSTALL_DIR is undefined")
        endif ()
        set(CMAKE_CXX_STANDARD 17)
        set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake)
        find_package(hipSYCL CONFIG REQUIRED)
        set(EXTRA_FLAGS -Wno-sign-compare -Wno-stringop-truncation)
    elseif (${SYCL_RUNTIME} STREQUAL "HIPSYCL-NEXT")

        if (NOT HIPSYCL_INSTALL_DIR)
            message(FATAL_ERROR "HIPSYCL_INSTALL_DIR is undefined")
        endif ()
        set(CMAKE_CXX_STANDARD 17)
        set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake/hipSYCL)
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        find_package(hipSYCL CONFIG REQUIRED)
        set(EXTRA_FLAGS -Wno-sign-compare -Wno-stringop-truncation)
    elseif (${SYCL_RUNTIME} STREQUAL "COMPUTECPP")

        if (NOT ComputeCpp_DIR)
            message(FATAL_ERROR "ComputeCpp_DIR is undefined")
        endif ()
        set(OpenCL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
        add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        find_package(ComputeCpp REQUIRED)
        set(COMPUTECPP_USER_FLAGS -O3 -no-serial-memop)
    elseif (${SYCL_RUNTIME} STREQUAL "DPCPP")

        if (NOT DPCPP_BIN)
            message(STATUS "DPCPP_BIN unspecified, using dpcpp on path")
            set(CMAKE_CXX_COMPILER "dpcpp")
        else()
            set(CMAKE_CXX_COMPILER ${DPCPP_BIN})
        endif ()

        if(DPCPP_INCLUDE)
            include_directories(${DPCPP_INCLUDE})
        endif()

        add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
        set(CMAKE_CXX_STANDARD 14)
        set(EXTRA_FLAGS -pedantic)
    else ()
        message(FATAL_ERROR "SYCL_RUNTIME unsupported, must be one of HIPSYCL|HIPSYCL-NEXT|COMPUTECPP|DPCPP, got ${SYCL_RUNTIME}")
    endif ()
else ()
    message(FATAL_ERROR "SYCL_RUNTIME not defined, must be one of HIPSYCL|HIPSYCL-NEXT|COMPUTECPP|DPCPP")
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
add_executable(cloverleaf ${SOURCES})

separate_arguments(CXX_EXTRA_FLAGS)
separate_arguments(CXX_EXTRA_LINKER_FLAGS)

target_compile_options(cloverleaf
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


if (DEFINED USE_PRE_SYCL121R3)
    add_definitions(-DUSE_PRE_SYCL121R3)
endif ()


set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer ${CXX_EXTRA_FLAGS})
set(RELEASE_OPTIONS -O3   ${CXX_EXTRA_FLAGS})

target_link_libraries(cloverleaf PUBLIC ${MPI_C_LIB})

target_compile_options(cloverleaf PUBLIC "$<$<CONFIG:RelWithDebInfo>:${RELEASE_OPTIONS}>")
target_compile_options(cloverleaf PUBLIC "$<$<CONFIG:Release>:${RELEASE_OPTIONS}>")
target_compile_options(cloverleaf PUBLIC "$<$<CONFIG:Debug>:${DEBUG_OPTIONS}>")

if (${CMAKE_VERSION} VERSION_LESS "3.13.0")
    message(WARNING "target_link_options is only available in CMake >= 3.13.0, using fallback target_link_libraries, this may cause issues with some compilers")
    message(WARNING "whitespaces are not supported for CXX_EXTRA_LINKER_FLAGS/CXX_EXTRA_FLAGS in this mode as they are treated as libraries arguments (CMake splits them)")
    if (DEFINED CXX_EXTRA_LINKER_FLAGS)
        list(APPEND EXTRA_LINK_FLAGS "-Wl,${CXX_EXTRA_LINKER_FLAGS}")
    endif ()

    target_link_libraries(cloverleaf PUBLIC ${EXTRA_LINK_FLAGS})
    target_link_libraries(cloverleaf PUBLIC ${CXX_EXTRA_FLAGS})

else ()
    target_link_options(cloverleaf PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})
    target_link_options(cloverleaf PUBLIC ${CXX_EXTRA_FLAGS})
endif ()


if (NOT ${SYCL_RUNTIME} STREQUAL "DPCPP")
    add_sycl_to_target(
            TARGET cloverleaf
            SOURCES ${SOURCES}) # must be the last
endif ()

