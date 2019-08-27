# A SYCL port of CloverLeaf

This is a port of [CloverLeaf](https://github.com/UoB-HPC/cloverleaf_kokkos) from MPI+Kokkos to MPI+SYCL.

## Tested configurations

The program was compiled and tested on the following configurations.

CC -  GCC 9.1.0  
SYCL - ComputeCpp 1.1.3
 

 * Intel CPU - [Intel OpenCL runtime 18.1](https://software.intel.com/en-us/articles/opencl-drivers)
   * 1 x i7-8850H
   * 1 x i7-6700K
   * 1 x i7-6770HQ
   * 1 x Xeon Gold 6126
   * 1 x Xeon E5-2630L v0
 * Intel GPU - [Intel compute-runtime 19.32.13826](https://github.com/intel/compute-runtime/releases/tag/19.32.13826)
   * 1 x UHD Graphics 630
   * 1 x Iris Pro Graphics 580
   

## Building

Prerequisites:

 * CentOS 7
 * cmake3
 * openmpi, opemmpi-devel
 * [ComputeCpp community edition](https://www.codeplay.com/products/computesuite/computecpp)
 * [devtoolset-7](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/)
 
First, generate a build:
 
    cmake3 -Bbuild -H. -DCMAKE_BUILD_TYPE=Release  -DComputeCpp_DIR=<path_to_computecpp> -DOpenCL_INCLUDE_DIR=include/

If parts of your toolchian are installed at different places, you'll have to specify it manually, for example:

    cmake3 -Bbuild -H.  \
    -DOpenCL_INCLUDE_DIR=include/ \
    -DOpenCL_LIBRARY=/nfs/software/x86_64/intel/opencl/18.1/opt/intel/opencl_compilers_and_libraries_18.1.0.015/linux/compiler/lib/intel64_lin/libOpenCL.so.2.0 \
    -DComputeCpp_DIR=/nfs/software/x86_64/computecpp/1.1.3 \
    -DCMAKE_C_COMPILER=/nfs/software/x86_64/gcc/9.1.0/bin/gcc \
    -DCMAKE_CXX_COMPILER=/nfs/software/x86_64/gcc/9.1.0/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release \

For experimental NVidia ptx support, add:

    -DCOMPUTECPP_BITCODE=ptx64

Proceed with compiling, adjust `<thread_count>` accordingly:
    
    cmake3 --build build --target clover_leaf --config Release -j <thread_count>
   

## Running

The main `clover_leaf` executable takes a `clover.in` file as parameter and outputs `clover.out` at working directory.

For example, after successful compilation, at **project root**:

    ./build/clover_leaf InputDecks/clover_bm16_short.in

See [Tested configurations](#tested-configurations) for tested platforms and drivers. Also see ComputeCpp's [platform support page](https://developer.codeplay.com/products/computecpp/ce/guides/platform-support) for supported configurations.

To run on a specific device, unload all other drivers or modify the device selector in `start.cpp:107` and recompile.

## Development

Prerequisites:

 * entr
 
You may also need to install `epel-release` for some of the packages.

For a quick build, use:

	source /opt/rh/devtoolset-7/enable
	module load mpi

    ./dev.sh <ComputeCpp_DIR>
    e.g
    ./dev.sh /home/tom/ComputeCpp-CE-1.1.3-CentOS-x86_64

## Known issues

 * Due to ComputeCpp's limitation where built-ins are missing when targeting ptx, NVidia based GPUs are not supported yet.

 * Selecting non-default devices requires recompiling.
