
# [WIP] A SYCL port of CloverLeaf

This is a port of [CloverLeaf](https://github.com/uk-mac/cloverleaf_ref) from MPI+OpenMP Fortran to MPI+SYCL C++.


## Development

Prerequisites:

 * CentOS 7
 * cmake3
 * openmpi, opemmpi-devel
 * [devtoolset-7](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/)
 * entr
 
You may also need to install `epel-release` for some of the packages.

For a quick build, use:

	source /opt/rh/devtoolset-7/enable
	module load mpi

    ./dev.sh <ComputeCpp_DIR>
    e.g
    ./dev.sh /home/tom/ComputeCpp-CE-1.1.3-CentOS-x86_64

