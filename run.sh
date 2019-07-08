#!/usr/bin/env sh

set -e

# mkdir -p build
cmake3 --build build --target clover_leaf -j 6
echo "CMake build done"
./build/clover_leaf