#!/usr/bin/env sh

set -e

# mkdir -p build
cmake3 --build build --target clover_leaf --config Debug  -j 8
echo "CMake build done"
./build/clover_leaf