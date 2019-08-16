#!/usr/bin/env sh

set -e

# mkdir -p build
cmake3 --build build --target clover_leaf --config Release  -j 12
echo "CMake build done"
./build/clover_leaf InputDecks/clover_bm16_short.in