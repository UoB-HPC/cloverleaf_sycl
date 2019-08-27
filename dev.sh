#!/usr/bin/env sh

set -e

rm -rf ./build


echo using ComputeCpp_DIR=$1
cmake3 -Bbuild -H. -DComputeCpp_DIR=$1 -DCMAKE_BUILD_TYPE=Release


# -o -name 'CMakeList.txt'
find . -name '*.cpp' -o -name '*.h' -o -name '*.hpp'  | entr -r -c ./run.sh
