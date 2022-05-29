#!/bin/bash

set -e

current_dir=$(pwd) 
mkdir -p build
(cd build && cmake -DCMAKE_PREFIX_PATH="$current_dir/libtorch" -DCMAKE_BUILD_TYPE=Debug .. && make -j)

./build/tests/cpp/run_tests
