#!/bin/bash

# ATM cmake is only used for running cpp-tests

current_dir=$(pwd) 
mkdir -p build
(cd build && cmake -DCMAKE_PREFIX_PATH="$current_dir/libtorch" -DCURRENT_DIR=$current_dir -DCMAKE_BUILD_TYPE=Debug .. && make -j)

./build/run_tests
