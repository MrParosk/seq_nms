cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(seq_nms)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(pt_seq_nms/csrc)
add_subdirectory(tests/cpp)
