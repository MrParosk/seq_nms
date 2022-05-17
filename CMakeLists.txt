cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(seq_nms)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(GTest REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(run_tests tests/tests.cpp)
target_sources(run_tests PRIVATE "${CURRENT_DIR}/src/seq_nms.cpp" )
target_sources(run_tests PRIVATE "${CURRENT_DIR}/src/box_utils.cpp" )
target_sources(run_tests PRIVATE "${CURRENT_DIR}/src/sequence_utils.cpp")

target_link_libraries(run_tests  ${GTEST_LIBRARIES} ${TORCH_LIBRARIES})
