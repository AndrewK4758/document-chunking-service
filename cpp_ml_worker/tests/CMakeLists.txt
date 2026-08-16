cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(cpp_ml_worker_test test_main.cpp test_span_generator.cpp)

target_include_directories(cpp_ml_worker_test PRIVATE
        ../src
)

target_link_libraries(cpp_ml_worker_test PRIVATE
        worker_lib
        GTest::gtest_main
)

gtest_discover_tests(cpp_ml_worker_test)
