find_package(GTest REQUIRED)

add_executable(cpp_ml_worker_test test_main.cpp)

target_include_directories(cpp_ml_worker_test PRIVATE 
  ../src
)

target_link_libraries(cpp_ml_worker_test PRIVATE
  worker_lib
  GTest::gtest_main
)

add_test(NAME cpp_ml_worker_test COMMAND cpp_ml_worker_test)
