cmake_minimum_required(VERSION 3.15)
project(cpp_ml_worker)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Probe <mdspan> per MPL-312 §5.8 — no polyfill
include(CheckIncludeFileCXX)
check_include_file_cxx("mdspan" HAVE_MDSPAN)
if(NOT HAVE_MDSPAN)
  message(FATAL_ERROR "C++26 <mdspan> not found — requires GCC 14+ or Clang 17+ with -std=c++26")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

find_package(gRPC REQUIRED)
find_package(Protobuf REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

include(FetchContent)

# Find onnxruntime natively, fallback to FetchContent
#find_package(onnxruntime QUIET)
if (TRUE)
  FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
  )
  FetchContent_MakeAvailable(onnxruntime)

  set(ORT_GPU_LIB "/home/ak/.asdf/installs/python/3.14.7/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0")
  set(ORT_GPU_DIR "/home/ak/.asdf/installs/python/3.14.7/lib/python3.14/site-packages/onnxruntime/capi")

  add_library(onnxruntime_imported SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime_imported PROPERTIES
    IMPORTED_LOCATION "${ORT_GPU_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
    NO_SYSTEM_FROM_IMPORTED TRUE
  )
  set(CMAKE_BUILD_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64")
  set(CMAKE_INSTALL_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

# Fetch tokenizers-cpp
FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG main
)

FetchContent_MakeAvailable(tokenizers_cpp)

# Fetch or find simdutf per MPL-316
find_package(simdutf CONFIG QUIET)
if(NOT TARGET simdutf::simdutf)
  FetchContent_Declare(
    simdutf
    GIT_REPOSITORY https://github.com/simdutf/simdutf.git
    GIT_TAG v6.0.3
  )
  FetchContent_MakeAvailable(simdutf)
endif()

if(NOT TARGET simdutf::simdutf)
  message(FATAL_ERROR "simdutf required for UTF-8 FFI boundary; install libsimdutf-dev or allow FetchContent")
endif()

set(PROTO_SRC_DIR ${CMAKE_SOURCE_DIR}/../../shared_protos)
set(PROTO_BUILD_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_BUILD_DIR})

set(PROTOS
    ${PROTO_SRC_DIR}/models.proto
    ${PROTO_SRC_DIR}/storage_gateway.proto
    ${PROTO_SRC_DIR}/ml_worker.proto
)

set(PROTO_SRCS)
set(PROTO_HDRS)
set(GRPC_SRCS)
set(GRPC_HDRS)

foreach(proto ${PROTOS})
    get_filename_component(PROTO_NAME ${proto} NAME_WE)
    list(APPEND PROTO_SRCS ${PROTO_BUILD_DIR}/${PROTO_NAME}.pb.cc)
    list(APPEND PROTO_HDRS ${PROTO_BUILD_DIR}/${PROTO_NAME}.pb.h)
    list(APPEND GRPC_SRCS ${PROTO_BUILD_DIR}/${PROTO_NAME}.grpc.pb.cc)
    list(APPEND GRPC_HDRS ${PROTO_BUILD_DIR}/${PROTO_NAME}.grpc.pb.h)

    add_custom_command(
        OUTPUT ${PROTO_BUILD_DIR}/${PROTO_NAME}.pb.cc ${PROTO_BUILD_DIR}/${PROTO_NAME}.pb.h ${PROTO_BUILD_DIR}/${PROTO_NAME}.grpc.pb.cc ${PROTO_BUILD_DIR}/${PROTO_NAME}.grpc.pb.h
        COMMAND protobuf::protoc
        ARGS --grpc_out=${PROTO_BUILD_DIR} --cpp_out=${PROTO_BUILD_DIR} -I ${PROTO_SRC_DIR} --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin> ${proto}
        DEPENDS ${proto}
    )
endforeach()

add_library(worker_lib
    src/StorageGatewayClient.cpp
    src/MLInferenceEngine.cpp
    src/MuPDFWrapper.cpp
    src/FzContextPool.cpp
    src/SpanGenerator.cpp
    src/SpatialLayoutAnalyzer.cpp
    ${PROTO_SRCS} ${GRPC_SRCS}
)

target_include_directories(worker_lib BEFORE PUBLIC
  ${onnxruntime_SOURCE_DIR}/include
  ${tokenizers_cpp_SOURCE_DIR}/include
  src
  ${PROTO_BUILD_DIR}
)

target_link_libraries(worker_lib PUBLIC
  gRPC::grpc++
  protobuf::libprotobuf
  onnxruntime_imported
  tokenizers_cpp
  simdutf::simdutf
  mupdf
  spdlog::spdlog
  fmt::fmt
)

target_compile_features(worker_lib PUBLIC cxx_std_26)

add_executable(cpp_ml_worker src/main.cpp)
target_link_libraries(cpp_ml_worker PRIVATE worker_lib)
set_target_properties(cpp_ml_worker PROPERTIES
  BUILD_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64"
  INSTALL_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64"
)

add_executable(test_local tests/test_local.cpp)
target_link_libraries(test_local PRIVATE worker_lib)
set_target_properties(test_local PROPERTIES
  BUILD_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64"
  INSTALL_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64"
)

add_executable(print_onnx src/print_onnx.cpp)
target_include_directories(print_onnx PRIVATE ${onnxruntime_SOURCE_DIR}/include)
target_link_libraries(print_onnx PRIVATE onnxruntime_imported spdlog::spdlog fmt::fmt)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(cpp_ml_worker_test tests/test_main.cpp tests/test_span_generator.cpp tests/test_spatial_layout_analyzer.cpp)
target_include_directories(cpp_ml_worker_test PRIVATE src)
target_link_libraries(cpp_ml_worker_test PRIVATE worker_lib GTest::gtest_main)
set_target_properties(cpp_ml_worker_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../../bin_tests"
  BUILD_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64"
  INSTALL_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64"
)

gtest_discover_tests(cpp_ml_worker_test DISCOVERY_MODE PRE_TEST)
