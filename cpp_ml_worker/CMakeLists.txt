cmake_minimum_required(VERSION 3.15)
project(cpp_ml_worker)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz
  )
  FetchContent_MakeAvailable(onnxruntime)
endif()

# Fetch tokenizers-cpp
FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG main
)

FetchContent_MakeAvailable(tokenizers_cpp)


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
    src/SpanGenerator.cpp
    ${PROTO_SRCS} ${GRPC_SRCS}
)

target_include_directories(worker_lib PUBLIC
  src
  ${PROTO_BUILD_DIR}
  ${onnxruntime_SOURCE_DIR}/include
  ${tokenizers_cpp_SOURCE_DIR}/include
)

target_link_libraries(worker_lib PUBLIC
  gRPC::grpc++
  protobuf::libprotobuf
  onnxruntime
  tokenizers_cpp
  mupdf
  spdlog::spdlog
  fmt::fmt
)

add_executable(cpp_ml_worker src/main.cpp)
target_link_libraries(cpp_ml_worker PRIVATE worker_lib)

add_executable(test_local tests/test_local.cpp)
target_link_libraries(test_local PRIVATE worker_lib)

add_executable(print_onnx src/print_onnx.cpp)
target_include_directories(print_onnx PRIVATE ${onnxruntime_SOURCE_DIR}/include)
target_link_libraries(print_onnx PRIVATE onnxruntime spdlog::spdlog fmt::fmt)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(cpp_ml_worker_test tests/test_main.cpp tests/test_span_generator.cpp)
target_include_directories(cpp_ml_worker_test PRIVATE src)
target_link_libraries(cpp_ml_worker_test PRIVATE worker_lib GTest::gtest_main)
gtest_discover_tests(cpp_ml_worker_test)
