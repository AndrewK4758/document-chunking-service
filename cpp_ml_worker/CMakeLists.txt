cmake_minimum_required(VERSION 3.22)
project(cpp_ml_worker)

# --- C++26 baseline (MPL-312 §5.8) ---
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# --- Hardware-native optimization for Zen 5 (9955HX3D) and Blackwell (RTX 5070 Ti / SM120) ---
option(NATIVE_OPTIMIZE "Build for current host CPU/GPU (znver5, SM120, AVX-512)" ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

if(NATIVE_OPTIMIZE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=znver5" HAS_ZNVER5)
  if(HAS_ZNVER5)
    add_compile_options(-march=znver5 -mtune=znver5)
    message(STATUS "NATIVE_OPTIMIZE: AMD Zen 5 (-march=znver5, AVX-512) enabled")
  else()
    add_compile_options(-march=native -mtune=native)
    message(STATUS "NATIVE_OPTIMIZE: -march=native enabled")
  endif()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  message(STATUS "NATIVE_OPTIMIZE: CUDA arch ${CMAKE_CUDA_ARCHITECTURES} (Blackwell RTX 5070 Ti)")
endif()

# Silence FetchContent timestamp + custom command warnings on CMake 4.4
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0175)
  cmake_policy(SET CMP0175 NEW)
endif()

# --- Latest system deps first (Arch rolling: protobuf 35.1, absl 20260817, gRPC 1.83) ---
# Order matters: protobuf-config.cmake does find_package(absl) + utf8_range
find_package(Threads REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(utf8_range CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)
find_package(CUDAToolkit QUIET)
find_library(CUDNN_LIBRARY NAMES cudnn PATHS /usr/lib /usr/lib/x86_64-linux-gnu /opt/cuda/lib64)

include(FetchContent)

# --- onnxruntime: prefer system, fallback to latest fetch ---
# System on Arch: /usr/lib/cmake/onnxruntime (1.28.x), header /usr/include/onnxruntime
find_package(onnxruntime CONFIG QUIET)
if(TARGET onnxruntime::onnxruntime)
  message(STATUS "Using system onnxruntime: ${onnxruntime_VERSION}")
  set(ORT_TARGET onnxruntime::onnxruntime)
  # system lib has no bundled GPU path; rely on default RPATH
  set(ORT_RPATH "")
else()
  message(STATUS "System onnxruntime not found - fetching latest GPU build (v1.23.x)")
  FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(onnxruntime)
  # Fallback to asdf GPU build if present (your previous 1.29), else use fetched
  if(EXISTS "/home/ak/.asdf/installs/python/3.14.7/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0")
    set(ORT_GPU_LIB "/home/ak/.asdf/installs/python/3.14.7/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0")
    set(ORT_GPU_DIR "/home/ak/.asdf/installs/python/3.14.7/lib/python3.14/site-packages/onnxruntime/capi")
  else()
    # Use fetched onnxruntime include/lib
    set(ORT_GPU_LIB "")
    set(ORT_GPU_DIR "")
  endif()
  if(ORT_GPU_LIB AND EXISTS "${ORT_GPU_LIB}")
    add_library(onnxruntime_imported SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime_imported PROPERTIES
      IMPORTED_LOCATION "${ORT_GPU_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
    )
    set(ORT_TARGET onnxruntime_imported)
    set(ORT_RPATH "${ORT_GPU_DIR}:/opt/cuda/lib64")
  else()
    # Generic imported from FetchContent (header-only fallback)
    if(NOT TARGET onnxruntime_imported AND DEFINED onnxruntime_SOURCE_DIR)
      add_library(onnxruntime_imported SHARED IMPORTED GLOBAL)
      # Try to locate the .so inside fetched tree
      file(GLOB _ort_so "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so*")
      if(_ort_so)
        list(GET _ort_so 0 _ort_so_first)
        set_target_properties(onnxruntime_imported PROPERTIES
          IMPORTED_LOCATION "${_ort_so_first}"
          INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
        )
      else()
        # No binary - use header only, rely on system .so
        add_library(onnxruntime_imported INTERFACE IMPORTED GLOBAL)
        set_target_properties(onnxruntime_imported PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
        )
      endif()
    endif()
    set(ORT_TARGET onnxruntime_imported)
    set(ORT_RPATH "/opt/cuda/lib64")
  endif()
  if(ORT_RPATH)
    set(CMAKE_BUILD_RPATH "${ORT_RPATH}")
    set(CMAKE_INSTALL_RPATH "${ORT_RPATH}")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  endif()
endif()

# --- tokenizers-cpp: force system absl/protobuf (latest) ---
# sentencepiece defaults to internal/module which bundles abseil 20260107 and
# protobuf - that ABI-mismatches system protobuf 35.1 / absl 20260817 (lts_20260526 vs 2608).
# We must set these CACHE vars BEFORE adding tokenizers_cpp.
set(SPM_ABSL_PROVIDER "package" CACHE STRING "Use system absl" FORCE)
set(SPM_PROTOBUF_PROVIDER "package" CACHE STRING "Use system protobuf" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG main
)
FetchContent_MakeAvailable(tokenizers_cpp)

# --- simdutf: prefer system, else fetch latest ---
find_package(simdutf CONFIG QUIET)
if(NOT TARGET simdutf::simdutf)
  message(STATUS "System simdutf not found - fetching v7.x latest")
  set(SIMDUTF_TESTS OFF CACHE BOOL "" FORCE)
  set(SIMDUTF_TOOLS OFF CACHE BOOL "" FORCE)
  set(SIMDUTF_BENCHMARKS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    simdutf
    GIT_REPOSITORY https://github.com/simdutf/simdutf.git
    GIT_TAG v7.3.0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(simdutf)
endif()

if(NOT TARGET simdutf::simdutf)
  message(FATAL_ERROR "simdutf required for UTF-8 FFI boundary; install libsimdutf-dev or allow FetchContent")
endif()

# --- mupdf: system ---
find_library(MUPDF_LIB mupdf REQUIRED)
find_library(MUPDFCPP_LIB mupdfcpp QUIET)
find_path(MUPDF_INCLUDE mupdf/fitz.h PATHS /usr/include REQUIRED)
if(NOT TARGET mupdf)
  add_library(mupdf UNKNOWN IMPORTED GLOBAL)
  set_target_properties(mupdf PROPERTIES
    IMPORTED_LOCATION "${MUPDF_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUPDF_INCLUDE}"
  )
  if(MUPDFCPP_LIB)
    set_property(TARGET mupdf APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${MUPDFCPP_LIB}")
  endif()
endif()

# --- Probe <mdspan> (MPL-312 §5.8 - no polyfill) ---
include(CheckIncludeFileCXX)
check_include_file_cxx("mdspan" HAVE_MDSPAN)
if(NOT HAVE_MDSPAN)
  message(FATAL_ERROR "C++26 <mdspan> not found — requires GCC 14+ or Clang 17+ with -std=c++26")
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
  src
  ${PROTO_BUILD_DIR}
)
# Only add fetched include dirs if they exist (system onnxruntime/tokenizers already on path)
if(DEFINED onnxruntime_SOURCE_DIR)
  target_include_directories(worker_lib BEFORE PUBLIC ${onnxruntime_SOURCE_DIR}/include)
endif()
if(DEFINED tokenizers_cpp_SOURCE_DIR)
  target_include_directories(worker_lib BEFORE PUBLIC ${tokenizers_cpp_SOURCE_DIR}/include)
endif()

target_link_libraries(worker_lib PUBLIC
  gRPC::grpc++
  protobuf::libprotobuf
  utf8_range::utf8_range
  utf8_range::utf8_validity
  absl::absl_log
  absl::absl_check
  absl::strings
  absl::base
  absl::synchronization
  ${ORT_TARGET}
  $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cudart>
  $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cublas>
  $<$<BOOL:${CUDNN_LIBRARY}>:-Wl,--no-as-needed;${CUDNN_LIBRARY};-Wl,--as-needed>
  tokenizers_cpp
  simdutf::simdutf
  mupdf
  spdlog::spdlog
  fmt::fmt
  Threads::Threads
)

target_compile_features(worker_lib PUBLIC cxx_std_26)

add_executable(cpp_ml_worker src/main.cpp)
target_link_libraries(cpp_ml_worker PRIVATE worker_lib)
if(ORT_RPATH)
  set_target_properties(cpp_ml_worker PROPERTIES
    BUILD_RPATH "${ORT_RPATH}"
    INSTALL_RPATH "${ORT_RPATH}"
  )
endif()

add_executable(test_local tests/test_local.cpp)
target_link_libraries(test_local PRIVATE worker_lib)
if(ORT_RPATH)
  set_target_properties(test_local PROPERTIES
    BUILD_RPATH "${ORT_RPATH}"
    INSTALL_RPATH "${ORT_RPATH}"
  )
endif()

add_executable(print_onnx src/print_onnx.cpp)
if(TARGET onnxruntime::onnxruntime)
  target_link_libraries(print_onnx PRIVATE onnxruntime::onnxruntime spdlog::spdlog fmt::fmt)
else()
  target_include_directories(print_onnx PRIVATE ${onnxruntime_SOURCE_DIR}/include)
  target_link_libraries(print_onnx PRIVATE ${ORT_TARGET} spdlog::spdlog fmt::fmt)
endif()

enable_testing()
include(GoogleTest)

add_executable(cpp_ml_worker_test tests/test_main.cpp tests/test_span_generator.cpp tests/test_spatial_layout_analyzer.cpp)
target_include_directories(cpp_ml_worker_test PRIVATE src)
target_link_libraries(cpp_ml_worker_test PRIVATE worker_lib GTest::gtest_main)
set_target_properties(cpp_ml_worker_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../../bin_tests"
)
if(ORT_RPATH)
  set_target_properties(cpp_ml_worker_test PROPERTIES
    BUILD_RPATH "${ORT_RPATH}"
    INSTALL_RPATH "${ORT_RPATH}"
  )
endif()

gtest_discover_tests(cpp_ml_worker_test DISCOVERY_MODE PRE_TEST)
