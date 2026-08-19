#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="${1:-Debug}"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=========================================="
echo " Building C++26 ML Worker (${BUILD_TYPE})"
echo "=========================================="

mkdir -p "${BUILD_DIR}"

# 1. Configure CMake with C++26 standard
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_CXX_STANDARD=26 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON

# 2. Compile targets in parallel
echo "Compiling targets (cpp_ml_worker, cpp_ml_worker_test, test_local)..."
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j"$(nproc)"

echo "=========================================="
echo " Build Succeeded!"
echo " Artifacts located in: ${SCRIPT_DIR}/out and ${BUILD_DIR}"
echo "=========================================="
