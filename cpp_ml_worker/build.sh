#!/usr/bin/env bash
set -euo pipefail

# build.sh — Hardware-native optimized build for any deploy host
# Detects CPU/GPU/RAM/OS at runtime and dynamically creates correct flags.
#   CPU: gcc -march=native (znver5, skylake, aarch64, etc) + AVX-512/VNNI/BF16 detection
#   GPU: NVIDIA compute_cap -> SM arch (120 for 5070 Ti, etc), CUDA/cuDNN version, ROCm
#   RAM: jobs = nproc, but capped for low-memory hosts
#   Build: Release + LTO + -march=native + NATIVE_OPTIMIZE=ON (best pipeline, long build OK)
# Usage: ./build.sh [Release|Debug]   (default: Release)
#        REBUILD_ORT=1 ./build.sh     to force onnxruntime rebuild from PKGBUILD

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="${1:-Release}"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=========================================="
echo " Building C++26 ML Worker (${BUILD_TYPE}) — Hardware-Native"
echo "=========================================="

# ---------- Host detection ----------
echo "[detect] Host: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"' || echo unknown)  kernel $(uname -r)  $(uname -m)"

# CPU
CPU_MODEL="$(lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//' | xargs || echo unknown)"
CPU_ARCH="$(gcc -march=native -Q --help=target 2>/dev/null | grep -m1 'march=' | awk '{print $2}' || true)"
if [[ -z "${CPU_ARCH:-}" ]]; then CPU_ARCH="native"; fi
CPU_FLAGS_RAW="$(grep -m1 flags /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo)"
echo "[detect] CPU: $CPU_MODEL"
echo "[detect] gcc -march=$CPU_ARCH"
HAS_AVX512=0; HAS_BF16=0; HAS_VNNI=0; HAS_VNNI512=0; HAS_AVX2=0
grep -q avx512f <<< "$CPU_FLAGS_RAW" && HAS_AVX512=1
grep -q avx512_bf16 <<< "$CPU_FLAGS_RAW" && HAS_BF16=1
grep -q "avx_vnni\|avx512_vnni" <<< "$CPU_FLAGS_RAW" && HAS_VNNI=1
grep -q avx512_vnni <<< "$CPU_FLAGS_RAW" && HAS_VNNI512=1
grep -q avx2 <<< "$CPU_FLAGS_RAW" && HAS_AVX2=1
echo "[detect] Features: AVX2=$HAS_AVX2 AVX512=$HAS_AVX512 BF16=$HAS_BF16 VNNI=$HAS_VNNI VNNI512=$HAS_VNNI512"
echo "[detect] Flags sample: $(echo $CPU_FLAGS_RAW | cut -d' ' -f1-12) ..."

# Memory / jobs
MEM_GB=$(free -g 2>/dev/null | awk '/Mem:/ {print $2}' || echo 16)
JOBS=$(nproc 2>/dev/null || echo 4)
if [[ "$MEM_GB" -lt 8 && "$JOBS" -gt 8 ]]; then JOBS=8; echo "[detect] Low RAM (${MEM_GB}G), capping jobs to $JOBS"; fi
echo "[detect] Jobs: $JOBS (RAM ${MEM_GB}G)"

# GPU — NVIDIA
CUDA_AVAILABLE=0; GPU_NAME=""; COMPUTE_CAP=""; SM_NUM=""; SM_ARCH=""; CUDA_VER=""; CUDNN_VER=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)"
  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)"
  if [[ -n "$COMPUTE_CAP" ]]; then
    CUDA_AVAILABLE=1
    SM_NUM="$(echo "$COMPUTE_CAP" | tr -d '.' || true)"
    SM_ARCH="${SM_NUM}-real;${SM_NUM}-virtual"
    CUDA_VER="$(pacman -Q cuda 2>/dev/null | awk '{print $2}' || nvcc --version 2>&1 | grep release | sed 's/.*release //' | cut -d, -f1 || echo unknown)"
    CUDNN_VER="$(pacman -Q cudnn 2>/dev/null | awk '{print $2}' || grep -m1 CUDNN_MAJOR /usr/include/cudnn_version.h 2>/dev/null | awk '{print $3}' || echo unknown)"
    echo "[detect] GPU: $GPU_NAME (compute_cap $COMPUTE_CAP -> SM $SM_ARCH)"
    echo "[detect] CUDA: $CUDA_VER  cuDNN: $CUDNN_VER  driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | xargs || echo unknown)"
  fi
else
  echo "[detect] GPU: none (CPU-only)"
fi

# GPU — ROCm
if command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
  echo "[detect] ROCm: detected ($(rocm-smi --showproductname 2>&1 | head -n1 | xargs || echo present))"
fi
if [[ -e /opt/rocm/bin/rocminfo ]]; then echo "[detect] ROCm: /opt/rocm present"; fi

# Arch / OS packager
if command -v pacman &>/dev/null; then PM="pacman (Arch)"; elif command -v apt &>/dev/null; then PM="apt (Debian/Ubuntu)"; elif command -v dnf &>/dev/null; then PM="dnf (Fedora)"; else PM="unknown"; fi
echo "[detect] Packager: $PM  compiler: $(gcc --version 2>&1 | head -n1)  cmake $(cmake --version 2>&1 | head -n1)"

# ---------- Dynamic flags ----------
NATIVE_CXXFLAGS="-march=native -mtune=native"
NATIVE_CFLAGS="$NATIVE_CXXFLAGS"
if [[ "$BUILD_TYPE" == "Release" ]]; then
  NATIVE_CXXFLAGS="$NATIVE_CXXFLAGS -O3 -DNDEBUG"
  NATIVE_CFLAGS="$NATIVE_CFLAGS -O3 -DNDEBUG"
else
  NATIVE_CXXFLAGS="$NATIVE_CXXFLAGS -O2 -g"
  NATIVE_CFLAGS="$NATIVE_CFLAGS -O2 -g"
fi
echo "[flags] NATIVE CXXFLAGS: $NATIVE_CXXFLAGS"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_CXX_STANDARD=26
  -DCMAKE_CXX_STANDARD_REQUIRED=ON
  -DNATIVE_OPTIMIZE=ON
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
)

# CUDA arch: only set if we have a GPU, let onnxruntime PKGBUILD handle its own _valid_sm() but
# pass it to our worker if it ever needs CUDA kernels
if [[ "$CUDA_AVAILABLE" == "1" && -n "$SM_NUM" ]]; then
  # CMake 3.22+ wants numeric arch like 120; keep both for compatibility
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$SM_NUM")
  echo "[flags] CMAKE_CUDA_ARCHITECTURES=$SM_NUM"
fi

# ---------- Optional: rebuild onnxruntime for this host ----------
# Catches the Arch cudnn 9 skew (v7 symbol) and ensures SM_NATIVE
if [[ "${REBUILD_ORT:-0}" == "1" && "$CUDA_AVAILABLE" == "1" ]]; then
  echo ""
  echo "=========================================="
  echo " Rebuilding onnxruntime-cuda for this host (REBUILD_ORT=1)"
  echo " Host: $GPU_NAME SM $SM_ARCH  $CUDA_VER / $CUDNN_VER  $CPU_ARCH"
  echo "=========================================="
  export CFLAGS="-march=native -mtune=native -O3 -pipe -fno-plt -fexceptions -Wp,-D_FORTIFY_SOURCE=3"
  export CXXFLAGS="$CFLAGS -Wp,-D_GLIBCXX_ASSERTIONS"
  export MAKEFLAGS="-j$JOBS"
  export CUDAFLAGS="-Xcompiler -march=native -Xcompiler -O3"
  export NVCC_CCBIN="$(which g++ 2>/dev/null || echo g++)"
  PKG_SRC="/tmp/onnxruntime-hw-optimized"
  if [[ ! -d "$PKG_SRC" ]]; then
    git clone https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime.git "$PKG_SRC"
  else
    git -C "$PKG_SRC" pull --ff-only || true
  fi
  cd "$PKG_SRC"
  if grep -q "$SM_ARCH" PKGBUILD 2>/dev/null; then
    echo "[ORT] SM $SM_ARCH already in PKGBUILD"
  else
    echo "[ORT] SM $SM_ARCH not in list — building with defaults (still uses native CFLAGS)"
  fi
  makepkg -srif --clean --noconfirm
  if nm -D /usr/lib/libonnxruntime_providers_cuda.so 2>/dev/null | grep -q cudnnGetConvolutionBackwardDataAlgorithm_v7; then
    echo "[ORT] WARNING: v7 symbol still present"
  else
    echo "[ORT] OK: no v7 symbol (cudnn 9 native)"
  fi
  cd "$SCRIPT_DIR"
elif [[ "$CUDA_AVAILABLE" == "1" ]]; then
  # Auto-detect stale onnxruntime without forcing rebuild
  if nm -D /usr/lib/libonnxruntime_providers_cuda.so 2>/dev/null | grep -q cudnnGetConvolutionBackwardDataAlgorithm_v7; then
    echo "[warn] onnxruntime-cuda still references cudnn v7 (stale vs $CUDNN_VER) — run REBUILD_ORT=1 ./build.sh to rebuild native"
  fi
fi

# ---------- Configure & build cpp_ml_worker ----------
echo ""
echo "=========================================="
echo " Configuring cpp_ml_worker ($BUILD_TYPE, native LTO)"
echo "=========================================="
mkdir -p "$BUILD_DIR"
# Pass native flags via CMake (NATIVE_OPTIMIZE adds -march=native) plus Release flags
cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR" "${CMAKE_ARGS[@]}" \
  -DCMAKE_CXX_FLAGS_RELEASE="$NATIVE_CXXFLAGS" \
  -DCMAKE_C_FLAGS_RELEASE="$NATIVE_CFLAGS" \
  -DCMAKE_CXX_FLAGS_DEBUG="$NATIVE_CXXFLAGS" \
  -DCMAKE_C_FLAGS_DEBUG="$NATIVE_CFLAGS"

echo "Compiling with $JOBS jobs (native)..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j"$JOBS"

echo ""
echo "=========================================="
echo " Build Succeeded (Native)!"
echo " Host: $CPU_MODEL | $CPU_ARCH | GPU: ${GPU_NAME:-none} ($COMPUTE_CAP) SM:$SM_NUM"
echo " Artifacts: $SCRIPT_DIR/out and $BUILD_DIR"
ls -lh "$SCRIPT_DIR/out/cpp_ml_worker" "$BUILD_DIR/libworker_lib.a" 2>&1 | head -n 10
echo "=========================================="
echo " Verify: ldd out/cpp_ml_worker | grep -E 'onnx|protobuf|absl|cuda' ; timeout 5 out/cpp_ml_worker"
