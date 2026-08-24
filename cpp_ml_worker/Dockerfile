# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM nvidia/cuda:13.0.0-devel-ubuntu26.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++-14 \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libprotobuf-dev \
    protobuf-compiler \
    libabsl-dev \
    libspdlog-dev \
    libfmt-dev \
    libmupdf-dev \
    libgtest-dev \
    libcudnn9-cuda-13 \
    libcudnn9-dev-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

ENV CXX=g++-14

WORKDIR /app

# Copy shared protobuf definitions
COPY shared_protos /app/shared_protos
COPY document_processing_service/cpp_ml_worker /app/document_processing_service/cpp_ml_worker

# Configure and build cpp_ml_worker with C++26 and CUDA support
WORKDIR /app/document_processing_service/cpp_ml_worker
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=26 \
    -DNATIVE_OPTIMIZE=OFF
RUN cmake --build build --target cpp_ml_worker -j3

# ==============================================================================
# Stage 2: Minimal GPU Runtime
# ==============================================================================
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgrpc++ \
    libprotobuf-lite \
    libmupdf \
    libspdlog1.12 \
    libfmt10 \
    libcudnn9-cuda-13 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled executable and runtime libraries
COPY --from=builder /app/document_processing_service/cpp_ml_worker/out/cpp_ml_worker /app/cpp_ml_worker

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 50060

ENTRYPOINT ["/app/cpp_ml_worker"]
