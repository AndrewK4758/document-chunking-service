FROM ubuntu:26.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libprotobuf-dev \
    protobuf-compiler \
    libcurl4-openssl-dev \
    libssl-dev \
    pkg-config \
    libgtest-dev

WORKDIR /app
COPY . .

RUN mkdir build && cd build && cmake .. && make -j4

EXPOSE 50060

CMD ["/app/build/cpp_ml_worker"]
