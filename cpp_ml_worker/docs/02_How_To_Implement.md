# How to Implement

## 🚀 Building & Running the C++ ML Worker Service

### Prerequisites
- Modern C++26 compiler (`clang++-20+` or `g++-15+`)
- CMake 3.28+
- ONNX Runtime 1.17+
- MuPDF development libraries
- Protobuf 3.21+ & gRPC C++ plugins

### Local Build with CMake
```bash
cd /home/ak/projects/mployee/document_processing_service/cpp_ml_worker
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Running Unit & Regression Tests (Google Test)
```bash
cd /home/ak/projects/mployee/document_processing_service/cpp_ml_worker/build
ctest --output-on-failure
```

### Local Execution
```bash
cd /home/ak/projects/mployee/document_processing_service/cpp_ml_worker
export PORT=50058
export MODEL_PATH=models/gliner_model.onnx
export TOKENIZER_PATH=models/tokenizer.json
export STORAGE_GATEWAY_URI=localhost:50056
./build/out/cpp_ml_worker
```

### Docker Deployment
```bash
cd /home/ak/projects/mployee/document_processing_service/cpp_ml_worker
docker build -t cpp_ml_worker .
docker run -p 50058:50058 --env-file .env.docker cpp_ml_worker
```
