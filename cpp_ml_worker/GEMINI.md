# C++23/26 Architecture & Agent Rules (cpp_ml_worker)

This document is the absolute source of truth for all C++ implementations within the ML Worker service. It extends the Global `GEMINI.md` mandates specifically for the modern C++ ecosystem.

## 🛑 1. Core Directives & Fail Fast
- **No Defensive Fallbacks**: If a required pointer (e.g., ONNX model weights, MuPDF context) is null, or an expected input is malformed and cannot be mathematically evaluated, you MUST `throw std::runtime_error` or call `std::terminate()`. Do not return an empty object just to "keep things moving."
- **RAII is Absolute**: You must NEVER use naked `new` or `delete`. Every resource (MuPDF contexts, ONNX sessions, file handles) must be managed by smart pointers (`std::unique_ptr`) or custom RAII wrappers. Memory leaks in long-running ML inference loops are catastrophic.
- **Throw on Construction**: If an object cannot be fully initialized (e.g., `MLInferenceEngine` fails to load the ONNX model from disk), throw an exception in the constructor. Do not rely on `Init()` methods or `IsValid()` checks.

## 🚀 2. Modern C++23 & C++26 Features
- **Ranges and Views**: Prefer `std::ranges` and `std::views` over traditional `for` loops and standard algorithms when transforming or filtering data.
- **std::expected / std::optional**: For operations that can gracefully fail (and where a crash is not required by the Fail Fast rule), return `std::expected<T, E>` (C++23) or `std::optional<T>` instead of out-parameters or magic return codes.
- **std::span and std::mdspan**: When passing arrays or buffers (especially ONNX tensors or token arrays), pass by `std::span<T>`. For multi-dimensional tensor data (like GLiNER's `span_idx`), use C++23's `std::mdspan` instead of manual index math (e.g., `buffer[batch * stride + index]`).
- **std::print and std::println**: Utilize C++23 `<print>` formatting instead of `std::cout` or `std::stringstream` where possible for logging and CLI output.
- **Deducing This**: Utilize C++23 explicit object parameters (`this auto&& self`) if it simplifies const/non-const method overloading.
- **constexpr Everything**: If a value or a function can be computed at compile time, mark it `constexpr` or `consteval`.

## 🧠 3. ML Inference & Memory Management
- **Zero-Copy when Possible**: When passing data between the MuPDF wrapper, tokenizers, and ONNX Runtime, avoid deep copies. Map raw data buffers directly to `Ort::Value::CreateTensor` using `std::span` or raw pointers mapped to the source data's lifetime.
- **Pre-allocate Hot Paths**: The ML worker loop processes thousands of spans per page. Do not repeatedly allocate `std::vector` inside the inner loops. Use `.reserve()` aggressively and hoist allocations outside of the page/span loops.
- **Sanitize FFI Boundaries**: When receiving `std::string` data from a C library (like MuPDF) to pass to a Rust/C++ FFI (like `tokenizers-cpp`), you MUST sanitize the UTF-8 boundaries. A panic in an FFI boundary will crash the entire C++ process without unwinding.

## 🧹 4. Linting, Code Quality, and Testing
- **Const Correctness**: Every variable, pointer, and method must be `const` unless it is explicitly mutated. 
- **Clang-Tidy**: All code must conform to strict `clang-tidy` checks, specifically modernizing checks (`modernize-*`), performance checks (`performance-*`), and concurrency checks (`concurrency-*`).
- **Test-Driven ML Logic**: Complex algorithmic logic (like sliding-window span generation for GLiNER) MUST be unit-tested in isolation using Google Test (GTest). Do not test span generation logic solely by running end-to-end PDF tests.
- **Fuzz Testing**: The ML worker parses arbitrary, unstructured byte streams (PDFs). You must assume the bytes are malformed. Fuzz the MuPDF wrappers and string sanitizers to ensure they gracefully catch formatting errors and throw, rather than segfaulting.

## 🏗️ 5. gRPC and Protobuf
- **Schema Ownership**: The Protobuf schemas (`ml_worker.proto`) dictate the exact inputs and outputs. Do not add custom properties or C++ side-channels that are not represented in the `.proto` files.
- **Error Propagation**: Any exception caught in the top-level gRPC handler must be converted to a `grpc::Status` with a clear, descriptive error message (including the underlying failure reason) before being returned to the Orchestrator. 
