# C++ ML Worker Service (cpp_ml_worker)

Welcome to the **C++ ML Worker Service**, a high-performance native C++26 microservice within the Document Processing subsystem. It provides ultra-fast zero-shot Named Entity Recognition (NER), entity span extraction, and document redaction using ONNX Runtime and MuPDF.

## Documentation Index

1. **[Core Components](docs/01_Core_Components.md)**  
   Overview of the GLiNER inference engine, MuPDF layout extraction, and tokenization pipeline.
2. **[How to Implement](docs/02_How_To_Implement.md)**  
   Instructions for building, running tests with Google Test, and deploying via Docker or CMake.
3. **[Service Communication](docs/03_Service_Communication.md)**  
   Details on gRPC communication implementing `CppMlWorkerService` and integration with `StorageGatewayClient`.

---

## High-Level Architecture

```mermaid
flowchart TD
    subgraph Upstream
        Orch[Orchestrator / Go Worker]
        Storage[Storage Gateway Service]
    end

    subgraph C++ ML Worker Service
        Server[gRPC Server: CppMlWorkerService]
        StorageClient[Storage Gateway Client]
        MuPDF[MuPDF Wrapper]
        Tokenizer[tokenizers-cpp]
        Engine[ML Inference Engine]
        ONNX[(ONNX Runtime: GLiNER Model)]
        SpanGen[Span Generator]
    end

    Orch -->|gRPC: RedactDocument| Server
    Server -->|Download PDF| StorageClient
    StorageClient -->|gRPC Download| Storage
    Storage -->|Raw PDF Bytes| StorageClient
    StorageClient -->|Raw PDF| Server
    Server -->|Parse PDF Layout| MuPDF
    MuPDF -->|Text & Bounding Boxes| SpanGen
    SpanGen -->|Candidate Spans| Tokenizer
    Tokenizer -->|Input Tensors| Engine
    Engine -->|Inference Execution| ONNX
    ONNX -->|Entity Logits & Spans| Engine
    Engine -->|Classified Entities| Server
    Server -->|Redacted PDF & Spans| Orch
```
