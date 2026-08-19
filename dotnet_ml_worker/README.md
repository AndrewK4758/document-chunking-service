# .NET ML Worker Service (dotnet_ml_worker)

Welcome to the **.NET ML Worker Service**, a high-performance C# / .NET 9 microservice within the Document Processing subsystem. It hosts GLiNER ONNX models and Microsoft ML pipelines to perform zero-shot entity extraction and redaction via gRPC.

## Documentation Index

1. **[Core Components](docs/01_Core_Components.md)**  
   Overview of the .NET GLiNER model service, redaction pipeline, and gRPC endpoints.
2. **[How to Implement](docs/02_How_To_Implement.md)**  
   Instructions for running, testing via xUnit, and deploying the .NET ML worker.
3. **[Service Communication](docs/03_Service_Communication.md)**  
   Details on gRPC integration with the Document Processor and Orchestrator.

---

## High-Level Architecture

```mermaid
flowchart TD
    subgraph Clients
        Orch[Document Processor / Orchestrator]
    end

    subgraph .NET ML Worker
        GRPC[MlWorkerServiceImpl (gRPC)]
        Pipeline[RedactionPipeline]
        Model[GlinerModelService (ONNX Runtime)]
    end

    Orch -->|gRPC: ExtractEntities / Redact| GRPC
    GRPC --> Pipeline
    Pipeline --> Model
    Model -->|Entity Predictions & Scores| Pipeline
    Pipeline --> GRPC
    GRPC -->|Structured Spans & Redaction Result| Orch
```
