# Core Components

The **.NET ML Worker Service** provides managed .NET inference and document redaction services within the Document Processing architecture.

## Key Components

### 1. `GlinerModelService`
- Integrates Microsoft.ML.OnnxRuntime to load and evaluate ONNX-exported GLiNER models.
- Manages tensor memory and tokenized input representations for zero-shot named entity recognition.

### 2. `RedactionPipeline`
- Orchestrates multi-label span classification and coordinate matching for sensitive data fields.
- Filters candidate predictions against user-defined confidence thresholds.

### 3. `MlWorkerServiceImpl`
- gRPC service implementation exposing high-throughput prediction and redaction RPCs to upstream orchestrators.
