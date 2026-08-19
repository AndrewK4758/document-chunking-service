# Core Components

The **C++ ML Worker Service** provides native, hardware-accelerated machine learning inference and document layout analysis for the Document Processing subsystem.

## Key Subsystems

### 1. `MLInferenceEngine` & ONNX Runtime
- Loads zero-shot **GLiNER** (Generalist and Lightweight Model for Named Entity Recognition) models in ONNX format.
- Uses C++26 zero-copy buffers (`std::span` and `std::mdspan`) to feed pre-allocated input tensors into the ONNX Runtime session.
- Runs bidirectional token and span classification to identify arbitrary target entity types (e.g., names, addresses, dates, account numbers) without retraining.

### 2. `MuPDFWrapper` (Document Parsing & Layout)
- Direct native integration with **MuPDF / fitz** C-API for high-throughput PDF parsing.
- Extracts character-level bounding boxes, line coordinates, and page dimensions.
- Applies visual redaction masks directly onto PDF pages while preserving document geometry and metadata.

### 3. `SpanGenerator` (Candidate Windowing)
- Implements sliding-window and n-gram candidate span generation across tokenized words.
- Maps token spans back to physical PDF page bounding boxes for pixel-accurate redactions.

### 4. `StorageGatewayClient`
- High-throughput gRPC client connecting directly to the `storage_gateway_service` to download original documents and upload processed/redacted artifacts.
