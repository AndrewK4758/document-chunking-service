# Service Communication

The C++ ML Worker Service exposes a high-performance gRPC interface defined in `shared_protos/ml_worker.proto` and consumes the Storage Gateway service.

## Inbound gRPC Interface: `CppMlWorkerService`

- **RPC Method:** `RedactDocument(RedactDocumentRequest) returns (RedactDocumentResponse)`
- **Request Payload:**
  - `identity`: Strongly-typed tenant, job, and execution IDs.
  - `document_uri`: Storage Gateway URI of the source PDF.
  - `entity_labels`: Target labels for GLiNER zero-shot detection (e.g. `["person_name", "ssn", "address"]`).
  - `confidence_threshold`: Minimum probability threshold for redaction.
- **Response Payload:**
  - `redacted_document_uri`: Storage URI of the output redacted PDF.
  - `detected_spans`: List of extracted spans with bounding box coordinates, text values, and confidence scores.

## Outbound Communication

- **Storage Gateway (`storage_gateway_service`)**:
  - Connects via gRPC to download input documents and stream back redacted outputs.
- **Fail-Fast Policy**:
  - Any missing required identity, invalid storage URI, or ONNX engine error triggers an immediate `grpc::Status::INVALID_ARGUMENT` or `grpc::Status::INTERNAL` with explicit diagnostics.
