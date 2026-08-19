# Service Communication

The .NET ML Worker Service provides a gRPC interface implementing the ML Worker contract defined in `shared_protos/ml_worker.proto`.

## Inbound gRPC Interface

- **Service:** `ml.worker.v1.MLWorkerService`
- **RPC:** `ExtractEntities(ExtractEntitiesRequest) returns (ExtractEntitiesResponse)`
- **Payload:**
  - `identity`: Multi-tenant tracking IDs (TenantId, AppId, JobId).
  - `text`: Raw text payload or page content to classify.
  - `entity_types`: List of target entity types for zero-shot recognition.
  - `threshold`: Score confidence threshold.
