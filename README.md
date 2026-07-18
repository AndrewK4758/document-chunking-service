# Document Processing Service

The Document Processing Service handles structural payload manipulation within the enterprise AI ecosystem. It acts as a dedicated worker to unzip packages, parse images, and intelligently chunk large PDFs before they are routed to the pure LLM extraction steps.

## 🏗 Architecture

The Orchestrator communicates with this service directly over **gRPC**. 
Unlike the pure LLM workers (which operate asynchronously via NATS JetStream), document manipulation is often required immediately as a precursor node (e.g., `PRIMITIVE_DOCUMENT_CHUNKING`). 

The Orchestrator sends the raw binary payload to this service, which processes it using optimized Go libraries, splits it into parallel-friendly chunks, and returns the structured byte arrays for the state machine to continue.

## 🚀 Running the Document Processing Service

Because this service depends on the `shared_protos`, it must be run within the context of the parent `doc_processor_service` workspace.

### Docker (All Platforms)
To run the entire ecosystem together seamlessly (Recommended):
```bash
cd /home/ak/projects/doc_processor_service
docker-compose up -d --build
```
The Document Processing Service will be exposed internally to the Orchestrator.

### Local Development (Linux / macOS)
```bash
cd /home/ak/projects/doc_processor_service/document_processing_service
export PORT=50054
go run cmd/server/main.go
```

### Local Development (Windows PowerShell)
```powershell
cd \home\ak\projects\doc_processor_service\document_processing_service
$env:PORT="50054"
go run cmd\server\main.go
```
