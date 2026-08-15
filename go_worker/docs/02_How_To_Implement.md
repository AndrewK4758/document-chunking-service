# How to Implement

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
export PORT=11114
go run cmd/server/main.go
```

### Local Development (Windows PowerShell)
```powershell
cd \home\ak\projects\doc_processor_service\document_processing_service
$env:PORT="11114"
go run cmd\server\main.go
```
