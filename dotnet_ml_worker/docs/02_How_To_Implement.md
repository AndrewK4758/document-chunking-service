# How to Implement

## 🚀 Running the .NET ML Worker Service

### Local Development (.NET 9 SDK)
```bash
cd /home/ak/projects/mployee/document_processing_service/dotnet_ml_worker
dotnet build
dotnet run
```

### Running Unit & Integration Tests
```bash
cd /home/ak/projects/mployee/document_processing_service/dotnet_ml_worker
dotnet test
```

### Docker Deployment
```bash
cd /home/ak/projects/mployee/document_processing_service/dotnet_ml_worker
docker build -t dotnet_ml_worker .
docker run -p 50059:50059 dotnet_ml_worker
```
