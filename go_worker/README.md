# Document Processing Service

Welcome to the **Document Processing Service**, a dedicated worker in the enterprise AI ecosystem responsible for structural payload manipulation. It handles tasks like unzipping packages, parsing images, and intelligently chunking large PDFs before they are routed to pure LLM extraction nodes.

## Documentation Index

1. **[Core Components](docs/01_Core_Components.md)**  
   Overview of the document manipulation worker and its operations.
2. **[How to Implement](docs/02_How_To_Implement.md)**  
   Instructions for running the service using Docker or local development environments.
3. **[Service Communication](docs/03_Service_Communication.md)**  
   Details on gRPC communication with the Orchestrator and data contract mandates.

---

## High-Level Architecture

```mermaid
flowchart TD
    subgraph Clients
        A[Orchestrator Service]
    end

    subgraph Document Processing Service
        B[gRPC Server]
        C[Payload Unzipper]
        D[Image Parser]
        E[PDF Chunker]
    end

    A -->|gRPC Request (Raw Binary)| B
    B --> C
    B --> D
    B --> E
    
    C -->|Processed Chunks| B
    D -->|Processed Chunks| B
    E -->|Processed Chunks| B
    
    B -->|gRPC Response (Structured Byte Arrays)| A
```
