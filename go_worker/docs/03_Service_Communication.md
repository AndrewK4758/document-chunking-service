# Service Communication

## 🏗 Architecture

The Orchestrator communicates with this service directly over **gRPC**. 

The Orchestrator sends the raw binary payload to this service, which processes it, splits it into chunks, and returns the structured data synchronously for the workflow state machine to proceed.

## 📜 Data Contracts & No Preemptive Serialization

This service adheres to the strict **No Preemptive Serialization** mandate. 
- All JSON-based structural state must be parsed natively using `google.protobuf.Struct` (or `structpb.Struct` in Go, `Dictionary<string, object>` in C#).
- Preemptive stringification (e.g. converting nested objects to strings before the final persistence layer) is forbidden to prevent schema drift, whitespace corruption, and nested parsing bugs.
- Boundary interfaces rely on typed object mappings rather than dynamic regex or fuzzy key lookups.
