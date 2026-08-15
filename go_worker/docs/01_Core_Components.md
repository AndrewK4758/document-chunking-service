# Core Components

The Document Processing Service handles structural payload manipulation within the enterprise AI ecosystem. It acts as a dedicated worker to unzip packages, parse images, and intelligently chunk large PDFs before they are routed to the pure LLM extraction steps.

## Document Manipulation Worker
Unlike the pure LLM workers (which operate asynchronously via NATS JetStream), document manipulation is often required immediately as a precursor node (e.g., `PRIMITIVE_DOCUMENT_CHUNKING`).

The service uses optimized Go libraries to:
* Unzip packages.
* Parse images.
* Chunk large PDFs into parallel-friendly segments.
* Return structured byte arrays for the state machine to continue.
