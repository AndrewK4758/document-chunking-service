package server

import (
	"context"
	"log"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"
	pb "github.com/AndrewK4758/shared_protos"
)

type DocumentProcessingServer struct {
	pb.UnimplementedDocumentProcessingServiceServer
	chunker *processor.ChunkingProcessor
}

func NewDocumentProcessingServer(aiClient pb.DocumentProcessorClient, store storage.StorageManager) *DocumentProcessingServer {
	return &DocumentProcessingServer{
		chunker: processor.NewChunkingProcessor(aiClient, store),
	}
}

func (s *DocumentProcessingServer) ChunkDocument(ctx context.Context, req *pb.ChunkDocumentRequest) (*pb.ChunkDocumentResponse, error) {
	log.Printf("Received ChunkDocument request for JobID: %s, RelativePath: %s", req.JobId, req.RelativeFilePath)

	// Call the processor logic to handle the extraction, chunking, and classification
	docs, traceEvents, err := s.chunker.ProcessZipFile(ctx, req)
	if err != nil {
		log.Printf("Error processing zip file for JobID %s: %v", req.JobId, err)
		return &pb.ChunkDocumentResponse{
			Success:      false,
			ErrorMessage: err.Error(),
			TraceEvents:  traceEvents,
		}, nil
	}

	log.Printf("Successfully processed zip file for JobID %s, extracted %d documents", req.JobId, len(docs))
	return &pb.ChunkDocumentResponse{
		Success:            true,
		ProcessedDocuments: docs,
		TraceEvents:        traceEvents,
	}, nil
}
