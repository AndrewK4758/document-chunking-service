package server

import (
	"fmt"
	"io"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/AndrewK4758/shared_utils/workflowparser"
)

type DocumentProcessingServer struct {
	pb.UnimplementedDocumentProcessingServiceServer
	chunker *processor.ChunkingProcessor
}

func NewDocumentProcessingServer(aiClient pb.DocumentProcessorClient, store storage.Manager) *DocumentProcessingServer {
	return &DocumentProcessingServer{
		chunker: processor.NewChunkingProcessor(aiClient, store),
	}
}

func extractDocUri(req *pb.ChunkDocumentRequest) string {
	if req == nil {
		return "unknown"
	}
	if req.OriginalRequest == nil {
		return "unknown"
	}
	parser := workflowparser.NewNodeRequestParser(req.OriginalRequest)
	uri := parser.GetDocumentUri()
	if uri == "" {
		return "unknown"
	}
	return uri
}

func (s *DocumentProcessingServer) ChunkDocument(stream pb.DocumentProcessingService_ChunkDocumentServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("failed to receive chunk document request from stream: %w", err)
		}

		docUri := extractDocUri(req)
		logger.Info("DocProcessor", "Received ChunkDocument request for JobID: %s, DocumentUri: %s", req.Identity.JobId, docUri)

		docs, traceEvents, err := s.chunker.ProcessZipFile(stream.Context(), req)
		resp := &pb.ChunkDocumentResponse{
			TraceEvents: traceEvents,
		}

		if err != nil {
			logger.Info("DocProcessor", "Error processing zip file for JobID %s: %v", req.Identity.JobId, err)
			resp.Success = false
			resp.ErrorMessage = err.Error()
		}

		if err == nil {
			logger.Info("DocProcessor", "Successfully processed zip file for JobID %s, extracted %d documents", req.Identity.JobId, len(docs))
			resp.Success = true
			resp.ProcessedDocuments = docs
		}

		err = stream.Send(resp)
		if err != nil {
			return fmt.Errorf("failed to send chunk document response to stream: %w", err)
		}
	}
}
