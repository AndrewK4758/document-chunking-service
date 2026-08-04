package server

import (
	"fmt"
	"io"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/identityparser"
	"github.com/AndrewK4758/shared_utils/logger"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
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
	if req.OriginalRequest == nil || req.OriginalRequest.Payload == nil {
		return "unknown"
	}
	payload := req.OriginalRequest.Payload.AsMap()
	uri, _ := payload["document_uri"].(string)
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

		if err := identityparser.NewIdentityParser(req.GetIdentity()).Validate(true); err != nil {
			logger.Error("DocProcessor", "FAIL-FAST ASSERTION FAILED (gRPC ChunkDocument): %v", err)
			return status.Errorf(codes.InvalidArgument, "invalid identity: %v", err)
		}
		if req.GetOriginalRequest() != nil && req.GetOriginalRequest().GetTaskSubject() == "" {
			err := fmt.Errorf("missing required TaskSubject in OriginalRequest")
			logger.Error("DocProcessor", "FAIL-FAST ASSERTION FAILED (gRPC ChunkDocument): %v", err)
			return status.Errorf(codes.InvalidArgument, "%v", err)
		}

		docUri := extractDocUri(req)
		reqJson, _ := protojson.Marshal(req)
		logger.Info("DocProcessor", "Received ChunkDocument request for JobID: %s, DocumentUri: %s, Payload: %s", req.Identity.GetJobId(), docUri, string(reqJson))

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
