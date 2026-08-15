package server

import (

	"fmt"
	"io"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/identityparser"
	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/AndrewK4758/shared_utils/workflowparser"
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

func extractDocUri(req *pb.ChunkDocumentRequest) (string, error) {
	if req == nil || req.OriginalRequest == nil {
		return "", fmt.Errorf("action failed for job ChunkDocument: missing original request")
	}
	parser := workflowparser.NewTaskRequestParser(req.OriginalRequest)
	payload := parser.MustGetExecuteTaskPayload()
	if payload == nil || payload.Storage == nil || payload.Storage.DocumentUri == "" {
		return "", fmt.Errorf("action failed for job ChunkDocument: missing document_uri in storage payload")
	}
	return payload.Storage.DocumentUri, nil
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

		identityparser.NewIdentityParser(req.GetIdentity()).MustValidate(true, "ChunkDocument")
		if req.GetOriginalRequest() != nil && req.GetOriginalRequest().GetTaskSubject() == "" {
			panic(fmt.Errorf("action failed for job ChunkDocument: missing required TaskSubject in OriginalRequest"))
		}

		docUri, err := extractDocUri(req)
		if err != nil {
			logger.Info("DocProcessor", "Error extracting Document URI: %v", err)
			err = stream.Send(&pb.ChunkDocumentResponse{
				Success: false,
				ErrorMessage: err.Error(),
			})
			if err != nil {
				return fmt.Errorf("failed to send chunk document response: %w", err)
			}
			continue
		}

		reqJson, err := protojson.Marshal(req)
		if err != nil {
			logger.Info("DocProcessor", "Received ChunkDocument request for JobID: %s, DocumentUri: %s, Payload: failed to marshal", req.Identity.GetJobId(), docUri)
		} else {
			logger.Info("DocProcessor", "Received ChunkDocument request for JobID: %s, DocumentUri: %s, Payload: %s", req.Identity.GetJobId(), docUri, string(reqJson))
		}

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
