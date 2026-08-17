package server

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	"golang.org/x/sync/errgroup"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/identityparser"
	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/AndrewK4758/shared_utils/workflowparser"
	"google.golang.org/protobuf/encoding/protojson"
)

type DocumentProcessingServer struct {
	pb.UnimplementedDocumentProcessingServiceServer
	chunker      *processor.ChunkingProcessor
	cppMlWorker  pb.CppMlWorkerServiceClient
	netMlWorker  pb.MlWorkerServiceClient
	store        storage.Manager
}

func NewDocumentProcessingServer(aiClient pb.DocumentProcessorClient, cppMlWorker pb.CppMlWorkerServiceClient, netMlWorker pb.MlWorkerServiceClient, store storage.Manager) *DocumentProcessingServer {
	chunker := processor.NewChunkingProcessor(aiClient, store)
	chunker.SetCppMlClient(cppMlWorker)
	return &DocumentProcessingServer{
		chunker:      chunker,
		cppMlWorker:  cppMlWorker,
		netMlWorker:  netMlWorker,
		store:        store,
	}
}

func (s *DocumentProcessingServer) ProcessDocument(ctx context.Context, req *pb.ProcessDocRequest) (*pb.ProcessDocResponse, error) {
	if req == nil || req.DocumentUri == "" || req.Identity == nil {
		panic(fmt.Errorf("action failed for job ProcessDocument: missing required document_uri or identity"))
	}

	isDev := os.Getenv("APP_ENV") == "development"
	timeout := 5 * time.Minute
	if isDev {
		timeout = 60 * time.Minute
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	fileBytes, err := s.store.FetchFile(ctx, req.DocumentUri, req.Identity)
	if err != nil {
		return nil, fmt.Errorf("failed to download document from storage gateway: %w", err)
	}

	var g errgroup.Group
	var cppSuccess, netSuccess bool
	var cppDuration, netDuration time.Duration
	var cppErr, netErr error

	g.Go(func() error {
		start := time.Now()
		resp, err := s.cppMlWorker.RedactDocument(ctx, &pb.RedactDocumentRequest{
			DocumentUri: req.DocumentUri,
			Identity:    req.Identity,
		})
		cppDuration = time.Since(start)
		if err != nil {
			cppErr = err
			return nil
		}
		cppSuccess = resp.GetSuccess()
		if !cppSuccess {
			cppErr = fmt.Errorf("cpp worker returned failure: %s", resp.GetErrorMessage())
		}
		return nil
	})

	g.Go(func() error {
		start := time.Now()
		resp, err := s.netMlWorker.RedactPii(ctx, &pb.PiiRedactionRequest{
			Identity:       req.Identity,
			TargetPiiTypes: req.TargetPiiTypes,
			RawFileBytes:   fileBytes,
		})
		netDuration = time.Since(start)
		if err != nil {
			netErr = err
			return nil
		}
		netSuccess = resp.GetSuccess()
		if !netSuccess {
			netErr = fmt.Errorf("net worker returned failure: %s", resp.GetErrorMessage())
		}
		return nil
	})

	_ = g.Wait()

	projectRoot := os.Getenv("PROJECT_ROOT")
	if projectRoot == "" {
		panic(fmt.Errorf("action failed for job ProcessDocument: missing required PROJECT_ROOT environment variable for benchmark logging"))
	}
	logFile := filepath.Join(projectRoot, "logs", "benchmark.log")
	
	if err := os.MkdirAll(filepath.Dir(logFile), 0755); err != nil {
		panic(fmt.Errorf("action failed for job ProcessDocument: failed to create logs directory: %w", err))
	}

	f, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(fmt.Errorf("action failed for job ProcessDocument: failed to write to benchmark log file: %w", err))
	}
	
	logLine := fmt.Sprintf("[%s] JobId: %s, AppId: %s, CppSuccess: %v, CppTime: %s, NetSuccess: %v, NetTime: %s\n",
		time.Now().Format(time.RFC3339),
		req.Identity.GetJobId(),
		req.Identity.GetAppId(),
		cppSuccess,
		cppDuration,
		netSuccess,
		netDuration,
	)
	f.WriteString(logLine)
	f.Close()

	success := cppSuccess || netSuccess
	var errMsgs []string
	if cppErr != nil {
		errMsgs = append(errMsgs, cppErr.Error())
	}
	if netErr != nil {
		errMsgs = append(errMsgs, netErr.Error())
	}

	errMsg := ""
	if len(errMsgs) > 0 {
		errMsg = fmt.Sprintf("errors: %v", errMsgs)
	}

	return &pb.ProcessDocResponse{
		Success:      success,
		ErrorMessage: errMsg,
	}, nil
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
		if req.GetOriginalRequest() != nil && req.GetOriginalRequest().GetTaskSubject() == pb.WorkerSubject_WORKER_SUBJECT_UNSPECIFIED {
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
