package server

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"google.golang.org/protobuf/proto"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/logger"
)

type NatsServer struct {
	nc      *nats.Conn
	js      jetstream.JetStream
	chunker *processor.ChunkingProcessor
}

func NewNatsServer(nc *nats.Conn, js jetstream.JetStream, aiClient pb.DocumentProcessorClient, store storage.Manager) *NatsServer {
	return &NatsServer{
		nc:      nc,
		js:      js,
		chunker: processor.NewChunkingProcessor(aiClient, store),
	}
}

func (s *NatsServer) Start(ctx context.Context) error {
	streamName := "WORKERS"

	var cons jetstream.Consumer
	var err error
	for i := 0; i < 30; i++ {
		cons, err = s.js.CreateOrUpdateConsumer(ctx, streamName, jetstream.ConsumerConfig{
			Durable:        "doc_processor_workers",
			AckPolicy:      jetstream.AckExplicitPolicy,
			FilterSubjects: []string{"worker.doc.>"},
			MaxAckPending:  10,
			AckWait:        5 * time.Minute,
		})
		if err == nil {
			break
		}
		logger.Info("DocProcessor", "Waiting for stream %s to be created... (%v)", streamName, err)
		time.Sleep(2 * time.Second)
	}
	if err != nil {
		return fmt.Errorf("failed to create JetStream consumer after retries: %w", err)
	}

	logger.Info("DocProcessor", "Starting Document Processing NATS consumer...")
	cc, err := cons.Consume(func(msg jetstream.Msg) {
		s.processMessage(ctx, msg)
	})
	if err != nil {
		return fmt.Errorf("failed to start consuming: %w", err)
	}

	<-ctx.Done()
	cc.Stop()
	return nil
}

func (s *NatsServer) processMessage(ctx context.Context, msg jetstream.Msg) {
	logger.Info("DocProcessor", "Received document processing request on subject: %s", msg.Subject())

	var req pb.ExecuteWorkflowNodeRequest
	if err := proto.Unmarshal(msg.Data(), &req); err != nil {
		logger.Info("DocProcessor", "Failed to unmarshal ExecuteWorkflowNodeRequest: %v", err)
		msg.NakWithDelay(5 * time.Second)
		return
	}

	var replySubj string
	if headers := msg.Headers(); headers != nil {
		replySubj = headers.Get("Reply-Subject")
	}

	// Extend ack window while processing
	done := make(chan struct{})
	defer close(done)
	go func() {
		ticker := time.NewTicker(2 * time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				msg.InProgress()
			case <-done:
				return
			case <-ctx.Done():
				return
			}
		}
	}()

	var docUri string
	if req.GlobalState != nil && req.GlobalState.Fields != nil {
		if uriVal, ok := req.GlobalState.Fields["DocumentStorageUri"]; ok && uriVal.GetStringValue() != "" {
			docUri = uriVal.GetStringValue()
		} else if pathVal, ok := req.GlobalState.Fields["DocumentPath"]; ok && pathVal.GetStringValue() != "" {
			docUri = pathVal.GetStringValue()
		}
	}

	if docUri == "" {
		logger.Info("DocProcessor", "No DocumentStorageUri or DocumentPath found in global state")
		msg.NakWithDelay(5 * time.Second)
		return
	}

	chunkReq := &pb.ChunkDocumentRequest{
		Identity: req.Identity,
		StepId:   req.StepId,
		DocumentChunking: &pb.DocumentChunkingContract{
			DocumentStorageUri: docUri,
		},
		OriginalRequest: &req,
	}

	docs, _, err := s.chunker.ProcessZipFile(ctx, chunkReq)
	
	resp := &pb.PerformActionResponse{
		Success: err == nil,
	}

	if err != nil {
		logger.Info("DocProcessor", "Error processing zip file for JobID %s: %v", req.Identity.GetJobId(), err)
		resp.IsTransientError = true
		resp.ErrorContract = &pb.ErrorContract{
			Code:    "DOCUMENT_PROCESSING_FAILED",
			Source:  "document_processing_service",
			Message: err.Error(),
		}
	} else {
		logger.Info("DocProcessor", "Successfully processed zip file for JobID %s, extracted %d documents", req.Identity.GetJobId(), len(docs))
		jsonBytes, err := json.Marshal(docs)
		if err != nil {
			logger.Info("DocProcessor", "Failed to marshal docs to JSON: %v", err)
			msg.NakWithDelay(5 * time.Second)
			return
		}
		resp.ActionResultJson = string(jsonBytes)
	}

	if replySubj != "" {
		respBytes, err := proto.Marshal(resp)
		if err != nil {
			logger.Info("DocProcessor", "Failed to marshal response: %v", err)
			msg.NakWithDelay(5 * time.Second)
			return
		}

		if _, err := s.js.Publish(ctx, replySubj, respBytes); err != nil {
			logger.Info("DocProcessor", "Failed to publish reply to NATS %s: %v", replySubj, err)
			msg.NakWithDelay(5 * time.Second)
			return
		}
	}

	if err := msg.Ack(); err != nil {
		logger.Info("DocProcessor", "Failed to ack msg: %v", err)
	} else {
		logger.Info("DocProcessor", "Successfully acknowledged processing request and replied to %s", replySubj)
	}
}
