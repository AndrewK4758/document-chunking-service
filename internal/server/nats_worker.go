package server

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/AndrewK4758/shared_utils/envutil"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"google.golang.org/protobuf/proto"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/AndrewK4758/shared_utils/natsutil"
	"github.com/AndrewK4758/shared_utils/workflowparser"
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
	retryCount := envutil.MustGetInt("NATS_CONSUMER_RETRY_COUNT")
	retrySleepSec := envutil.MustGetInt("NATS_CONSUMER_RETRY_SLEEP_SEC")
	
	for range retryCount {
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
		time.Sleep(time.Duration(retrySleepSec) * time.Second)
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
	startTime := time.Now()
	meta, _ := msg.Metadata()
	var attempt uint64 = 1
	if meta != nil {
		attempt = meta.NumDelivered
	}

	logger.Info("DocProcessor", "Received document processing request on subject: %s", msg.Subject())

	var req pb.ExecuteWorkflowNodeRequest
	if err := proto.Unmarshal(msg.Data(), &req); err != nil {
		logger.Info("DocProcessor", "Failed to unmarshal ExecuteWorkflowNodeRequest: %v", err)
		natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
			Component:  "DocProcessor",
			Event:      "TERMINATED",
			Stream:     "WORKERS",
			Subject:    msg.Subject(),
			Consumer:   "doc_processor_workers",
			TenantID:   "default",
			AppID:      "default",
			JobID:      "default",
			Attempt:    attempt,
			DurationMs: time.Since(startTime).Milliseconds(),
			Error:      err,
		})
		msg.Term()
		return
	}

	parser := workflowparser.NewNodeRequestParser(&req)
	t, a, j := parser.ExtractIdentity()

	natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
		Component: "DocProcessor",
		Event:     "DEQUEUED",
		Stream:    "WORKERS",
		Subject:   msg.Subject(),
		Consumer:  "doc_processor_workers",
		TenantID:  t,
		AppID:     a,
		JobID:     j,
		SizeBytes: len(msg.Data()),
		Attempt:   attempt,
	})

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
				if err := msg.InProgress(); err == nil {
					natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
						Component: "DocProcessor",
						Event:     "IN_PROGRESS",
						Stream:    "WORKERS",
						Subject:   msg.Subject(),
						Consumer:  "doc_processor_workers",
						TenantID:  t,
						AppID:     a,
						JobID:     j,
						Attempt:   attempt,
					})
				}
			case <-done:
				return
			case <-ctx.Done():
				return
			}
		}
	}()

	docUri := parser.GetDocumentUri()

	if docUri == "" {
		logger.Info("DocProcessor", "StorageContract not provided or DocumentUri is empty")

		if replySubj != "" {
			resp := &pb.PerformActionResponse{
				Success: false,
				ErrorContract: &pb.ErrorContract{
					Code:    "MISSING_DOCUMENT_URI",
					Source:  "document_processing_service",
					Message: "StorageContract.DocumentUri must be provided",
				},
			}
			respBytes, err := proto.Marshal(resp)
			if err != nil {
				logger.Warn("DocProcessor", "Failed to marshal error response: %v", err)
			} else {
				_, pubErr := s.js.Publish(ctx, replySubj, respBytes)
				natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
					Component: "DocProcessor",
					Event:     "ENQUEUED",
					Stream:    "ORCHESTRATOR_RESPONSES",
					Subject:   replySubj,
					TenantID:  t,
					AppID:     a,
					JobID:     j,
					SizeBytes: len(respBytes),
					Error:     pubErr,
				})
			}
		}

		durationMs := time.Since(startTime).Milliseconds()
		if err := msg.Ack(); err == nil {
			natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
				Component:  "DocProcessor",
				Event:      "ACKNOWLEDGED",
				Stream:     "WORKERS",
				Subject:    msg.Subject(),
				Consumer:   "doc_processor_workers",
				TenantID:   t,
				AppID:      a,
				JobID:      j,
				Attempt:    attempt,
				DurationMs: durationMs,
			})
		}
		return
	}

	var resp *pb.PerformActionResponse

	chunkReq := &pb.ChunkDocumentRequest{
		Identity:          parser.GetIdentity(),
		StepId:            parser.GetStepId(),
		DocumentProcessor: parser.GetDocumentProcessorContract(),
		OriginalRequest:   &req,
	}

	docs, _, err := s.chunker.ProcessZipFile(ctx, chunkReq)

	resp = &pb.PerformActionResponse{
		Success: err == nil,
	}

	if err != nil {
		logger.Info("DocProcessor", "Error processing document for JobID %s: %v", j, err)
		resp.IsTransientError = true
		resp.ErrorContract = &pb.ErrorContract{
			Code:    "DOCUMENT_PROCESSING_FAILED",
			Source:  "document_processing_service",
			Message: err.Error(),
		}
	} else {
		logger.Info("DocProcessor", "Successfully processed document for JobID %s, extracted %d chunks", j, len(docs))
		jsonBytes, err := json.Marshal(docs)
		if err != nil {
			logger.Info("DocProcessor", "Failed to marshal docs to JSON: %v", err)
			nackTimeoutJson := envutil.MustGetInt("NACK_JSON_TIMEOUT")
			natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
				Component:  "DocProcessor",
				Event:      "NACKED",
				Stream:     "WORKERS",
				Subject:    msg.Subject(),
				Consumer:   "doc_processor_workers",
				TenantID:   t,
				AppID:      a,
				JobID:      j,
				Attempt:    attempt,
				DurationMs: time.Since(startTime).Milliseconds(),
				Error:      err,
			})
			msg.NakWithDelay(time.Duration(nackTimeoutJson) * time.Second)
			return
		}
		resp.ActionResultJson = string(jsonBytes)
	}

	if replySubj != "" {
		respBytes, err := proto.Marshal(resp)
		if err != nil {
			logger.Info("DocProcessor", "Failed to marshal response: %v", err)
			natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
				Component:  "DocProcessor",
				Event:      "NACKED",
				Stream:     "WORKERS",
				Subject:    msg.Subject(),
				Consumer:   "doc_processor_workers",
				TenantID:   t,
				AppID:      a,
				JobID:      j,
				Attempt:    attempt,
				DurationMs: time.Since(startTime).Milliseconds(),
				Error:      err,
			})
			msg.NakWithDelay(5 * time.Second)
			return
		}

		_, pubErr := s.js.Publish(ctx, replySubj, respBytes)
		natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
			Component: "DocProcessor",
			Event:     "ENQUEUED",
			Stream:    "ORCHESTRATOR_RESPONSES",
			Subject:   replySubj,
			TenantID:  t,
			AppID:     a,
			JobID:     j,
			SizeBytes: len(respBytes),
			Error:     pubErr,
		})
		if pubErr != nil {
			logger.Info("DocProcessor", "Failed to publish reply to NATS %s: %v", replySubj, pubErr)
			natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
				Component:  "DocProcessor",
				Event:      "NACKED",
				Stream:     "WORKERS",
				Subject:    msg.Subject(),
				Consumer:   "doc_processor_workers",
				TenantID:   t,
				AppID:      a,
				JobID:      j,
				Attempt:    attempt,
				DurationMs: time.Since(startTime).Milliseconds(),
				Error:      pubErr,
			})
			msg.NakWithDelay(5 * time.Second)
			return
		}
	}

	durationMs := time.Since(startTime).Milliseconds()
	if err := msg.Ack(); err != nil {
		logger.Info("DocProcessor", "Failed to ack msg: %v", err)
	} else {
		logger.Info("DocProcessor", "Successfully acknowledged processing request and replied to %s", replySubj)
		natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
			Component:  "DocProcessor",
			Event:      "ACKNOWLEDGED",
			Stream:     "WORKERS",
			Subject:    msg.Subject(),
			Consumer:   "doc_processor_workers",
			TenantID:   t,
			AppID:      a,
			JobID:      j,
			Attempt:    attempt,
			DurationMs: durationMs,
		})
	}
}
