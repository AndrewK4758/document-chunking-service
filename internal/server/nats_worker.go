package server

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/AndrewK4758/shared_utils/envutil"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/identityparser"
	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/AndrewK4758/shared_utils/natsutil"
	"github.com/AndrewK4758/shared_utils/workflowparser"
)

const (
	ErrorCode   = "DOCUMENT_PROCESSING_FAILED"
	ErrorSource = "document_processing_service"
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
	streamName := envutil.GetStringOrDefault("NATS_WORKER_STREAM_NAME", "WORKERS")
	consumerName := envutil.GetStringOrDefault("NATS_WORKER_CONSUMER_NAME", "doc_processor_workers")

	var cons jetstream.Consumer
	var err error
	retryCount := envutil.MustGetInt("NATS_CONSUMER_RETRY_COUNT")
	retrySleepSec := envutil.MustGetInt("NATS_CONSUMER_RETRY_SLEEP_SEC")

	for range retryCount {
		cons, err = s.js.CreateOrUpdateConsumer(ctx, streamName, jetstream.ConsumerConfig{
			Durable:        consumerName,
			AckPolicy:      jetstream.AckExplicitPolicy,
			FilterSubjects: []string{"worker.doc.>"}, // Or from env if necessary, assuming only durable and stream are hard violations mentioned
			MaxAckPending:  envutil.MustGetInt("MAX_ACK_PENDING"),
			AckWait:        envutil.MustGetDuration("WORKER_ACK_WAIT"),
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

func (s *NatsServer) sendErrorReply(ctx context.Context, replySubj string, code string, msgStr string, identity *pb.InfrastructureIdentity) {
	if replySubj == "" {
		return
	}
	resp := &pb.PerformActionResponse{
		Success: false,
		ErrorContract: &pb.ErrorContract{
			Code:    code,
			Source:  ErrorSource,
			Message: msgStr,
		},
	}
	respBytes, err := proto.Marshal(resp)
	if err != nil {
		logger.Warn("DocProcessor", "Failed to marshal error response: %v", err)
		return
	}
	_, pubErr := s.js.Publish(ctx, replySubj, respBytes)
	natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
		Component: "DocProcessor",
		Event:     "ENQUEUED",
		Stream:    "ORCHESTRATOR_RESPONSES",
		Subject:   replySubj,
		TenantID:  identity.GetTenantId(),
		AppID:     identity.GetAppId(),
		JobID:     identity.GetJobId(),
		SizeBytes: len(respBytes),
		Error:     pubErr,
	})
}

func (s *NatsServer) ackMessage(msg jetstream.Msg, startTime time.Time, identity *pb.InfrastructureIdentity, attempt uint64) {
	err := msg.Ack()
	if err != nil {
		logger.Info("DocProcessor", "Failed to ack msg: %v", err)
		return
	}
	natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
		Component:  "DocProcessor",
		Event:      "ACKNOWLEDGED",
		Stream:     "WORKERS",
		Subject:    msg.Subject(),
		Consumer:   "doc_processor_workers",
		TenantID:   identity.GetTenantId(),
		AppID:      identity.GetAppId(),
		JobID:      identity.GetJobId(),
		Attempt:    attempt,
		DurationMs: time.Since(startTime).Milliseconds(),
	})
}

func buildDocsStructResult(docs []*pb.ProcessedDocument) (*structpb.Struct, error) {
	jsonBytes, err := json.Marshal(docs)
	if err != nil {
		return nil, err
	}
	var structResult structpb.Struct
	jsonString := fmt.Sprintf(`{"documents": %s}`, string(jsonBytes))
	err = protojson.Unmarshal([]byte(jsonString), &structResult)
	if err != nil {
		logger.Warn("DocProcessor", "Failed to unmarshal docs to structpb.Struct: %v", err)
		return nil, nil
	}
	return &structResult, nil
}

func (s *NatsServer) publishReply(ctx context.Context, msg jetstream.Msg, replySubj string, resp *pb.PerformActionResponse, startTime time.Time, identity *pb.InfrastructureIdentity, attempt uint64) error {
	if replySubj == "" {
		return nil
	}
	respBytes, err := proto.Marshal(resp)
	if err != nil {
		logger.Info("DocProcessor", "Failed to marshal response: %v", err)
		natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
			Component:  "DocProcessor",
			Event:      "NACKED",
			Stream:     "WORKERS",
			Subject:    msg.Subject(),
			Consumer:   "doc_processor_workers",
			TenantID:   identity.GetTenantId(),
			AppID:      identity.GetAppId(),
			JobID:      identity.GetJobId(),
			Attempt:    attempt,
			DurationMs: time.Since(startTime).Milliseconds(),
			Error:      err,
		})
		msg.NakWithDelay(5 * time.Second)
		return err
	}

	_, pubErr := s.js.Publish(ctx, replySubj, respBytes)
	natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
		Component: "DocProcessor",
		Event:     "ENQUEUED",
		Stream:    "ORCHESTRATOR_RESPONSES",
		Subject:   replySubj,
		TenantID:  identity.GetTenantId(),
		AppID:     identity.GetAppId(),
		JobID:     identity.GetJobId(),
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
			TenantID:   identity.GetTenantId(),
			AppID:      identity.GetAppId(),
			JobID:      identity.GetJobId(),
			Attempt:    attempt,
			DurationMs: time.Since(startTime).Milliseconds(),
			Error:      pubErr,
		})
		msg.NakWithDelay(5 * time.Second)
		return pubErr
	}
	return nil
}

func (s *NatsServer) runAckWindowExtender(ctx context.Context, msg jetstream.Msg, done chan struct{}, identity *pb.InfrastructureIdentity, attempt uint64) {
	ticker := time.NewTicker(2 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			err := msg.InProgress()
			if err == nil {
				natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
					Component: "DocProcessor",
					Event:     "IN_PROGRESS",
					Stream:    "WORKERS",
					Subject:   msg.Subject(),
					Consumer:  "doc_processor_workers",
					TenantID:  identity.GetTenantId(),
					AppID:     identity.GetAppId(),
					JobID:     identity.GetJobId(),
					Attempt:   attempt,
				})
			}
		case <-done:
			return
		case <-ctx.Done():
			return
		}
	}
}

func (s *NatsServer) handleProcessingError(ctx context.Context, msg jetstream.Msg, replySubj string, err error, startTime time.Time, identity *pb.InfrastructureIdentity, attempt uint64) {
	logger.Info("DocProcessor", "Error processing document for JobID %s: %v", identity.GetJobId(), err)
	resp := &pb.PerformActionResponse{
		Success:          false,
		IsTransientError: true,
		ErrorContract: &pb.ErrorContract{
			Code:    ErrorCode,
			Source:  ErrorSource,
			Message: err.Error(),
		},
	}
	pubErr := s.publishReply(ctx, msg, replySubj, resp, startTime, identity, attempt)
	if pubErr != nil {
		return
	}
	s.ackMessage(msg, startTime, identity, attempt)
}

func (s *NatsServer) processMessage(ctx context.Context, msg jetstream.Msg) {
	startTime := time.Now()
	var attempt uint64 = 1
	meta, err := msg.Metadata()
	if err == nil && meta != nil {
		attempt = meta.NumDelivered
	}

	logger.Info("DocProcessor", "Received document processing request on subject: %s", msg.Subject())

	var req pb.ExecuteTaskRequest
	err = proto.Unmarshal(msg.Data(), &req)
	if err != nil {
		logger.Info("DocProcessor", "Failed to unmarshal ExecuteTaskRequest: %v", err)
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

	identityparser.NewIdentityParser(req.GetIdentity()).MustValidate(true, "DocProcessorWorker")

	if req.GetTaskSubject() == "" {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: missing required TaskSubject"))
	}

	identity := req.GetIdentity()
	if identity == nil {
		identity = &pb.InfrastructureIdentity{TenantId: "unknown", AppId: "unknown", JobId: "unknown"}
	}

	natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
		Component: "DocProcessor",
		Event:     "DEQUEUED",
		Stream:    "WORKERS",
		Subject:   msg.Subject(),
		Consumer:  "doc_processor_workers",
		TenantID:  identity.GetTenantId(),
		AppID:     identity.GetAppId(),
		JobID:     identity.GetJobId(),
		SizeBytes: len(msg.Data()),
		Attempt:   attempt,
	})

	var replySubj string
	headers := msg.Headers()
	if headers != nil {
		replySubj = headers.Get("Reply-Subject")
	}

	// Extend ack window while processing
	done := make(chan struct{})
	defer close(done)
	go s.runAckWindowExtender(ctx, msg, done, identity, attempt)

	if req.Payload == nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: missing payload"))
	}

	reqJson, err := protojson.Marshal(&req)
	if err == nil {
		logger.Info("DocProcessor", "DEQUEUED Input Payload: %s", string(reqJson))
	} else {
		logger.Info("DocProcessor", "DEQUEUED Input Payload: failed to marshal proto")
	}

	parser := workflowparser.NewTaskRequestParser(&req)
	parser.MustValidateRootPayloadStrict(identity.GetJobId())
	executeTaskPayload := parser.MustGetExecuteTaskPayload()
	if executeTaskPayload.Storage == nil || executeTaskPayload.Storage.DocumentUri == "" {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: document_uri missing from storage"))
	}
	var structResult *structpb.Struct
	var processErr error

	if req.GetTaskSubject() == "worker.doc.inspect" {
		structResult, processErr = s.chunker.InspectDocument(ctx, &req, executeTaskPayload.Storage.DocumentUri, func(msg string, level string) {
			logger.Info("DocProcessor", "[%s] %s", level, msg)
		})
	} else {
		chunkReq := &pb.ChunkDocumentRequest{
			Identity:        req.GetIdentity(),
			StepId:          req.GetStepId(),
			OriginalRequest: &req,
		}
		docs, _, err := s.chunker.ProcessZipFile(ctx, chunkReq)
		processErr = err
		if err == nil {
			logger.Info("DocProcessor", "Successfully processed document for JobID %s, extracted %d chunks", identity.GetJobId(), len(docs))
			structResult, processErr = buildDocsStructResult(docs)
		}
	}

	if processErr != nil {
		s.handleProcessingError(ctx, msg, replySubj, processErr, startTime, identity, attempt)
		return
	}

	if structResult == nil {
		nackTimeoutJson := envutil.MustGetInt("NACK_JSON_TIMEOUT")
		natsutil.LogJetStreamEvent(natsutil.JetStreamLogEvent{
			Component:  "DocProcessor",
			Event:      "NACKED",
			Stream:     "WORKERS",
			Subject:    msg.Subject(),
			Consumer:   "doc_processor_workers",
			TenantID:   identity.GetTenantId(),
			AppID:      identity.GetAppId(),
			JobID:      identity.GetJobId(),
			Attempt:    attempt,
			DurationMs: time.Since(startTime).Milliseconds(),
			Error:      fmt.Errorf("structResult is nil"),
		})
		msg.NakWithDelay(time.Duration(nackTimeoutJson) * time.Second)
		return
	}

	resp := &pb.PerformActionResponse{
		Success:      true,
		ActionResult: structResult,
	}

	if resp.ActionResult != nil {
		outBytes, err := protojson.Marshal(resp.ActionResult)
		if err == nil {
			logger.Info("DocProcessor", "DISPATCHING ActionResult Output: %s", string(outBytes))
		} else {
			outJson, jsonErr := json.Marshal(resp.ActionResult.AsMap())
			if jsonErr == nil {
				logger.Info("DocProcessor", "DISPATCHING ActionResult Output: %s", string(outJson))
			} else {
				logger.Info("DocProcessor", "DISPATCHING ActionResult Output: <failed to marshal>")
			}
		}
	} else {
		logger.Info("DocProcessor", "DISPATCHING ActionResult Output: {}")
	}

	pubErr := s.publishReply(ctx, msg, replySubj, resp, startTime, identity, attempt)
	if pubErr != nil {
		return
	}

	s.ackMessage(msg, startTime, identity, attempt)
}
