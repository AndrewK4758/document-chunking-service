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

func (s *NatsServer) sendErrorReply(ctx context.Context, replySubj string, code string, msgStr string, t, a, j string) {
	if replySubj == "" {
		return
	}
	resp := &pb.PerformActionResponse{
		Success: false,
		ErrorContract: &pb.ErrorContract{
			Code:    code,
			Source:  "document_processing_service",
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
		TenantID:  t,
		AppID:     a,
		JobID:     j,
		SizeBytes: len(respBytes),
		Error:     pubErr,
	})
}

func (s *NatsServer) ackMessage(msg jetstream.Msg, startTime time.Time, t, a, j string, attempt uint64) {
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
		TenantID:   t,
		AppID:      a,
		JobID:      j,
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

func (s *NatsServer) publishReply(ctx context.Context, msg jetstream.Msg, replySubj string, resp *pb.PerformActionResponse, startTime time.Time, t, a, j string, attempt uint64) error {
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
			TenantID:   t,
			AppID:      a,
			JobID:      j,
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
		return pubErr
	}
	return nil
}

func (s *NatsServer) runAckWindowExtender(ctx context.Context, msg jetstream.Msg, done chan struct{}, t, a, j string, attempt uint64) {
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
}

func (s *NatsServer) handleProcessingError(ctx context.Context, msg jetstream.Msg, replySubj string, err error, startTime time.Time, t, a, j string, attempt uint64) {
	logger.Info("DocProcessor", "Error processing document for JobID %s: %v", j, err)
	resp := &pb.PerformActionResponse{
		Success:          false,
		IsTransientError: true,
		ErrorContract: &pb.ErrorContract{
			Code:    "DOCUMENT_PROCESSING_FAILED",
			Source:  "document_processing_service",
			Message: err.Error(),
		},
	}
	pubErr := s.publishReply(ctx, msg, replySubj, resp, startTime, t, a, j, attempt)
	if pubErr != nil {
		return
	}
	s.ackMessage(msg, startTime, t, a, j, attempt)
}

func (s *NatsServer) processMessage(ctx context.Context, msg jetstream.Msg) {
	startTime := time.Now()
	meta, _ := msg.Metadata()
	var attempt uint64 = 1
	if meta != nil {
		attempt = meta.NumDelivered
	}

	logger.Info("DocProcessor", "Received document processing request on subject: %s", msg.Subject())

	var req pb.ExecuteTaskRequest
	err := proto.Unmarshal(msg.Data(), &req)
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
	t := identity.GetTenantId()
	a := identity.GetAppId()
	j := identity.GetJobId()

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
	headers := msg.Headers()
	if headers != nil {
		replySubj = headers.Get("Reply-Subject")
	}

	// Extend ack window while processing
	done := make(chan struct{})
	defer close(done)
	go s.runAckWindowExtender(ctx, msg, done, t, a, j, attempt)

	var payload map[string]interface{}
	if req.Payload != nil {
		payload = req.Payload.AsMap()
	} else {
		payload = make(map[string]interface{})
	}

	reqJson, err := protojson.Marshal(&req)
	if err == nil {
		logger.Info("DocProcessor", "DEQUEUED Input Payload: %s", string(reqJson))
	} else {
		payloadBytes, _ := json.Marshal(payload)
		logger.Info("DocProcessor", "DEQUEUED Input Payload: %s", string(payloadBytes))
	}

	parser := workflowparser.NewTaskRequestParser(&req)
	parser.MustValidateRootPayloadStrict(j)
	funcState := parser.MustGetFunctionalState(j)
	
	docUriVal, hasDocUri := funcState["document_uri"]
	if !hasDocUri || docUriVal == nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: document_uri missing from global_state.functional"))
	}
	docUri, ok := docUriVal.(string)
	if !ok || docUri == "" {
		logger.Info("DocProcessor", "document_uri is empty")
		s.sendErrorReply(ctx, replySubj, "MISSING_DOCUMENT_URI", "[Step: WORKER_DEQUEUE, URI: unknown] document_uri must be provided", t, a, j)
		s.ackMessage(msg, startTime, t, a, j, attempt)
		return
	}

	chunkReq := &pb.ChunkDocumentRequest{
		Identity:          req.GetIdentity(),
		StepId:            req.GetStepId(),
		OriginalRequest:   &req,
	}

	docs, _, err := s.chunker.ProcessZipFile(ctx, chunkReq)

	if err != nil {
		s.handleProcessingError(ctx, msg, replySubj, err, startTime, t, a, j, attempt)
		return
	}

	logger.Info("DocProcessor", "Successfully processed document for JobID %s, extracted %d chunks", j, len(docs))
	structResult, marshalErr := buildDocsStructResult(docs)
	if marshalErr != nil {
		logger.Info("DocProcessor", "Failed to marshal docs to JSON: %v", marshalErr)
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
			Error:      marshalErr,
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
			outJson, _ := json.Marshal(resp.ActionResult.AsMap())
			logger.Info("DocProcessor", "DISPATCHING ActionResult Output: %s", string(outJson))
		}
	} else {
		logger.Info("DocProcessor", "DISPATCHING ActionResult Output: {}")
	}

	pubErr := s.publishReply(ctx, msg, replySubj, resp, startTime, t, a, j, attempt)
	if pubErr != nil {
		return
	}

	s.ackMessage(msg, startTime, t, a, j, attempt)
}
