package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/AndrewK4758/shared_utils/envutil"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	"document_processing_service/internal/db"
	"document_processing_service/internal/processor"
	"document_processing_service/internal/storage"

	"gorm.io/gorm"

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
	nc             *nats.Conn
	js             jetstream.JetStream
	chunker        *processor.ChunkingProcessor
	cppMlClient    pb.CppMlWorkerServiceClient
	dotnetMlClient pb.MlWorkerServiceClient
	db             *gorm.DB
	store          storage.Manager
}

func NewNatsServer(nc *nats.Conn, js jetstream.JetStream, aiClient pb.DocumentProcessorClient, store storage.Manager, cppMlClient pb.CppMlWorkerServiceClient, dotnetMlClient pb.MlWorkerServiceClient, db *gorm.DB) *NatsServer {
	chunker := processor.NewChunkingProcessor(aiClient, store)
	chunker.SetCppMlClient(cppMlClient)
	return &NatsServer{
		nc:             nc,
		js:             js,
		chunker:        chunker,
		cppMlClient:    cppMlClient,
		dotnetMlClient: dotnetMlClient,
		db:             db,
		store:          store,
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
			BackOff: []time.Duration{
				5 * time.Second,
				30 * time.Second,
				1 * time.Minute,
				5 * time.Minute,
			},
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
	workerPoolSize := envutil.GetIntOrDefault("WORKER_POOL_SIZE", 10)
	sem := make(chan struct{}, workerPoolSize)
	cc, err := cons.Consume(func(msg jetstream.Msg) {
		go s.HandleMessageWithRecovery(ctx, msg, sem)
	})
	if err != nil {
		return fmt.Errorf("failed to start consuming: %w", err)
	}

	<-ctx.Done()
	cc.Stop()
	return nil
}

func (s *NatsServer) HandleMessageWithRecovery(ctx context.Context, msg jetstream.Msg, sem chan struct{}) {
	sem <- struct{}{}
	defer func() {
		<-sem
		if r := recover(); r != nil {
			var t, a, j = "unknown", "unknown", "unknown"
			var req pb.ExecuteTaskRequest
			if err := proto.Unmarshal(msg.Data(), &req); err == nil && req.Identity != nil {
				t = req.Identity.TenantId
				a = req.Identity.AppId
				j = req.Identity.JobId
			}
			logger.Error("DocProcessor", "Panic recovered in NATS consumer goroutine for TenantID: %s, AppID: %s, JobID: %s. Error: %v", t, a, j, r)
			msg.NakWithDelay(5 * time.Second)
		}
	}()
	s.processMessage(ctx, msg)
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
	docList := make([]interface{}, len(docs))
	m := protojson.MarshalOptions{
		UseProtoNames:   true,
		EmitUnpopulated: true,
	}
	for i, doc := range docs {
		docBytes, err := m.Marshal(doc)
		if err != nil {
			return nil, err
		}
		var docMap map[string]interface{}
		if err := json.Unmarshal(docBytes, &docMap); err != nil {
			return nil, err
		}
		docList[i] = docMap
	}
	
	resultMap := map[string]interface{}{
		"documents": docList,
	}
	return structpb.NewStruct(resultMap)
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

	if strings.TrimSpace(req.GetTaskSubject()) == "" {
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

	logger.Info("DocProcessor", "DEQUEUED Input Payload for TaskSubject: %v", req.GetTaskSubject())

	parser := workflowparser.NewTaskRequestParser(&req)
	parser.MustValidateRootPayloadStrict(identity.GetJobId())
	executeTaskPayload := parser.MustGetExecuteTaskPayload()
	if executeTaskPayload.Storage == nil || executeTaskPayload.Storage.DocumentUri == "" {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: document_uri missing from storage"))
	}
	var structResult *structpb.Struct
	var processErr error

	switch req.GetTaskSubject() {
	case "worker.doc.inspect":
		structResult, processErr = s.chunker.InspectDocument(ctx, &req, executeTaskPayload.Storage.DocumentUri, func(msg string, level string) {
			logger.Info("DocProcessor", "[%s] %s", level, msg)
		})
	case "worker.doc.processing.cpp_ml":
		structResult, processErr = s.handleCppMLRequest(ctx, &req, executeTaskPayload, identity)
	case "worker.doc.processing.dotnet_ml":
		structResult, processErr = s.handleDotnetMLRequest(ctx, &req, executeTaskPayload, identity)
	case "worker.doc.processing.dual_ml":
		structResult, processErr = s.handleDualMLRequest(ctx, &req, executeTaskPayload, identity)
	default:
		// Autonomous Routing Logic: Evaluate complexity
		pageCount := int32(0)
		if executeTaskPayload.Storage != nil {
			pageCount = executeTaskPayload.Storage.PageCount
		}

		if true { // TEMPORARY: Force all to ML service for testing
			logger.Info("DocProcessor", "Document is complex (PageCount: %d). Routing to CPP ML Worker (JobID: %s)", pageCount, identity.GetJobId())
			structResult, processErr = s.handleCppMLRequest(ctx, &req, executeTaskPayload, identity)
			break
		}

		logger.Info("DocProcessor", "Document is simple (PageCount: %d). Forwarding to AI Inference (JobID: %s)", pageCount, identity.GetJobId())
		
		req.TaskSubject = "worker.ai.inference"
		forwardData, marshalErr := proto.Marshal(&req)
		if marshalErr != nil {
			processErr = fmt.Errorf("failed to marshal forwarded request: %w", marshalErr)
			break
		}

		headers := msg.Headers()
		if headers == nil {
			headers = nats.Header{}
		}
		headers.Set("Reply-Subject", replySubj)
		
		pubMsg := &nats.Msg{
			Subject: req.TaskSubject,
			Header:  headers,
			Data:    forwardData,
		}
		
		if _, pubErr := s.js.PublishMsg(ctx, pubMsg); pubErr != nil {
			processErr = fmt.Errorf("failed to forward to AI Inference: %w", pubErr)
			break
		}
		
		// Successfully forwarded, ack and exit since the AI worker will reply to the orchestrator.
		s.ackMessage(msg, startTime, identity, attempt)
		return
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
		keys := make([]string, 0, len(resp.ActionResult.Fields))
		for k := range resp.ActionResult.Fields {
			keys = append(keys, k)
		}
		logger.Info("DocProcessor", "DISPATCHING ActionResult Output keys: %v", keys)
	} else {
		logger.Info("DocProcessor", "DISPATCHING ActionResult Output: {}")
	}

	pubErr := s.publishReply(ctx, msg, replySubj, resp, startTime, identity, attempt)
	if pubErr != nil {
		return
	}

	s.ackMessage(msg, startTime, identity, attempt)
}

func (s *NatsServer) handleCppMLRequest(ctx context.Context, req *pb.ExecuteTaskRequest, payload *pb.ExecuteTaskPayload, identity *pb.InfrastructureIdentity) (res *structpb.Struct, err error) {
	now := time.Now()
	metric := db.MlMetric{
		TenantID:  identity.GetTenantId(),
		AppID:     identity.GetAppId(),
		JobID:     identity.GetJobId(),
		StepName:  "RedactPii",
		Status:    "STARTED",
		StartedAt: now,
	}
	if s.db != nil {
		s.db.Create(&metric)
	}

	defer func() {
		if err != nil {
			metric.Status = "FAILED"
			metric.ErrorMessage = err.Error()
		} else {
			metric.Status = "COMPLETED"
		}
		comp := time.Now()
		metric.CompletedAt = &comp
		if s.db != nil {
			s.db.Save(&metric)
		}
	}()

	// Boundary: gRPC call to cpp_ml_worker
	// We dispatch a synchronous gRPC call passing the RedactDocumentRequest using the Document URI.
	// This ensures we block the NATS consumer so that the message isn't ACKed until the worker finishes,
	// keeping data safe. The ack window extender handles the NATS message timeouts.
	mlReq := &pb.RedactDocumentRequest{
		Identity:    identity,
		DocumentUri: payload.Storage.DocumentUri,	}

	if s.cppMlClient == nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: ML client not configured"))
	}

	resp, mlErr := s.cppMlClient.RedactDocument(ctx, mlReq)
	if mlErr != nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: ML RedactDocument RPC failed: %w", mlErr))
	}
	if !resp.GetSuccess() {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: ML RedactDocument returned false: %s", resp.GetErrorMessage()))
	}

	resultMap := map[string]interface{}{
		"redacted_document_uri": payload.Storage.DocumentUri,	}

	resStruct, structErr := structpb.NewStruct(resultMap)
	if structErr != nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: failed to create structResult: %w", structErr))
	}

	return resStruct, nil
}

func (s *NatsServer) handleDotnetMLRequest(ctx context.Context, req *pb.ExecuteTaskRequest, payload *pb.ExecuteTaskPayload, identity *pb.InfrastructureIdentity) (res *structpb.Struct, err error) {
	now := time.Now()
	metric := db.MlMetric{
		TenantID:  identity.GetTenantId(),
		AppID:     identity.GetAppId(),
		JobID:     identity.GetJobId(),
		StepName:  "RedactPii_DotNet",
		Status:    "STARTED",
		StartedAt: now,
	}
	if s.db != nil {
		s.db.Create(&metric)
	}

	defer func() {
		if err != nil {
			metric.Status = "FAILED"
			metric.ErrorMessage = err.Error()
		} else {
			metric.Status = "COMPLETED"
		}
		comp := time.Now()
		metric.CompletedAt = &comp
		if s.db != nil {
			s.db.Save(&metric)
		}
	}()

	if s.dotnetMlClient == nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: Dotnet ML client not configured"))
	}

	rawBytes, err := s.store.FetchFile(ctx, payload.Storage.DocumentUri, identity)
	if err != nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: failed to fetch document for redaction: %w", err))
	}

	mlReq := &pb.PiiRedactionRequest{
		Identity:       identity,
		TargetPiiTypes: []pb.PiiType{pb.PiiType_PII_TYPE_PERSON_NAME, pb.PiiType_PII_TYPE_SSN, pb.PiiType_PII_TYPE_PHONE_NUMBER, pb.PiiType_PII_TYPE_EMAIL, pb.PiiType_PII_TYPE_ADDRESS}, // Defaulting to some standard types as this comes from custom instructions usually
		RawFileBytes:   rawBytes,
	}

	resp, mlErr := s.dotnetMlClient.RedactPii(ctx, mlReq)
	if mlErr != nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: ML RedactPii RPC failed: %w", mlErr))
	}
	if !resp.GetSuccess() {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: ML RedactPii returned false: %s", resp.GetErrorMessage()))
	}

	redactedUri, err := s.store.SaveFile(ctx, payload.Storage.DocumentUri+"_redacted.pdf", identity, resp.GetRedactedFileBytes())
	if err != nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: failed to save redacted document: %w", err))
	}

	resultMap := map[string]interface{}{
		"redacted_document_uri": redactedUri,
	}

	resStruct, structErr := structpb.NewStruct(resultMap)
	if structErr != nil {
		panic(fmt.Errorf("action failed for job DocProcessorWorker: failed to create structResult: %w", structErr))
	}

	return resStruct, nil
}

func (s *NatsServer) handleDualMLRequest(ctx context.Context, req *pb.ExecuteTaskRequest, payload *pb.ExecuteTaskPayload, identity *pb.InfrastructureIdentity) (*structpb.Struct, error) {
	var cppResult *structpb.Struct
	var dotnetResult *structpb.Struct
	var cppErr error
	var dotnetErr error

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		cppResult, cppErr = s.handleCppMLRequest(ctx, req, payload, identity)
	}()

	go func() {
		defer wg.Done()
		dotnetResult, dotnetErr = s.handleDotnetMLRequest(ctx, req, payload, identity)
	}()

	wg.Wait()

	diffMetrics := map[string]interface{}{
		"cpp_has_error":     cppErr != nil,
		"dotnet_has_error":  dotnetErr != nil,
		"cpp_keys_count":    0,
		"dotnet_keys_count": 0,
		"keys_match":        false,
	}

	if cppResult != nil {
		diffMetrics["cpp_keys_count"] = len(cppResult.Fields)
	}
	if dotnetResult != nil {
		diffMetrics["dotnet_keys_count"] = len(dotnetResult.Fields)
	}
	diffMetrics["keys_match"] = diffMetrics["cpp_keys_count"] == diffMetrics["dotnet_keys_count"]

	metricsBytes, err := json.Marshal(diffMetrics)
	if err == nil {
		logger.Info("DocProcessor", "Dual ML Worker Result Diff for JobID %s: %s", identity.GetJobId(), string(metricsBytes))
	} else {
		logger.Info("DocProcessor", "Dual ML Worker Result Diff for JobID %s: <failed to marshal metrics>", identity.GetJobId())
	}

	if cppErr != nil {
		return nil, cppErr
	}

	return cppResult, nil
}
