package server

import (
	"bytes"
	"context"
	"io"
	"os"
	"testing"
	"time"

	"document_processing_service/internal/processor"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

type mockMsgForTerm struct {
	data           []byte
	subject        string
	termed         bool
	acked          bool
	nakedWithDelay bool
}

func (m *mockMsgForTerm) Metadata() (*jetstream.MsgMetadata, error) {
	return nil, nil
}
func (m *mockMsgForTerm) Subject() string {
	if m.subject == "" {
		return "worker.doc.chunking"
	}
	return m.subject
}
func (m *mockMsgForTerm) Data() []byte {
	return m.data
}
func (m *mockMsgForTerm) Headers() nats.Header {
	return nil
}
func (m *mockMsgForTerm) Reply() string {
	return ""
}
func (m *mockMsgForTerm) Ack() error {
	m.acked = true
	return nil
}
func (m *mockMsgForTerm) DoubleAck(context.Context) error {
	return nil
}
func (m *mockMsgForTerm) Nak() error {
	return nil
}
func (m *mockMsgForTerm) NakWithDelay(delay time.Duration) error {
	m.nakedWithDelay = true
	return nil
}
func (m *mockMsgForTerm) InProgress() error {
	return nil
}
func (m *mockMsgForTerm) Term() error {
	m.termed = true
	return nil
}
func (m *mockMsgForTerm) TermWithReason(reason string) error {
	m.termed = true
	return nil
}

func TestProcessMessage_InvalidProtobufPayload_CallsTerm(t *testing.T) {
	server := &NatsServer{}

	// Invalid wire format protobuf payload
	invalidPayload := []byte{0xFF, 0xFF, 0xFF, 0xFF, 0x07, 0x00}
	msg := &mockMsgForTerm{
		data:    invalidPayload,
		subject: "worker.doc.test",
	}

	server.processMessage(context.Background(), msg)

	if !msg.termed {
		t.Fatalf("Expected msg.Term() to be called on invalid protobuf payload, but it was not")
	}
	if msg.acked {
		t.Fatalf("Expected msg.Ack() NOT to be called on invalid protobuf payload")
	}
}

func init() {
	os.Setenv("WORKERS_STREAM", "WORKERS")
}

func TestProcessMessage_MissingUri_ErrorContractEnriched(t *testing.T) {
	server := &NatsServer{}

	payload, _ := structpb.NewStruct(map[string]interface{}{})
	// Valid ExecuteTaskRequest but missing document_uri in Payload
	req := &pb.ExecuteTaskRequest{
		StepId: "step-no-uri",
		Identity: &pb.InfrastructureIdentity{
			TenantId: "t1",
			AppId:    "a1",
			JobId:    "j1",
		},
		TaskSubject: "worker.doc.inspect",
		Payload:     payload,
	}
	b, err := proto.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	msg := &mockMsgForTerm{
		data:    b,
		subject: "worker.doc.test",
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("Expected panic due to missing document_uri, but got no panic")
		}
	}()

	server.processMessage(context.Background(), msg)
}

func TestBuildDocsStructResult_PropagatesIsScanned(t *testing.T) {
	docs := []*pb.ProcessedDocument{
		{
			DocumentType:     "classified",
			RelativeFilePath: "test/scanned.pdf",
			PageCount:        1,
			IsScanned:        true,
		},
		{
			DocumentType:     "classified",
			RelativeFilePath: "test/text.pdf",
			PageCount:        2,
			IsScanned:        false,
		},
	}
	res, err := buildDocsStructResult(docs)
	if err != nil {
		t.Fatalf("buildDocsStructResult returned unexpected error: %v", err)
	}
	if res == nil {
		t.Fatalf("buildDocsStructResult returned nil struct")
	}
	docsValue, ok := res.Fields["documents"]
	if !ok {
		t.Fatalf("Expected 'documents' key in result struct")
	}
	list := docsValue.GetListValue().GetValues()
	if len(list) != 2 {
		t.Fatalf("Expected 2 documents in list, got %d", len(list))
	}
	scannedVal := list[0].GetStructValue().Fields["is_scanned"].GetBoolValue()
	if !scannedVal {
		t.Errorf("Expected doc 0 is_scanned to be true, got false")
	}
}

func TestHandleMessageWithRecovery_Panic_NakWithDelay(t *testing.T) {
	server := &NatsServer{}
	msg := &mockMsgForTerm{
		subject: "worker.doc.test",
	}
	sem := make(chan struct{}, 1)

	// processMessage panics because req is empty (GetTaskSubject == "")
	server.HandleMessageWithRecovery(context.Background(), msg, sem)

	if !msg.nakedWithDelay {
		t.Errorf("Expected NakWithDelay to be called on panic")
	}
}

type mockStorageForInspect struct {
	data []byte
}

func (m *mockStorageForInspect) FetchFile(ctx context.Context, path string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	return m.data, nil
}
func (m *mockStorageForInspect) FetchStream(ctx context.Context, path string, identity *pb.InfrastructureIdentity) (io.ReadCloser, error) {
	return io.NopCloser(bytes.NewReader(m.data)), nil
}
func (m *mockStorageForInspect) SaveFile(ctx context.Context, path string, identity *pb.InfrastructureIdentity, data []byte) (string, error) {
	return path, nil
}

func TestInspectDocument_TiffImageInspection(t *testing.T) {
	tiffBytes := []byte{0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00}
	store := &mockStorageForInspect{data: tiffBytes}
	chunker := processor.NewChunkingProcessor(nil, store)

	req := &pb.ExecuteTaskRequest{
		Identity: &pb.InfrastructureIdentity{
			TenantId: "tenant-1",
			AppId:    "app-1",
			JobId:    "job-tiff-1",
		},
		TaskSubject: "worker.doc.inspect",
	}

	res, err := chunker.InspectDocument(context.Background(), req, "internal-store://tenant-1/app-1/job-tiff-1/document.tif", func(msg, level string) {})
	if err != nil {
		t.Fatalf("InspectDocument returned error: %v", err)
	}

	if res == nil || res.Fields == nil {
		t.Fatalf("Expected non-nil result fields")
	}

	if res.Fields["MimeType"].GetStringValue() != "image/tiff" {
		t.Errorf("Expected MimeType=image/tiff, got %s", res.Fields["MimeType"].GetStringValue())
	}
	if !res.Fields["IsScanned"].GetBoolValue() {
		t.Errorf("Expected IsScanned=true, got %v", res.Fields["IsScanned"].GetBoolValue())
	}
}
