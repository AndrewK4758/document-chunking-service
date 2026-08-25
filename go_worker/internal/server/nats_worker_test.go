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
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

type mockMsgForTerm struct {
	data           []byte
	subject        string
	headers        nats.Header
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
	return m.headers
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

	payload, _ := structpb.NewStruct(map[string]any{})
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

func TestProcessMessage_SubjectValidationAndFailFast(t *testing.T) {
	server := &NatsServer{}

	t.Run("Missing TaskSubject panics immediately", func(t *testing.T) {
		req := &pb.ExecuteTaskRequest{
			Identity: &pb.InfrastructureIdentity{
				TenantId: "tenant-1",
				AppId:    "app-1",
				JobId:    "job-missing-subj",
			},
			TaskSubject: "",
		}
		data, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{
			data:    data,
			subject: "worker.doc.processing.cpp_ml",
		}

		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("expected panic on empty TaskSubject, but none occurred")
			}
		}()

		server.processMessage(context.Background(), msg)
	})

	t.Run("Missing Identity panics immediately", func(t *testing.T) {
		req := &pb.ExecuteTaskRequest{
			Identity:    nil,
			TaskSubject: "worker.doc.processing.cpp_ml",
		}
		data, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{
			data:    data,
			subject: "worker.doc.processing.cpp_ml",
		}

		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("expected panic on nil Identity, but none occurred")
			}
		}()

		server.processMessage(context.Background(), msg)
	})

	t.Run("Missing Payload panics immediately", func(t *testing.T) {
		req := &pb.ExecuteTaskRequest{
			Identity: &pb.InfrastructureIdentity{
				TenantId: "tenant-1",
				AppId:    "app-1",
				JobId:    "job-missing-payload",
			},
			TaskSubject: "worker.doc.processing.cpp_ml",
		}
		data, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{
			data:    data,
			subject: "worker.doc.processing.cpp_ml",
		}

		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("expected panic on nil Payload, but none occurred")
			}
		}()

		server.processMessage(context.Background(), msg)
	})

	t.Run("Missing Storage DocumentUri panics immediately", func(t *testing.T) {
		payload, _ := structpb.NewStruct(map[string]any{
			"nodeConfig": map[string]any{
				"node_name": "Layout Discovery",
			},
		})
		req := &pb.ExecuteTaskRequest{
			Identity: &pb.InfrastructureIdentity{
				TenantId: "tenant-1",
				AppId:    "app-1",
				JobId:    "job-missing-uri",
			},
			TaskSubject: "worker.doc.processing.cpp_ml",
			Payload:     payload,
		}
		data, _ := proto.Marshal(req)
		msg := &mockMsgForTerm{
			data:    data,
			subject: "worker.doc.processing.cpp_ml",
		}

		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("expected panic on missing storage document_uri, but none occurred")
			}
		}()

		server.processMessage(context.Background(), msg)
	})
}

type mockJSForForwarding struct {
	jetstream.JetStream
	publishedMsgs []*nats.Msg
}

func (m *mockJSForForwarding) Publish(ctx context.Context, subj string, data []byte, opts ...jetstream.PublishOpt) (*jetstream.PubAck, error) {
	return &jetstream.PubAck{}, nil
}

func (m *mockJSForForwarding) PublishMsg(ctx context.Context, msg *nats.Msg, opts ...jetstream.PublishOpt) (*jetstream.PubAck, error) {
	m.publishedMsgs = append(m.publishedMsgs, msg)
	return &jetstream.PubAck{}, nil
}

func TestProcessMessage_AutonomousRouting_TableDriven(t *testing.T) {
	tests := []struct {
		name              string
		pageCount         int32
		isScanned         bool
		envThreshold      string
		expectCppMLCalled bool
		expectAIPublished bool
	}{
		{
			name:              "Multi-page digital PDF routes to C++ ML worker",
			pageCount:         3,
			isScanned:         false,
			expectCppMLCalled: true,
			expectAIPublished: false,
		},
		{
			name:              "Single-page scanned PDF routes to C++ ML worker",
			pageCount:         1,
			isScanned:         true,
			expectCppMLCalled: true,
			expectAIPublished: false,
		},
		{
			name:              "Single-page digital PDF forwards to AI inference",
			pageCount:         1,
			isScanned:         false,
			expectCppMLCalled: false,
			expectAIPublished: true,
		},
		{
			name:              "Custom threshold allows 3-page digital PDF to forward to AI inference",
			pageCount:         3,
			isScanned:         false,
			envThreshold:      "5",
			expectCppMLCalled: false,
			expectAIPublished: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.envThreshold != "" {
				os.Setenv("ML_ROUTING_PAGE_THRESHOLD", tc.envThreshold)
				defer os.Unsetenv("ML_ROUTING_PAGE_THRESHOLD")
			} else {
				os.Unsetenv("ML_ROUTING_PAGE_THRESHOLD")
			}

			cppMLCalled := false
			mockCpp := &MockCppMlWorkerServiceClient{
				RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
					cppMLCalled = true
					return &pb.RedactDocumentResponse{Success: true}, nil
				},
			}
			mockJS := &mockJSForForwarding{}
			server := &NatsServer{
				js:          mockJS,
				cppMlClient: mockCpp,
			}

			payload, _ := structpb.NewStruct(map[string]any{
				"storage": map[string]any{
					"document_uri": "s3://bucket/tenant/app/job/doc.pdf",
					"page_count":   tc.pageCount,
				},
				"globalState": map[string]any{
					"functional": map[string]any{
						"is_scanned": tc.isScanned,
					},
				},
				"nodeConfig": map[string]any{
					"node_name": "Autonomous Processing",
				},
			})

			req := &pb.ExecuteTaskRequest{
				Identity: &pb.InfrastructureIdentity{
					TenantId: "tenant-auton",
					AppId:    "app-auton",
					JobId:    "job-auton",
				},
				TaskSubject: "worker.doc.processing", // Unclassified subject falls into default autonomous routing
				StepId:      "step-auton-1",
				Payload:     payload,
			}

			data, err := proto.Marshal(req)
			if err != nil {
				t.Fatalf("failed to marshal request: %v", err)
			}

			headers := nats.Header{}
			headers.Set("Reply-Subject", "ORCHESTRATOR_RESPONSES")

			msg := &mockMsgForTerm{
				data:    data,
				subject: "worker.doc.processing",
				headers: headers,
			}

			server.processMessage(context.Background(), msg)

			if tc.expectCppMLCalled && !cppMLCalled {
				t.Errorf("expected C++ ML worker to be called, but it was not")
			}
			if !tc.expectCppMLCalled && cppMLCalled {
				t.Errorf("expected C++ ML worker NOT to be called, but it was called")
			}

			if tc.expectAIPublished {
				if len(mockJS.publishedMsgs) != 1 {
					t.Fatalf("expected 1 message published to AI inference, got %d", len(mockJS.publishedMsgs))
				}
				pub := mockJS.publishedMsgs[0]
				if pub.Subject != "worker.ai.inference" {
					t.Errorf("expected published subject 'worker.ai.inference', got '%s'", pub.Subject)
				}
				if pub.Header.Get("Reply-Subject") != "ORCHESTRATOR_RESPONSES" {
					t.Errorf("expected Reply-Subject 'ORCHESTRATOR_RESPONSES', got '%s'", pub.Header.Get("Reply-Subject"))
				}
				if !msg.acked {
					t.Errorf("expected incoming message to be ACKed upon successful forwarding, but it was not")
				}
			} else {
				if len(mockJS.publishedMsgs) > 0 {
					t.Errorf("expected 0 messages published to AI inference, got %d", len(mockJS.publishedMsgs))
				}
			}
		})
	}
}


