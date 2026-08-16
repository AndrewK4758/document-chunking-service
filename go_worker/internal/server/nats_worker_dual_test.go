package server

import (
	"context"
	"io"
	"testing"
	"time"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

type mockStorageManager struct{}

func (m *mockStorageManager) FetchFile(ctx context.Context, uri string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	return []byte("dummy file content"), nil
}

func (m *mockStorageManager) FetchStream(ctx context.Context, uri string, identity *pb.InfrastructureIdentity) (io.ReadCloser, error) {
	return nil, nil
}

func (m *mockStorageManager) SaveFile(ctx context.Context, uri string, identity *pb.InfrastructureIdentity, content []byte) (string, error) {
	return uri + "_redacted.pdf", nil
}


func TestProcessMessage_DualMLRequest_CppFails(t *testing.T) {
	mockCpp := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			panic("cpp failure")
		},
	}

	mockDotnet := &MockMlWorkerServiceClient{
		RedactPiiFunc: func(ctx context.Context, in *pb.PiiRedactionRequest, opts ...grpc.CallOption) (*pb.PiiRedactionResponse, error) {
			return &pb.PiiRedactionResponse{
				Success:           true,
				RedactedFileBytes: []byte("dotnet redacted bytes"),
			}, nil
		},
	}

	server := &NatsServer{
		cppMlClient:    mockCpp,
		dotnetMlClient: mockDotnet,
		store:          &mockStorageManager{},
	}

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{},
		"storage": map[string]interface{}{
			"document_uri": "test_doc_uri.pdf",
		},
	})
	
	req := &pb.ExecuteTaskRequest{
		StepId: "step-dual-cpp-fail",
		Identity: &pb.InfrastructureIdentity{
			TenantId: "t1",
			AppId:    "a1",
			JobId:    "j1",
		},
		TaskSubject: "WORKER.DOCUMENT_PROCESSING.DUAL_ML",
		Payload:     payload,
	}
	b, err := proto.Marshal(req)
	assert.NoError(t, err)

	msg := &mockMsgForTerm{
		data:    b,
		subject: "worker.doc.dual",
	}

	server.processMessage(context.Background(), msg)

	// Since cpp fails, the task should fail, meaning reply is sent with Success: false
	// processMessage calls ackMessage after handleProcessingError
	assert.True(t, msg.acked, "Expected msg.Ack() to be called after processing failure")
}

func TestProcessMessage_DualMLRequest_DotnetFails(t *testing.T) {
	mockCpp := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			return &pb.RedactDocumentResponse{
				Success: true,
			}, nil
		},
	}

	mockDotnet := &MockMlWorkerServiceClient{
		RedactPiiFunc: func(ctx context.Context, in *pb.PiiRedactionRequest, opts ...grpc.CallOption) (*pb.PiiRedactionResponse, error) {
			panic("dotnet failure")
		},
	}

	server := &NatsServer{
		cppMlClient:    mockCpp,
		dotnetMlClient: mockDotnet,
		store:          &mockStorageManager{},
	}

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{},
		"storage": map[string]interface{}{
			"document_uri": "test_doc_uri.pdf",
		},
	})
	
	req := &pb.ExecuteTaskRequest{
		StepId: "step-dual-dotnet-fail",
		Identity: &pb.InfrastructureIdentity{
			TenantId: "t1",
			AppId:    "a1",
			JobId:    "j1",
		},
		TaskSubject: "WORKER.DOCUMENT_PROCESSING.DUAL_ML",
		Payload:     payload,
	}
	b, err := proto.Marshal(req)
	assert.NoError(t, err)

	msg := &mockMsgForTerm{
		data:    b,
		subject: "worker.doc.dual",
	}

	server.processMessage(context.Background(), msg)

	// Since dotnet fails but cpp succeeds, the task succeeds
	assert.True(t, msg.acked, "Expected msg.Ack() to be called after processing dual ML request")
}

func TestProcessMessage_DualMLRequest(t *testing.T) {
	mockCpp := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			time.Sleep(10 * time.Millisecond) // simulate work
			return &pb.RedactDocumentResponse{
				Success: true,
			}, nil
		},
	}

	mockDotnet := &MockMlWorkerServiceClient{
		RedactPiiFunc: func(ctx context.Context, in *pb.PiiRedactionRequest, opts ...grpc.CallOption) (*pb.PiiRedactionResponse, error) {
			time.Sleep(20 * time.Millisecond) // simulate work, slightly slower
			return &pb.PiiRedactionResponse{
				Success:           true,
				RedactedFileBytes: []byte("dotnet redacted bytes"),
			}, nil
		},
	}

	server := &NatsServer{
		cppMlClient:    mockCpp,
		dotnetMlClient: mockDotnet,
		store:          &mockStorageManager{},
	}

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{},
		"storage": map[string]interface{}{
			"document_uri": "test_doc_uri.pdf",
		},
	})
	
	req := &pb.ExecuteTaskRequest{
		StepId: "step-dual-1",
		Identity: &pb.InfrastructureIdentity{
			TenantId: "t1",
			AppId:    "a1",
			JobId:    "j1",
		},
		TaskSubject: "WORKER.DOCUMENT_PROCESSING.DUAL_ML",
		Payload:     payload,
	}
	b, err := proto.Marshal(req)
	assert.NoError(t, err)

	msg := &mockMsgForTerm{
		data:    b,
		subject: "worker.doc.dual",
	}

	// This should run the dual handler and finish both
	server.processMessage(context.Background(), msg)

	assert.True(t, msg.acked, "Expected msg.Ack() to be called after processing dual ML request")
	assert.False(t, msg.termed, "Expected msg.Term() not to be called")
}
