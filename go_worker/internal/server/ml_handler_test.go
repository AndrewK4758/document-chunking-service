package server

import (
	"context"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"document_processing_service/internal/storage"
)

type mockStorageForMl struct {
	storage.Manager
}

func (m *mockStorageForMl) FetchFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	return []byte("test_bytes"), nil
}

func (m *mockStorageForMl) SaveFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity, data []byte) (string, error) {
	return "s3://redacted", nil
}

type MockDotnetMlWorkerServiceClient struct {
	RedactPiiFunc func(ctx context.Context, in *pb.PiiRedactionRequest, opts ...grpc.CallOption) (*pb.PiiRedactionResponse, error)
}

func (m *MockDotnetMlWorkerServiceClient) RedactPii(ctx context.Context, in *pb.PiiRedactionRequest, opts ...grpc.CallOption) (*pb.PiiRedactionResponse, error) {
	if m.RedactPiiFunc != nil {
		return m.RedactPiiFunc(ctx, in, opts...)
	}
	return nil, nil
}

func TestHandleCppMLRequest_Success(t *testing.T) {
	mockMlClient := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			assert.Equal(t, "s3://original", in.DocumentUri)
			assert.Equal(t, "t", in.Identity.TenantId)

			return &pb.RedactDocumentResponse{
				Success: true,
			}, nil
		},
	}

	server := &NatsServer{
		cppMlClient: mockMlClient,
		store:       &mockStorageForMl{},
	}

	req := &pb.ExecuteTaskRequest{}
	
	payload := &pb.ExecuteTaskPayload{
		Storage: &pb.StorageContract{
			DocumentUri: "s3://original",
		},
	}
	identity := &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"}

	res, err := server.handleCppMLRequest(context.Background(), req, payload, identity)
	assert.NoError(t, err)
	assert.NotNil(t, res)
}

func TestHandleCppMLRequest_PanicOnFailure(t *testing.T) {
	mockMlClient := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			return &pb.RedactDocumentResponse{
				Success: false,
				ErrorMessage: "ml error",
			}, nil
		},
	}

	server := &NatsServer{
		cppMlClient: mockMlClient,
		store:       &mockStorageForMl{},
	}

	payload := &pb.ExecuteTaskPayload{
		Storage: &pb.StorageContract{
			DocumentUri: "s3://original",
		},
	}
	identity := &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"}

	assert.Panics(t, func() {
		server.handleCppMLRequest(context.Background(), &pb.ExecuteTaskRequest{}, payload, identity)
	})
}

func TestHandleDotnetMLRequest_Success(t *testing.T) {
	mockDotnetClient := &MockDotnetMlWorkerServiceClient{
		RedactPiiFunc: func(ctx context.Context, in *pb.PiiRedactionRequest, opts ...grpc.CallOption) (*pb.PiiRedactionResponse, error) {
			assert.Equal(t, "t", in.Identity.TenantId)
			assert.NotNil(t, in.RawFileBytes)

			return &pb.PiiRedactionResponse{
				Success: true,
				RedactedFileBytes: []byte("redacted"),
			}, nil
		},
	}

	server := &NatsServer{
		dotnetMlClient: mockDotnetClient,
		store:          &mockStorageForMl{},
	}

	req := &pb.ExecuteTaskRequest{}
	
	payload := &pb.ExecuteTaskPayload{
		Storage: &pb.StorageContract{
			DocumentUri: "s3://original",
		},
	}
	identity := &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"}

	res, err := server.handleDotnetMLRequest(context.Background(), req, payload, identity)
	assert.NoError(t, err)
	assert.NotNil(t, res)
}
