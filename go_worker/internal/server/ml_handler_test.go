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

func TestHandleMLRequest_Success(t *testing.T) {
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
		mlClient: mockMlClient,
		store:    &mockStorageForMl{},
	}

	req := &pb.ExecuteTaskRequest{}
	
	payload := &pb.ExecuteTaskPayload{
		Storage: &pb.StorageContract{
			DocumentUri: "s3://original",
		},
	}
	identity := &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"}

	res, err := server.handleMLRequest(context.Background(), req, payload, identity)
	assert.NoError(t, err)
	assert.NotNil(t, res)
}

func TestHandleMLRequest_PanicOnFailure(t *testing.T) {
	mockMlClient := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			return &pb.RedactDocumentResponse{
				Success: false,
				ErrorMessage: "ml error",
			}, nil
		},
	}

	server := &NatsServer{
		mlClient: mockMlClient,
		store:    &mockStorageForMl{},
	}

	payload := &pb.ExecuteTaskPayload{
		Storage: &pb.StorageContract{
			DocumentUri: "s3://original",
		},
	}
	identity := &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"}

	assert.Panics(t, func() {
		server.handleMLRequest(context.Background(), &pb.ExecuteTaskRequest{}, payload, identity)
	})
}
