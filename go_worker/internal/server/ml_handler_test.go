package server

import (
    "context"
    "testing"

    "document_processing_service/internal/storage"

    pb "github.com/AndrewK4758/shared_protos"
    "github.com/stretchr/testify/assert"
    "google.golang.org/grpc"
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

	payload := &pb.ExecuteTaskPayload{
		Storage: &pb.StorageContract{
			DocumentUri: "s3://original",
		},
	}
	identity := &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"}

	res, err := server.handleCppMLRequest(context.Background(), payload, identity)
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
		server.handleCppMLRequest(context.Background(), payload, identity)
	})
}

func TestHandleCppMLRequest_WithResidualReport_Success(t *testing.T) {
	mockMlClient := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			return &pb.RedactDocumentResponse{
				Success: true,
				ResidualReport: &pb.ResidualSweepReport{
					IsConverged:        true,
					ResidualValueCount: 2,
					ResidualSamples:    []string{"$1,200.00", "2026-08-24"},
					PassCount:          1,
				},
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

	res, err := server.handleCppMLRequest(context.Background(), payload, identity)
	assert.NoError(t, err)
	assert.NotNil(t, res)

	fields := res.GetFields()
	assert.Contains(t, fields, "residual_report")
	reportStruct := fields["residual_report"].GetStructValue()
	assert.NotNil(t, reportStruct)
	assert.True(t, reportStruct.Fields["is_converged"].GetBoolValue())
	assert.Equal(t, float64(2), reportStruct.Fields["residual_value_count"].GetNumberValue())
	assert.Equal(t, float64(1), reportStruct.Fields["pass_count"].GetNumberValue())

	samplesList := reportStruct.Fields["residual_samples"].GetListValue()
	assert.NotNil(t, samplesList)
	assert.Equal(t, 2, len(samplesList.Values))
	assert.Equal(t, "$1,200.00", samplesList.Values[0].GetStringValue())
	assert.Equal(t, "2026-08-24", samplesList.Values[1].GetStringValue())
}

func TestHandleCppMLRequest_UnconvergedExhausted_Panics(t *testing.T) {
	mockMlClient := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			return &pb.RedactDocumentResponse{
				Success: true,
				ResidualReport: &pb.ResidualSweepReport{
					IsConverged:        false,
					ResidualValueCount: 5,
					ResidualSamples:    []string{"$5,000"},
					PassCount:          3,
				},
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
		server.handleCppMLRequest(context.Background(), payload, identity)
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

	payload := &pb.ExecuteTaskPayload{
		Storage: &pb.StorageContract{
			DocumentUri: "s3://original",
		},
	}
	identity := &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"}

	res, err := server.handleDotnetMLRequest(context.Background(), payload, identity)
	assert.NoError(t, err)
	assert.NotNil(t, res)
}

