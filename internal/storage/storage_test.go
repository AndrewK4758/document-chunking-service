package storage

import (
	"context"
	"fmt"
	"io"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// --- MOCK DEFINITIONS ---

type mockStorageGatewayClient struct {
	uploadStream   *mockUploadStream
	downloadStream *mockDownloadStream
	uploadErr      error
	downloadErr    error
}

func (m *mockStorageGatewayClient) UploadDocument(ctx context.Context, opts ...grpc.CallOption) (pb.StorageGatewayService_UploadDocumentClient, error) {
	if m.uploadErr != nil {
		return nil, m.uploadErr
	}
	return m.uploadStream, nil
}

func (m *mockStorageGatewayClient) DownloadDocument(ctx context.Context, in *pb.DownloadRequest, opts ...grpc.CallOption) (pb.StorageGatewayService_DownloadDocumentClient, error) {
	if m.downloadErr != nil {
		return nil, m.downloadErr
	}
	return m.downloadStream, nil
}

func (m *mockStorageGatewayClient) ClearJobData(ctx context.Context, in *pb.ClearJobDataRequest, opts ...grpc.CallOption) (*pb.ClearJobDataResponse, error) {
	return &pb.ClearJobDataResponse{Success: true}, nil
}

// mockUploadStream
type mockUploadStream struct {
	grpc.ClientStream
	sendErr      error
	closeResp    *pb.UploadResponse
	closeErr     error
	receivedReqs []*pb.UploadRequest
}

func (m *mockUploadStream) Send(req *pb.UploadRequest) error {
	if m.sendErr != nil {
		return m.sendErr
	}
	m.receivedReqs = append(m.receivedReqs, req)
	return nil
}

func (m *mockUploadStream) CloseAndRecv() (*pb.UploadResponse, error) {
	return m.closeResp, m.closeErr
}

// mockDownloadStream
type mockDownloadStream struct {
	grpc.ClientStream
	responses []*pb.DownloadResponse
	errs      []error
	idx       int
}

func (m *mockDownloadStream) Recv() (*pb.DownloadResponse, error) {
	if m.idx >= len(m.responses) {
		if m.idx < len(m.errs) && m.errs[m.idx] != nil {
			return nil, m.errs[m.idx]
		}
		return nil, io.EOF
	}
	resp := m.responses[m.idx]
	err := m.errs[m.idx]
	m.idx++
	return resp, err
}

// --- TESTS ---

func TestGatewayStorage_FetchFile_Success(t *testing.T) {
	mockClient := &mockStorageGatewayClient{
		downloadStream: &mockDownloadStream{
			responses: []*pb.DownloadResponse{
				{ChunkData: []byte("hello ")},
				{ChunkData: []byte("world")},
			},
			errs: []error{nil, nil, io.EOF},
		},
	}
	store := NewGatewayStorage(mockClient)

	ctx := context.Background()
	data, err := store.FetchFile(ctx, "internal-store://test", nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if string(data) != "hello world" {
		t.Errorf("Expected 'hello world', got '%s'", string(data))
	}
}

func TestGatewayStorage_SaveFile_Success(t *testing.T) {
	uploadStream := &mockUploadStream{
		closeResp: &pb.UploadResponse{
			Success:    true,
			StorageUri: "internal-store://job123/file.pdf",
		},
	}
	mockClient := &mockStorageGatewayClient{
		uploadStream: uploadStream,
	}
	store := NewGatewayStorage(mockClient)

	ctx := context.Background()
	identity := &pb.InfrastructureIdentity{JobId: "job123"}
	uri, err := store.SaveFile(ctx, "job123/file.pdf", identity, []byte("data"))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if uri != "internal-store://job123/file.pdf" {
		t.Errorf("Unexpected URI: %s", uri)
	}
	if len(uploadStream.receivedReqs) != 2 { // 1 metadata chunk + 1 data chunk
		t.Errorf("Expected 2 chunks sent, got %d", len(uploadStream.receivedReqs))
	}
}

func TestGatewayStorage_SaveFile_InvalidPath(t *testing.T) {
	store := NewGatewayStorage(&mockStorageGatewayClient{})
	identity := &pb.InfrastructureIdentity{JobId: "job123"}
	_, err := store.SaveFile(context.Background(), "invalidpath", identity, []byte("data"))
	if err == nil {
		t.Fatal("Expected error for invalid path, got nil")
	}
}

func TestGatewayStorage_FetchFile_StreamError(t *testing.T) {
	mockClient := &mockStorageGatewayClient{
		downloadStream: &mockDownloadStream{
			responses: []*pb.DownloadResponse{
				{ChunkData: []byte("hello ")},
				nil, // Second fetch will return error
			},
			errs: []error{nil, fmt.Errorf("network partition")},
		},
	}
	store := NewGatewayStorage(mockClient)

	_, err := store.FetchFile(context.Background(), "internal-store://test", nil)
	if err == nil {
		t.Fatal("Expected error on stream interruption, got nil")
	}
}

func TestGatewayStorage_SaveFile_Rejected(t *testing.T) {
	mockClient := &mockStorageGatewayClient{
		uploadStream: &mockUploadStream{
			closeResp: &pb.UploadResponse{
				Success:      false,
				ErrorMessage: "quota exceeded",
			},
		},
	}
	store := NewGatewayStorage(mockClient)

	identity := &pb.InfrastructureIdentity{JobId: "job123"}
	_, err := store.SaveFile(context.Background(), "job123/file.pdf", identity, []byte("data"))
	if err == nil {
		t.Fatal("Expected error on server rejection, got nil")
	}
	if err.Error() != "gateway rejected upload for file.pdf: quota exceeded" {
		t.Errorf("Unexpected error message: %v", err)
	}
}

func TestGatewayStorage_ContextCancellation(t *testing.T) {
	store := NewGatewayStorage(&mockStorageGatewayClient{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := store.FetchFile(ctx, "internal-store://test", nil)
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled on FetchFile, got %v", err)
	}

	identity := &pb.InfrastructureIdentity{JobId: "job123"}
	_, err = store.SaveFile(ctx, "job123/file.pdf", identity, []byte("data"))
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled on SaveFile, got %v", err)
	}
}

func TestGatewayStorage_ForwardMetadata(t *testing.T) {
	// Add incoming metadata
	md := metadata.Pairs("tenant_id", "test_tenant")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	// Forward it
	reqCtx := forwardMetadata(ctx)

	// Verify it exists in outgoing
	outMD, ok := metadata.FromOutgoingContext(reqCtx)
	if !ok {
		t.Fatal("Expected outgoing metadata")
	}
	if len(outMD["tenant_id"]) == 0 || outMD["tenant_id"][0] != "test_tenant" {
		t.Errorf("Expected tenant_id=test_tenant in outgoing metadata, got %v", outMD)
	}
}
