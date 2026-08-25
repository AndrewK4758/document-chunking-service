package server

import (
    "context"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "strings"
    "testing"

    "document_processing_service/internal/testutil"

    pb "github.com/AndrewK4758/shared_protos"
    "google.golang.org/protobuf/types/known/structpb"
)

// mockStorage provides a dummy local disk storage for unit tests
type mockStorage struct {
	basePath string
}

func (m *mockStorage) FetchFile(ctx context.Context, relPath string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	return os.ReadFile(filepath.Join(m.basePath, relPath))
}

func (m *mockStorage) FetchStream(ctx context.Context, relPath string, identity *pb.InfrastructureIdentity) (io.ReadCloser, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockStorage) SaveFile(ctx context.Context, relPath string, identity *pb.InfrastructureIdentity, data []byte) (string, error) {
	full := filepath.Join(m.basePath, relPath)
	os.MkdirAll(filepath.Dir(full), 0755)
	err := os.WriteFile(full, data, 0644)
	return relPath, err
}

func TestChunkDocument_Success(t *testing.T) {
	tempDir := t.TempDir()
	store := &mockStorage{basePath: tempDir}
	srv := NewDocumentProcessingServer(nil, nil, nil, store)

	zipPath := filepath.Join(tempDir, "test_valid.zip")

	testutil.CreateTestZip(t, map[string]string{
		"doc1.pdf": "dummy",
	}, zipPath)

	zipBytes, _ := os.ReadFile(zipPath)
	payload1, _ := structpb.NewStruct(map[string]any{
		"node_config": map[string]any{},
		"global_state": map[string]any{
			"functional": map[string]any{},
		},
		"storage": map[string]any{"document_uri": "test_valid.zip"},
		"document_processor": map[string]any{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
			"action_name_classify":   "action",
			"doc_type_classified":    "classified",
			"action_name_boundary":   "boundary",
			"doc_type_chunked":       "chunked",
		},
	})
	req := &pb.ChunkDocumentRequest{
		Identity:     &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_123"},
		RawFileBytes: zipBytes,
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_123"},
			TaskSubject: "worker.doc.inspect",
			Payload:     payload1,
		},
	}

	mockStream := &mockChunkDocumentServer{req: req}
	err := srv.ChunkDocument(mockStream)
	resp := mockStream.resp
	if err != nil {
		t.Fatalf("Expected no err, got %v", err)
	}
	if !resp.Success {
		t.Errorf("Expected Success=true, got false. ErrorMessage: %s", resp.ErrorMessage)
	}
	if len(resp.ProcessedDocuments) != 1 {
		t.Errorf("Expected 1 processed document, got %d", len(resp.ProcessedDocuments))
	}
}

func TestChunkDocument_Failure(t *testing.T) {
	tempDir := t.TempDir()
	store := &mockStorage{basePath: tempDir}
	srv := NewDocumentProcessingServer(nil, nil, nil, store)

	payload2, _ := structpb.NewStruct(map[string]any{
		"node_config": map[string]any{},
		"global_state": map[string]any{
			"functional": map[string]any{},
		},
		"storage": map[string]any{"document_uri": "nonexistent/path/to/bad.zip"},
		"document_processor": map[string]any{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
		},
	})
	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_bad"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_bad"},
			TaskSubject: "worker.doc.inspect",
			Payload:     payload2,
		},
	}

	mockStream := &mockChunkDocumentServer{req: req}
	err := srv.ChunkDocument(mockStream)
	resp := mockStream.resp
	// It shouldn't return a raw gRPC error, but rather return a graceful pb.ChunkDocumentResponse with Success=false
	if err != nil {
		t.Fatalf("Expected no gRPC error, got %v", err)
	}
	if resp.Success {
		t.Error("Expected Success=false for bad zip path, got true")
	}
	if resp.ErrorMessage == "" {
		t.Error("Expected an ErrorMessage, got empty string")
	}
	expectedPrefix := "[Step: JIT_FETCH, URI: nonexistent/path/to/bad.zip]"
	if !strings.HasPrefix(resp.ErrorMessage, expectedPrefix) {
		t.Errorf("Expected ErrorMessage prefix %q, got %q", expectedPrefix, resp.ErrorMessage)
	}
}

func TestChunkDocument_EmptyRequest(t *testing.T) {
	srv := NewDocumentProcessingServer(nil, nil, nil, nil)

	payload3, _ := structpb.NewStruct(map[string]any{
		"node_config": map[string]any{},
		"global_state": map[string]any{
			"functional": map[string]any{},
		},
	})
	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_empty"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_empty"},
			TaskSubject: "worker.doc.inspect",
			Payload:     payload3,
		},
	}

	mockStream := &mockChunkDocumentServer{req: req}
	err := srv.ChunkDocument(mockStream)
	resp := mockStream.resp
	if err != nil {
		t.Fatalf("Expected no gRPC error, got %v", err)
	}
	if resp.Success {
		t.Error("Expected Success=false for empty request, got true")
	}
	if !strings.Contains(resp.ErrorMessage, "missing document_uri in storage payload") {
		t.Errorf("Unexpected ErrorMessage: %s", resp.ErrorMessage)
	}
}
