package server

import (
	"archive/zip"
	"context"
	"os"
	"path/filepath"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
)

// mockStorage provides a dummy local disk storage for unit tests
type mockStorage struct {
	basePath string
}

func (m *mockStorage) FetchFile(ctx context.Context, relPath string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	return os.ReadFile(filepath.Join(m.basePath, relPath))
}

func (m *mockStorage) SaveFile(ctx context.Context, relPath string, identity *pb.InfrastructureIdentity, data []byte) (string, error) {
	full := filepath.Join(m.basePath, relPath)
	os.MkdirAll(filepath.Dir(full), 0755)
	err := os.WriteFile(full, data, 0644)
	return relPath, err
}

// Helper function to create a test zip
func createTestZip(t *testing.T, files map[string]string, zipPath string) {
	t.Helper()
	outFile, err := os.Create(zipPath)
	if err != nil {
		t.Fatalf("Failed to create test zip file: %v", err)
	}
	defer outFile.Close()

	w := zip.NewWriter(outFile)
	for name, content := range files {
		f, err := w.Create(name)
		if err != nil {
			t.Fatalf("Failed to add file to zip: %v", err)
		}
		_, err = f.Write([]byte(content))
		if err != nil {
			t.Fatalf("Failed to write content to zip file: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Failed to close zip writer: %v", err)
	}
}

func TestChunkDocument_Success(t *testing.T) {
	tempDir := t.TempDir()
	store := &mockStorage{basePath: tempDir}
	srv := NewDocumentProcessingServer(nil, store)

	zipPath := filepath.Join(tempDir, "test_valid.zip")

	createTestZip(t, map[string]string{
		"doc1.pdf": "dummy",
	}, zipPath)

	zipBytes, _ := os.ReadFile(zipPath)
	req := &pb.ChunkDocumentRequest{
		Identity:     &pb.InfrastructureIdentity{JobId: "job_123"},
		RawFileBytes: zipBytes,
		OriginalRequest: &pb.ExecuteWorkflowNodeRequest{
			DocumentProcessor: &pb.DocumentProcessorContract{
				MaxPagesBeforeSplit: 10,
				SlidingWindowSize: 5,
				DocTypeCorrupted: "corrupted",
				ActionNameClassify: "action",
				DocTypeClassified: "classified",
				ActionNameBoundary: "boundary",
				DocTypeChunked: "chunked",
			},
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
	srv := NewDocumentProcessingServer(nil, store)

	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{JobId: "job_bad"},
		OriginalRequest: &pb.ExecuteWorkflowNodeRequest{
			Storage: &pb.StorageContract{DocumentUri: "nonexistent/path/to/bad.zip"},
			DocumentProcessor: &pb.DocumentProcessorContract{
				MaxPagesBeforeSplit: 10,
				SlidingWindowSize: 5,
				DocTypeCorrupted: "corrupted",
			},
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
}

func TestChunkDocument_EmptyRequest(t *testing.T) {
	srv := NewDocumentProcessingServer(nil, nil)

	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{JobId: "job_empty"},
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
	if resp.ErrorMessage != "no file data provided in request" {
		t.Errorf("Unexpected ErrorMessage: %s", resp.ErrorMessage)
	}
}
