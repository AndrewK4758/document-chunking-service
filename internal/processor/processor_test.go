package processor

import (
	"archive/zip"
	"context"
	"os"
	"path/filepath"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
)

// mockAiClient is a simple mock for the gRPC client
type mockAiClient struct{} // We don't actually call it in ProcessZipFile yet since it's stubbed out.

// mockStorage provides a dummy local disk storage for unit tests
type mockStorage struct {
	basePath string
}

func (m *mockStorage) FetchFile(ctx context.Context, relPath string) ([]byte, error) {
	return os.ReadFile(filepath.Join(m.basePath, relPath))
}

func (m *mockStorage) SaveFile(ctx context.Context, relPath string, data []byte) (string, error) {
	full := filepath.Join(m.basePath, relPath)
	os.MkdirAll(filepath.Dir(full), 0755)
	err := os.WriteFile(full, data, 0644)
	return relPath, err
}

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

func TestProcessZipFile_HappyPath(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "test.zip")

	createTestZip(t, map[string]string{
		"doc1.pdf":  "dummy pdf content",
		"doc2.pdf":  "dummy pdf content",
		"image.jpg": "dummy image",
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	zipBytes, _ := os.ReadFile(zipPath)
	chunker := NewChunkingProcessor(nil, store)
	docs, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{RawFileBytes: zipBytes})
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// We expect 2 docs since there are 2 PDFs
	if len(docs) != 2 {
		t.Errorf("Expected 2 processed documents, got %d", len(docs))
	}
}

func TestProcessZipFile_EmptyPath(t *testing.T) {
	chunker := NewChunkingProcessor(nil, nil)
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{})
	if err == nil {
		t.Fatal("Expected error for empty zip path, got nil")
	}
	if err.Error() != "no file data provided in request" {
		t.Errorf("Unexpected error message: %v", err)
	}
}

func TestProcessZipFile_InvalidZip(t *testing.T) {
	tempDir := t.TempDir()
	invalidZip := filepath.Join(tempDir, "bad.zip")

	if err := os.WriteFile(invalidZip, []byte("this is not a real zip format"), 0644); err != nil {
		t.Fatalf("Failed to write bad zip: %v", err)
	}

	store := &mockStorage{basePath: tempDir}
	chunker := NewChunkingProcessor(nil, store)
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{RelativeFilePath: "bad.zip"})
	if err == nil {
		t.Fatal("Expected error for invalid zip, got nil")
	}
}

func TestProcessZipFile_ZipSlip(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "slip.zip")

	// Create a zip with a relative path attempt
	createTestZip(t, map[string]string{
		"../../malicious.pdf": "dummy pdf content",
		"safe.pdf":            "safe pdf content",
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	zipBytes, _ := os.ReadFile(zipPath)
	chunker := NewChunkingProcessor(nil, store)
	docs, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{RawFileBytes: zipBytes})
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Should only extract the safe.pdf
	if len(docs) != 1 {
		t.Errorf("Expected 1 processed document (malicious skipped), got %d", len(docs))
	}
}

func TestProcessZipFile_ContextCancel(t *testing.T) {
	chunker := NewChunkingProcessor(nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, _, err := chunker.ProcessZipFile(ctx, &pb.ChunkDocumentRequest{RawFileBytes: []byte("dummy")})
	if err == nil {
		t.Fatal("Expected error due to context cancellation, got nil")
	}
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled error, got %v", err)
	}
}

func TestProcessZipFile_StorageDown(t *testing.T) {
	// A valid storage manager, but we pass a file that doesn't exist
	tempDir := t.TempDir()
	store := &mockStorage{basePath: tempDir}
	chunker := NewChunkingProcessor(nil, store)

	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		RelativeFilePath: "missing_from_storage.zip",
	})

	if err == nil {
		t.Fatal("Expected error due to missing file, got nil")
	}
	if err.Error() == "no file data provided in request" {
		t.Errorf("Expected storage fetch error, got missing data error")
	}
}
