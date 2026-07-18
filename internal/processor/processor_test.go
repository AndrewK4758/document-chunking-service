package processor

import (
	"archive/zip"
	"context"
	"os"
	"path/filepath"
	"testing"
)

// mockAiClient is a simple mock for the gRPC client
type mockAiClient struct{} // We don't actually call it in ProcessZipFile yet since it's stubbed out.

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
		"doc1.pdf": "dummy pdf content",
		"doc2.pdf": "dummy pdf content",
		"image.jpg": "dummy image",
	}, zipPath)

	chunker := NewChunkingProcessor(nil)
	docs, err := chunker.ProcessZipFile(context.Background(), zipPath)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// We expect 2 docs since there are 2 PDFs
	if len(docs) != 2 {
		t.Errorf("Expected 2 processed documents, got %d", len(docs))
	}
}

func TestProcessZipFile_EmptyPath(t *testing.T) {
	chunker := NewChunkingProcessor(nil)
	_, err := chunker.ProcessZipFile(context.Background(), "")
	if err == nil {
		t.Fatal("Expected error for empty zip path, got nil")
	}
	if err.Error() != "zip path cannot be empty" {
		t.Errorf("Unexpected error message: %v", err)
	}
}

func TestProcessZipFile_InvalidZip(t *testing.T) {
	tempDir := t.TempDir()
	invalidZip := filepath.Join(tempDir, "bad.zip")
	
	if err := os.WriteFile(invalidZip, []byte("this is not a real zip format"), 0644); err != nil {
		t.Fatalf("Failed to write bad zip: %v", err)
	}

	chunker := NewChunkingProcessor(nil)
	_, err := chunker.ProcessZipFile(context.Background(), invalidZip)
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

	chunker := NewChunkingProcessor(nil)
	docs, err := chunker.ProcessZipFile(context.Background(), zipPath)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Should only extract the safe.pdf
	if len(docs) != 1 {
		t.Errorf("Expected 1 processed document (malicious skipped), got %d", len(docs))
	}
}
