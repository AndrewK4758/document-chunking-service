package archive

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"document_processing_service/internal/testutil"
)

func createTestZip(t *testing.T, files map[string]string, zipPath string) {
	t.Helper()
	testutil.CreateTestZip(t, files, zipPath)
}

func createTestTarGz(t *testing.T, files map[string]string, tarPath string) {
	t.Helper()
	outFile, err := os.Create(tarPath)
	if err != nil {
		t.Fatalf("Failed to create tar.gz file: %v", err)
	}
	defer outFile.Close()

	gw := gzip.NewWriter(outFile)
	defer gw.Close()

	tw := tar.NewWriter(gw)
	defer tw.Close()

	for name, content := range files {
		hdr := &tar.Header{
			Name: name,
			Mode: 0600,
			Size: int64(len(content)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("Failed to write tar header: %v", err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatalf("Failed to write tar content: %v", err)
		}
	}
}

func TestExtract_RawDocument(t *testing.T) {
	tempDir := t.TempDir()
	docPath := filepath.Join(tempDir, "document.pdf")

	if err := os.WriteFile(docPath, []byte("dummy pdf content"), 0644); err != nil {
		t.Fatalf("Failed to write dummy pdf: %v", err)
	}

	extractDir := t.TempDir()
	extracted, err := Extract(docPath, extractDir)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(extracted) != 1 {
		t.Fatalf("Expected exactly 1 extracted path, got %d", len(extracted))
	}
	if extracted[0] != docPath {
		t.Errorf("Expected extracted path to be exactly the input path %s, got %s", docPath, extracted[0])
	}
}

func TestExtract_ValidZip(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "test.zip")
	createTestZip(t, map[string]string{
		"doc1.pdf": "doc1",
		"doc2.txt": "doc2",
	}, zipPath)

	extractDir := t.TempDir()
	extracted, err := Extract(zipPath, extractDir)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(extracted) != 2 {
		t.Fatalf("Expected 2 extracted files, got %d", len(extracted))
	}

	var foundPdf, foundTxt bool
	for _, f := range extracted {
		if strings.HasSuffix(f, "doc1.pdf") {
			foundPdf = true
		}
		if strings.HasSuffix(f, "doc2.txt") {
			foundTxt = true
		}
		if !strings.HasPrefix(f, extractDir) {
			t.Errorf("Extracted file %s is not within destination dir %s", f, extractDir)
		}
	}

	if !foundPdf || !foundTxt {
		t.Errorf("Failed to find both extracted files in array: %v", extracted)
	}
}

func TestExtract_ValidTarGz(t *testing.T) {
	tempDir := t.TempDir()
	tarPath := filepath.Join(tempDir, "test.tar.gz")
	createTestTarGz(t, map[string]string{
		"doc.pdf": "doc content",
	}, tarPath)

	extractDir := t.TempDir()
	extracted, err := Extract(tarPath, extractDir)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(extracted) != 1 {
		t.Fatalf("Expected 1 extracted file, got %d", len(extracted))
	}
	if !strings.HasSuffix(extracted[0], "doc.pdf") {
		t.Errorf("Expected extracted file to be doc.pdf, got %s", extracted[0])
	}
}

func TestExtract_CorruptArchive(t *testing.T) {
	tempDir := t.TempDir()
	badZip := filepath.Join(tempDir, "corrupt.zip")
	os.WriteFile(badZip, bytes.Repeat([]byte("bad zip data"), 10), 0644)

	extractDir := t.TempDir()
	_, err := Extract(badZip, extractDir)
	if err == nil {
		t.Fatal("Expected error for corrupt zip file, got nil")
	}
	if !strings.Contains(err.Error(), "archiver failed to extract") {
		t.Errorf("Expected archiver error, got %v", err)
	}

	bad7z := filepath.Join(tempDir, "corrupt.7z")
	os.WriteFile(bad7z, bytes.Repeat([]byte("bad 7z data"), 10), 0644)
	_, err = Extract(bad7z, extractDir)
	if err == nil {
		t.Fatal("Expected error for corrupt 7z file, got nil")
	}
	if !strings.Contains(err.Error(), "sevenzip failed to extract") {
		t.Errorf("Expected sevenzip error, got %v", err)
	}
}

func TestExtract_UnsupportedExtension(t *testing.T) {
	tempDir := t.TempDir()
	unknownPath := filepath.Join(tempDir, "file.unknown_format")
	os.WriteFile(unknownPath, []byte("some data"), 0644)

	extractDir := t.TempDir()
	extracted, err := Extract(unknownPath, extractDir)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(extracted) != 1 {
		t.Fatalf("Expected exactly 1 extracted path, got %d", len(extracted))
	}
	if extracted[0] != unknownPath {
		t.Errorf("Expected extracted path to be %s, got %s", unknownPath, extracted[0])
	}
}
