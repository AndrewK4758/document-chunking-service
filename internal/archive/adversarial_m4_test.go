package archive

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

func TestAdversarial_Extract_ZeroByteArchive(t *testing.T) {
	tempDir := t.TempDir()
	emptyArchive := filepath.Join(tempDir, "zero_byte.zip")
	if err := os.WriteFile(emptyArchive, []byte{}, 0644); err != nil {
		t.Fatalf("Failed to create zero byte file: %v", err)
	}

	destDir := t.TempDir()
	extracted, err := Extract(emptyArchive, destDir)
	if err == nil {
		t.Fatalf("Expected error when extracting 0-byte zip file, got nil. Extracted: %v", extracted)
	}
}

func TestAdversarial_Extract_ZipWithDirectoriesOnly(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "dirs_only.zip")

	outFile, err := os.Create(zipPath)
	if err != nil {
		t.Fatalf("Failed to create zip: %v", err)
	}
	w := zip.NewWriter(outFile)
	_, err = w.Create("nested_folder/")
	if err != nil {
		t.Fatalf("Failed to create dir entry in zip: %v", err)
	}
	w.Close()
	outFile.Close()

	destDir := t.TempDir()
	extracted, err := Extract(zipPath, destDir)
	if err != nil {
		t.Fatalf("Unexpected error extracting dir-only zip: %v", err)
	}
	if len(extracted) != 0 {
		t.Errorf("Expected 0 extracted file paths (directories excluded), got %d: %v", len(extracted), extracted)
	}
}

func TestAdversarial_Extract_ZipPathTraversalProtection(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "traversal.zip")

	outFile, err := os.Create(zipPath)
	if err != nil {
		t.Fatalf("Failed to create zip: %v", err)
	}
	w := zip.NewWriter(outFile)
	f, err := w.Create("../outside.pdf")
	if err != nil {
		t.Fatalf("Failed to create entry in zip: %v", err)
	}
	f.Write([]byte("malicious content"))
	w.Close()
	outFile.Close()

	destDir := t.TempDir()
	extracted, err := Extract(zipPath, destDir)
	// Extract should either error out or filter out malicious paths
	for _, p := range extracted {
		if filepath.Clean(p) == filepath.Clean(filepath.Join(destDir, "../outside.pdf")) {
			t.Errorf("ZipSlip vulnerability detected: file extracted outside target destination! %s", p)
		}
	}
	_ = err
}
