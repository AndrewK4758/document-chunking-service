package processor

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/grpc"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/protobuf/types/known/structpb"
)

// mockAiClient is a simple mock for the gRPC client
type mockAiClient struct{}

var _ pb.DocumentProcessorClient = (*mockAiClient)(nil)

func (m *mockAiClient) GetAvailableModels(ctx context.Context, in *pb.GetAvailableModelsRequest, opts ...grpc.CallOption) (*pb.GetAvailableModelsResponse, error) {
	return nil, nil
}

func (m *mockAiClient) PerformAction(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[pb.PerformActionRequest, pb.PerformActionResponse], error) {
	return nil, nil
}

func (m *mockAiClient) ProcessDocument(ctx context.Context, opts ...grpc.CallOption) (grpc.BidiStreamingClient[pb.ProcessDocumentRequest, pb.ProcessDocumentResponse], error) {
	return nil, nil
}

// mockStorage provides a dummy local disk storage for unit tests
type mockStorage struct {
	basePath  string
	fetchData []byte
	fetchErr  error
}

func (m *mockStorage) FetchFile(ctx context.Context, relPath string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	if m.fetchErr != nil {
		return nil, m.fetchErr
	}
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
	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "test.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
			"action_name_classify":   "action",
			"doc_type_classified":    "classified",
			"action_name_boundary":   "boundary",
			"doc_type_chunked":       "chunked",
		},
	})
	chunker := NewChunkingProcessor(nil, store)
	docs, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		RawFileBytes: zipBytes,
		Identity:     &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// We expect 3 docs since there are 2 PDFs and 1 image (image will be flagged as corrupted/non-pdf but forwarded)
	if len(docs) != 3 {
		t.Errorf("Expected 3 processed documents, got %d", len(docs))
	}
}

func TestProcessZipFile_EmptyPath(t *testing.T) {
	chunker := NewChunkingProcessor(nil, nil)
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{})
	if err == nil {
		t.Fatal("Expected error for empty zip path, got nil")
	}
	if !strings.Contains(err.Error(), "document_uri missing from storage payload") {
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
	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "bad.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
		},
	})
	chunker := NewChunkingProcessor(nil, store)
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})
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
	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "slip.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
			"action_name_classify":   "action",
			"doc_type_classified":    "classified",
			"action_name_boundary":   "boundary",
			"doc_type_chunked":       "chunked",
		},
	})
	chunker := NewChunkingProcessor(nil, store)
	docs, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		RawFileBytes: zipBytes,
		Identity:     &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})
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

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "dummy.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
		},
	})
	_, _, err := chunker.ProcessZipFile(ctx, &pb.ChunkDocumentRequest{
		RawFileBytes: []byte("dummy"),
		Identity:     &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})
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

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "missing_from_storage.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
		},
	})
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})

	if err == nil {
		t.Fatal("Expected error due to missing file, got nil")
	}
	if err.Error() == "no file data provided in request" {
		t.Errorf("Expected storage fetch error, got missing data error")
	}
}

func TestProcessZipFile_MissingBoundaryDetectionPrompt(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "multi.zip")

	createTestZip(t, map[string]string{
		"doc1.pdf": "dummy pdf content",
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	zipBytes, _ := os.ReadFile(zipPath)
	// Create mock AiClient non-nil so sliding window attempts AI call
	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "multi.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split":    float64(1),
			"sliding_window_size":       float64(5),
			"doc_type_corrupted":        "corrupted",
			"boundary_detection_prompt": "",
		},
	})
	chunker := NewChunkingProcessor(&mockAiClient{}, store)
	docs, _, _ := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		RawFileBytes: zipBytes,
		Identity:     &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})
	// Validate that zip processing completed and handled invalid/dummy PDF gracefully
	if len(docs) != 1 {
		t.Errorf("Expected 1 processed document (corrupted/dummy PDF handled), got %d", len(docs))
	}
}

func TestProcessZipFile_EnrichedErrorFormat(t *testing.T) {
	tempDir := t.TempDir()
	store := &mockStorage{basePath: tempDir}
	chunker := NewChunkingProcessor(nil, store)

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "internal-store://tenant/app/job/missing.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
		},
	})
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})

	if err == nil {
		t.Fatal("Expected error for missing storage file, got nil")
	}

	expectedPrefix := "[Step: JIT_FETCH, URI: internal-store://tenant/app/job/missing.zip]"
	if !strings.HasPrefix(err.Error(), expectedPrefix) {
		t.Errorf("Expected error prefix %q, got %q", expectedPrefix, err.Error())
	}
}

// syntheticTextPDF is a valid minimal 1-page PDF containing text stream operators (BT ... ET and Tj)
const syntheticTextPDF = `%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000236 00000 n 
0000000305 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
399
%%EOF`

// syntheticScannedPDF is a valid minimal 1-page PDF containing an image XObject (/Im1 Do) and NO text operators
const syntheticScannedPDF = `%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im1 6 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< >>
endobj
5 0 obj
<< /Length 17 >>
stream
q
1 0 0 1 0 0 cm
/Im1 Do
Q
endstream
endobj
6 0 obj
<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>
stream
ÿÿÿ
endstream
endobj
xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000234 00000 n 
0000000255 00000 n 
0000000322 00000 n 
trailer
<< /Size 7 /Root 1 0 R >>
startxref
460
%%EOF`

func createSyntheticPDF(t *testing.T, dir string, filename string, content string) string {
	t.Helper()
	filePath := filepath.Join(dir, filename)
	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		t.Fatalf("Failed to create synthetic PDF %s: %v", filename, err)
	}
	return filePath
}

func TestDetectIsScannedPDF(t *testing.T) {
	tempDir := t.TempDir()
	textPdfPath := createSyntheticPDF(t, tempDir, "text.pdf", syntheticTextPDF)
	scannedPdfPath := createSyntheticPDF(t, tempDir, "scanned.pdf", syntheticScannedPDF)
	corruptedPath := filepath.Join(tempDir, "corrupted.pdf")
	err := os.WriteFile(corruptedPath, []byte("invalid pdf content header"), 0644)
	if err != nil {
		t.Fatalf("failed to write corrupted doc: %v", err)
	}

	tests := []struct {
		name          string
		filePath      string
		wantIsScanned bool
		wantErr       bool
	}{
		{
			name:          "StructuredTextPDF_ReturnsFalse",
			filePath:      textPdfPath,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name:          "ScannedImagePDF_ReturnsTrue",
			filePath:      scannedPdfPath,
			wantIsScanned: true,
			wantErr:       false,
		},
		{
			name:          "CorruptedFile_ReturnsError",
			filePath:      corruptedPath,
			wantIsScanned: false,
			wantErr:       true,
		},
		{
			name:          "NonExistentFile_ReturnsError",
			filePath:      filepath.Join(tempDir, "does_not_exist.pdf"),
			wantIsScanned: false,
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotIsScanned, err := detectIsScannedPDF(tt.filePath)
			if (err != nil) != tt.wantErr {
				t.Fatalf("detectIsScannedPDF(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
			}
			if !tt.wantErr && gotIsScanned != tt.wantIsScanned {
				t.Errorf("detectIsScannedPDF(%s) gotIsScanned = %v, want %v", tt.name, gotIsScanned, tt.wantIsScanned)
			}
		})
	}
}

func TestProcessZipFile_PDFTypeDetection(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "mixed_type.zip")

	createTestZip(t, map[string]string{
		"text_doc.pdf":    syntheticTextPDF,
		"scanned_doc.pdf": syntheticScannedPDF,
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	zipBytes, err := os.ReadFile(zipPath)
	if err != nil {
		t.Fatalf("Failed to read test zip: %v", err)
	}

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "mixed_type.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
			"action_name_classify":   "action",
			"doc_type_classified":    "classified",
			"action_name_boundary":   "boundary",
			"doc_type_chunked":       "chunked",
		},
	})
	chunker := NewChunkingProcessor(nil, store)
	req := &pb.ExecuteTaskRequest{
		Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
		Payload:     payload,
	}
	docs, traces, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		RawFileBytes:    zipBytes,
		Identity:        &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "j1"},
		OriginalRequest: req,
	})
	if err != nil {
		t.Fatalf("ProcessZipFile failed: %v", err)
	}

	if len(docs) != 2 {
		t.Fatalf("Expected 2 processed documents, got %d", len(docs))
	}

	for _, doc := range docs {
		if strings.Contains(doc.RelativeFilePath, "text_doc") {
			if doc.IsScanned {
				t.Errorf("Expected text_doc.pdf to have IsScanned = false, got true")
			}
		} else if strings.Contains(doc.RelativeFilePath, "scanned_doc") {
			if !doc.IsScanned {
				t.Errorf("Expected scanned_doc.pdf to have IsScanned = true, got false")
			}
		}
	}

	var foundTextTrace, foundScannedTrace bool
	for _, tr := range traces {
		if strings.Contains(tr.Message, "has embedded text. Routing to Standard LLM.") {
			foundTextTrace = true
		}
		if strings.Contains(tr.Message, "appears to be a scanned document. Routing to OCR Vision LLM.") {
			foundScannedTrace = true
		}
	}
	if !foundTextTrace {
		t.Errorf("Trace events missing embedded text routing message")
	}
	if !foundScannedTrace {
		t.Errorf("Trace events missing scanned document routing message")
	}
}

func TestUnescapePDFLiteralString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "EmptyString",
			input:    "",
			expected: "",
		},
		{
			name:     "PlainString",
			input:    "Hello World",
			expected: "Hello World",
		},
		{
			name:     "StandardEscapes",
			input:    `\n\r\t\b\f\(\)\\`,
			expected: "\n\r\t\b\f()\\",
		},
		{
			name:     "Octal1Digit",
			input:    `\0`,
			expected: "\x00",
		},
		{
			name:     "Octal2Digits",
			input:    `\12`,
			expected: "\n",
		},
		{
			name:     "Octal3Digits",
			input:    `\101`,
			expected: "A",
		},
		{
			name:     "OctalFollowedByNonOctal",
			input:    `\12X`,
			expected: "\nX",
		},
		{
			name:     "LineContinuationCRLF",
			input:    "line1\\\r\nline2",
			expected: "line1line2",
		},
		{
			name:     "LineContinuationLF",
			input:    "line1\\\nline2",
			expected: "line1line2",
		},
		{
			name:     "UnknownEscapeSequence",
			input:    `\z`,
			expected: "z",
		},
		{
			name:     "TrailingBackslash",
			input:    `abc\`,
			expected: "abc",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := unescapePDFLiteralString(tt.input)
			if got != tt.expected {
				t.Errorf("unescapePDFLiteralString(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}
