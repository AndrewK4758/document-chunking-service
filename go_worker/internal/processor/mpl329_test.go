package processor

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/structpb"
)

type mockCppMlWorkerForMPL329 struct {
	pb.UnimplementedCppMlWorkerServiceServer
	response *pb.RedactDocumentResponse
	err      error
}

func (m *mockCppMlWorkerForMPL329) RedactDocument(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.response, nil
}

func createMinimalValidPDF() string {
	return `%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj
4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
5 0 obj << /Length 44 >> stream
BT
/F1 12 Tf
50 700 Td
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
0000000242 00000 n 
0000000313 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
407
%%EOF`
}

func createMPL329TestPayload() *structpb.Struct {
	payload, _ := structpb.NewStruct(map[string]any{
		"node_config": map[string]any{},
		"global_state": map[string]any{
			"functional": map[string]any{},
			"execution":  map[string]any{},
		},
		"storage": map[string]any{
			"document_uri": "test.zip",
		},
		"document_processor": map[string]any{
			"max_pages_before_split":    float64(5),
			"sliding_window_size":       float64(2),
			"doc_type_classified":       "SINGLE_DOC",
			"doc_type_chunked":          "CHUNKED_DOC",
			"doc_type_corrupted":        "CORRUPTED_DOC",
			"action_name_classify":      "classify",
			"action_name_boundary":      "boundary",
			"boundary_detection_prompt": "detect",
		},
	})
	return payload
}

func TestMPL329_GoWorker_ResidualSweepReport_Converged(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "test.zip")
	createTestZip(t, map[string]string{
		"doc1.pdf": createMinimalValidPDF(),
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	mockClient := &mockCppMlWorkerForMPL329{
		response: &pb.RedactDocumentResponse{
			Success: true,
			ResidualReport: &pb.ResidualSweepReport{
				IsConverged:        true,
				ResidualValueCount: 0,
				PassCount:          1,
			},
		},
	}

	proc := NewChunkingProcessor(nil, store)
	proc.SetCppMlClient(mockClient)

	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{
			TenantId: "tenant-1",
			AppId:    "app-1",
			JobId:    "job-mpl329-01",
		},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Payload: createMPL329TestPayload(),
		},
	}

	docs, traces, err := proc.ProcessZipFile(context.Background(), req)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("Expected 1 processed doc, got %d", len(docs))
	}

	var foundConvergedTrace bool
	for _, tr := range traces {
		if strings.Contains(tr.Message, "[SWEEP: CONVERGED]") {
			foundConvergedTrace = true
			break
		}
	}
	if !foundConvergedTrace {
		t.Errorf("Expected [SWEEP: CONVERGED] trace event, got traces: %v", traces)
	}
}

func TestMPL329_GoWorker_ResidualSweepReport_Pass1_Delta(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "test.zip")
	createTestZip(t, map[string]string{
		"doc1.pdf": createMinimalValidPDF(),
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	mockClient := &mockCppMlWorkerForMPL329{
		response: &pb.RedactDocumentResponse{
			Success: true,
			ResidualReport: &pb.ResidualSweepReport{
				IsConverged:        false,
				ResidualValueCount: 2,
				ResidualSamples:    []string{"1***5:10"},
				PassCount:          1,
			},
		},
	}

	proc := NewChunkingProcessor(nil, store)
	proc.SetCppMlClient(mockClient)

	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{
			TenantId: "tenant-1",
			AppId:    "app-1",
			JobId:    "job-mpl329-02",
		},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Payload: createMPL329TestPayload(),
		},
	}

	docs, traces, err := proc.ProcessZipFile(context.Background(), req)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("Expected 1 processed doc, got %d", len(docs))
	}

	var foundDeltaTrace bool
	for _, tr := range traces {
		if strings.Contains(tr.Message, "[SWEEP: DELTA]") {
			foundDeltaTrace = true
			break
		}
	}
	if !foundDeltaTrace {
		t.Errorf("Expected [SWEEP: DELTA] trace event, got traces: %v", traces)
	}
}

func TestMPL329_GoWorker_ResidualSweepReport_Pass3_FailFastLineageGap(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "test.zip")
	createTestZip(t, map[string]string{
		"doc1.pdf": createMinimalValidPDF(),
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	mockClient := &mockCppMlWorkerForMPL329{
		response: &pb.RedactDocumentResponse{
			Success: true,
			ResidualReport: &pb.ResidualSweepReport{
				IsConverged:        false,
				ResidualValueCount: 3,
				ResidualSamples:    []string{"S***N:11"},
				PassCount:          3,
			},
		},
	}

	proc := NewChunkingProcessor(nil, store)
	proc.SetCppMlClient(mockClient)

	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{
			TenantId: "tenant-1",
			AppId:    "app-1",
			JobId:    "job-mpl329-03",
		},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Payload: createMPL329TestPayload(),
		},
	}

	_, _, err := proc.ProcessZipFile(context.Background(), req)
	if err == nil {
		t.Fatalf("Expected error for pass 3 unconverged state, got nil")
	}
	if !strings.Contains(err.Error(), "[SEAM: LINEAGE_GAP]") {
		t.Errorf("Expected [SEAM: LINEAGE_GAP] error, got: %v", err)
	}
}
