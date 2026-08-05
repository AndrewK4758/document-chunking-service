package processor

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestChallenger_ParallelDetectionAndGoroutineLeak(t *testing.T) {
	tempDir := t.TempDir()
	textPdfPath := createSyntheticPDF(t, tempDir, "text_parallel.pdf", syntheticTextPDF)
	scannedPdfPath := createSyntheticPDF(t, tempDir, "scanned_parallel.pdf", syntheticScannedPDF)

	runtime.GC()
	initialGoroutines := runtime.NumGoroutine()

	const concurrency = 100
	var wg sync.WaitGroup
	wg.Add(concurrency)

	errCh := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		go func(idx int) {
			defer wg.Done()
			path := textPdfPath
			expectedScanned := false
			if idx%2 == 1 {
				path = scannedPdfPath
				expectedScanned = true
			}

			isScanned, err := detectIsScannedPDF(path)
			if err != nil {
				errCh <- err
				return
			}
			if isScanned != expectedScanned {
				t.Errorf("Goroutine %d: expected isScanned=%v, got %v", idx, expectedScanned, isScanned)
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("Parallel detectIsScannedPDF error: %v", err)
	}

	runtime.GC()
	time.Sleep(50 * time.Millisecond)
	finalGoroutines := runtime.NumGoroutine()

	// Allow a small delta if Go internal runtime spawned background workers
	if finalGoroutines > initialGoroutines+2 {
		t.Errorf("Goroutine leak detected! Initial: %d, Final: %d", initialGoroutines, finalGoroutines)
	}
}

func TestChallenger_ParallelProcessZipFileAndGoroutineLeak(t *testing.T) {
	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "parallel.zip")

	createTestZip(t, map[string]string{
		"text_doc.pdf":    syntheticTextPDF,
		"scanned_doc.pdf": syntheticScannedPDF,
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	zipBytes, err := os.ReadFile(zipPath)
	if err != nil {
		t.Fatalf("Failed to read test zip: %v", err)
	}

	runtime.GC()
	initialGoroutines := runtime.NumGoroutine()

	const concurrency = 20
	var wg sync.WaitGroup
	wg.Add(concurrency)

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{
				"document_uri":           "parallel.zip",
				"max_pages_before_split": float64(10),
				"sliding_window_size":   float64(5),
				"doc_type_corrupted":    "corrupted",
				"action_name_classify":  "action",
				"doc_type_classified":   "classified",
				"action_name_boundary":  "boundary",
				"doc_type_chunked":     "chunked",
			},
		},
	})
	req := &pb.ExecuteTaskRequest{
		Identity: &pb.InfrastructureIdentity{
			TenantId: "t1",
			AppId:    "a1",
			JobId:    "job-parallel",
		},
		TaskSubject: "worker.doc.process",
		Payload:     payload,
	}

	for i := 0; i < concurrency; i++ {
		go func(idx int) {
			defer wg.Done()
			chunker := NewChunkingProcessor(nil, store)

			docs, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
				RawFileBytes:    zipBytes,
				OriginalRequest: req,
				Identity: &pb.InfrastructureIdentity{
					TenantId: "t1",
					AppId:    "a1",
					JobId:    "job-parallel",
				},
			})

			if err != nil {
				t.Errorf("Goroutine %d: ProcessZipFile failed: %v", idx, err)
				return
			}

			if len(docs) != 2 {
				t.Errorf("Goroutine %d: Expected 2 docs, got %d", idx, len(docs))
			}
		}(i)
	}

	wg.Wait()

	runtime.GC()
	time.Sleep(50 * time.Millisecond)
	finalGoroutines := runtime.NumGoroutine()

	if finalGoroutines > initialGoroutines+2 {
		t.Errorf("Goroutine leak in ProcessZipFile! Initial: %d, Final: %d", initialGoroutines, finalGoroutines)
	}
}

func TestChallenger_TempFileCleanupIsolation(t *testing.T) {
	sysTemp := os.TempDir()

	// Find pre-existing chunking temp dirs
	preEntries, err := os.ReadDir(sysTemp)
	if err != nil {
		t.Fatalf("Failed to read sys temp dir: %v", err)
	}
	preChunkDirs := make(map[string]bool)
	for _, entry := range preEntries {
		if strings.HasPrefix(entry.Name(), "chunking_job_") {
			preChunkDirs[entry.Name()] = true
		}
	}

	tempDir := t.TempDir()
	zipPath := filepath.Join(tempDir, "cleanup.zip")
	createTestZip(t, map[string]string{
		"doc1.pdf": syntheticTextPDF,
		"doc2.pdf": syntheticScannedPDF,
	}, zipPath)

	store := &mockStorage{basePath: tempDir}
	zipBytes, _ := os.ReadFile(zipPath)

	chunker := NewChunkingProcessor(nil, store)

	cleanupPayload, _ := structpb.NewStruct(map[string]interface{}{
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{
				"document_uri":           "cleanup.zip",
				"max_pages_before_split": float64(10),
				"sliding_window_size":   float64(5),
				"doc_type_corrupted":    "corrupted",
				"action_name_classify":  "action",
				"doc_type_classified":   "classified",
				"action_name_boundary":  "boundary",
				"doc_type_chunked":     "chunked",
			},
		},
	})
	cleanupReq := &pb.ExecuteTaskRequest{
		Identity: &pb.InfrastructureIdentity{
			TenantId: "t1",
			AppId:    "a1",
			JobId:    "job-cleanup",
		},
		TaskSubject: "worker.doc.process",
		Payload:     cleanupPayload,
	}

	for i := 0; i < 25; i++ {
		_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
			RawFileBytes:    zipBytes,
			OriginalRequest: cleanupReq,
			Identity: &pb.InfrastructureIdentity{
				TenantId: "t1",
				AppId:    "a1",
				JobId:    "job-cleanup",
			},
		})
		if err != nil {
			t.Fatalf("ProcessZipFile iteration %d failed: %v", i, err)
		}
	}

	postEntries, err := os.ReadDir(sysTemp)
	if err != nil {
		t.Fatalf("Failed to read sys temp dir after test: %v", err)
	}

	for _, entry := range postEntries {
		if strings.HasPrefix(entry.Name(), "chunking_job_") {
			if !preChunkDirs[entry.Name()] {
				t.Errorf("Temp file leak detected! Leftover dir in os.TempDir(): %s", entry.Name())
			}
		}
	}
}

// Synthetic PDF with Hex Text operator: <48656c6c6f> Tj
const syntheticHexPDF = `%PDF-1.4
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
<< /Length 43 >>
stream
BT
/F1 12 Tf
100 700 Td
<48656c6c6f> Tj
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
398
%%EOF`

// Synthetic PDF with Array TJ operator: [(Hello) 10 (World)] TJ
const syntheticArrayTJPDF = `%PDF-1.4
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
<< /Length 53 >>
stream
BT
/F1 12 Tf
100 700 Td
[(Hello) 10 (World)] TJ
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
408
%%EOF`

// Synthetic PDF with BT ... ET block containing only spaces: (   ) Tj
const syntheticSpacesPDF = `%PDF-1.4
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
<< /Length 39 >>
stream
BT
/F1 12 Tf
100 700 Td
(   ) Tj
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
394
%%EOF`

func TestChallenger_PDFOperatorEdgeCases(t *testing.T) {
	tempDir := t.TempDir()

	hexPdf := createSyntheticPDF(t, tempDir, "hex.pdf", syntheticHexPDF)
	arrayTjPdf := createSyntheticPDF(t, tempDir, "array_tj.pdf", syntheticArrayTJPDF)
	spacesPdf := createSyntheticPDF(t, tempDir, "spaces.pdf", syntheticSpacesPDF)

	// Hex string <48656c6c6f> Tj => "Hello" => isScanned = false
	isScannedHex, err := detectIsScannedPDF(hexPdf)
	if err != nil {
		t.Fatalf("detectIsScannedPDF(hexPdf) failed: %v", err)
	}
	if isScannedHex {
		t.Errorf("Expected isScannedHex = false for hex string text, got true")
	}

	// Array TJ [(Hello) 10 (World)] TJ => isScanned = false
	isScannedArray, err := detectIsScannedPDF(arrayTjPdf)
	if err != nil {
		t.Fatalf("detectIsScannedPDF(arrayTjPdf) failed: %v", err)
	}
	if isScannedArray {
		t.Errorf("Expected isScannedArray = false for array TJ text, got true")
	}

	// Spaces only (   ) Tj => isScanned = true (no printable non-space text)
	isScannedSpaces, err := detectIsScannedPDF(spacesPdf)
	if err != nil {
		t.Fatalf("detectIsScannedPDF(spacesPdf) failed: %v", err)
	}
	if !isScannedSpaces {
		t.Errorf("Expected isScannedSpaces = true for spaces-only text block, got false")
	}
}
