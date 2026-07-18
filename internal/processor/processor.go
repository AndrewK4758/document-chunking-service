package processor

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/pdfcpu/pdfcpu/pkg/api"
	"github.com/pdfcpu/pdfcpu/pkg/pdfcpu/model"
	pb "github.com/AndrewK4758/shared_protos"
)

type ChunkingProcessor struct {
	AiClient pb.DocumentProcessorClient
}

func NewChunkingProcessor(aiClient pb.DocumentProcessorClient) *ChunkingProcessor {
	return &ChunkingProcessor{
		AiClient: aiClient,
	}
}

func (p *ChunkingProcessor) ProcessZipFile(ctx context.Context, zipPath string) ([]*pb.ProcessedDocument, []*pb.ChunkingTraceEvent, error) {
	if zipPath == "" {
		return nil, nil, fmt.Errorf("zip path cannot be empty")
	}

	var traceEvents []*pb.ChunkingTraceEvent
	addTrace := func(msg string, evtType string) {
		traceEvents = append(traceEvents, &pb.ChunkingTraceEvent{
			Timestamp: time.Now().Format(time.RFC3339),
			Message:   msg,
			EventType: evtType,
		})
	}

	addTrace(fmt.Sprintf("Starting zip extraction for %s", zipPath), "INFO")

	extractDir := filepath.Join(filepath.Dir(zipPath), fmt.Sprintf("extracted_%d", time.Now().UnixNano()))
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return nil, traceEvents, fmt.Errorf("failed to create extraction directory: %w", err)
	}

	r, err := zip.OpenReader(zipPath)
	if err != nil {
		addTrace(fmt.Sprintf("Failed to open zip file: %v", err), "ERROR")
		return nil, traceEvents, fmt.Errorf("failed to open zip file: %w", err)
	}
	defer r.Close()

	var extractedFiles []string
	for _, f := range r.File {
		if strings.Contains(f.Name, "..") {
			addTrace(fmt.Sprintf("Skipping malicious zip path: %s", f.Name), "WARN")
			continue
		}

		extractedFilePath := filepath.Join(extractDir, f.Name)
		if f.FileInfo().IsDir() {
			os.MkdirAll(extractedFilePath, f.Mode())
			continue
		}

		if err := os.MkdirAll(filepath.Dir(extractedFilePath), 0755); err != nil {
			return nil, traceEvents, fmt.Errorf("failed to create directory for file %s: %w", f.Name, err)
		}

		dstFile, err := os.OpenFile(extractedFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return nil, traceEvents, fmt.Errorf("failed to create extracted file %s: %w", f.Name, err)
		}

		srcFile, err := f.Open()
		if err != nil {
			dstFile.Close()
			return nil, traceEvents, fmt.Errorf("failed to open zip content %s: %w", f.Name, err)
		}

		_, err = io.Copy(dstFile, srcFile)
		dstFile.Close()
		srcFile.Close()
		if err != nil {
			return nil, traceEvents, fmt.Errorf("failed to write extracted file %s: %w", f.Name, err)
		}

		if strings.HasSuffix(strings.ToLower(f.Name), ".pdf") {
			extractedFiles = append(extractedFiles, extractedFilePath)
		}
	}

	addTrace(fmt.Sprintf("Extracted %d PDF files", len(extractedFiles)), "INFO")

	maxPagesEnv := os.Getenv("MAX_PAGES_BEFORE_SPLIT")
	maxPagesBeforeSplit := 20
	if val, err := strconv.Atoi(maxPagesEnv); err == nil && val > 0 {
		maxPagesBeforeSplit = val
	}

	var processedDocs []*pb.ProcessedDocument

	for _, pdfPath := range extractedFiles {
		// 1. Validate PDF and get page count
		if err := api.ValidateFile(pdfPath, model.NewDefaultConfiguration()); err != nil {
			addTrace(fmt.Sprintf("PDF Validation failed for %s (possible encryption or corruption): %v", filepath.Base(pdfPath), err), "WARN")
			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType: "Corrupted_Or_Encrypted_Document",
				FilePath:     pdfPath,
				PageCount:    0,
			})
			continue
		}

		pageCount, err := api.PageCountFile(pdfPath)
		if err != nil {
			addTrace(fmt.Sprintf("Failed to get page count for %s: %v", filepath.Base(pdfPath), err), "ERROR")
			continue
		}

		addTrace(fmt.Sprintf("Analyzed %s (Pages: %d)", filepath.Base(pdfPath), pageCount), "INFO")

		if pageCount <= maxPagesBeforeSplit {
			// Small document, no splitting needed. Probe text to see if Scanned vs Structured
			hasText := probeText(pdfPath)
			modelChoice := pb.ModelChoice_MODEL_CHOICE_CLASSIFICATION
			if !hasText {
				modelChoice = pb.ModelChoice_MODEL_CHOICE_OCR
				addTrace(fmt.Sprintf("%s appears to be a scanned document. Routing to OCR.", filepath.Base(pdfPath)), "INFO")
			}

			// TODO: Execute AI classification here using p.AiClient.PerformAction(..., modelChoice)
			_ = modelChoice

			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType: "Classified_Document",
				FilePath:     pdfPath,
				PageCount:    int32(pageCount),
			})
		} else {
			// Large document, execute sliding window boundary detection
			addTrace(fmt.Sprintf("%s exceeds max pages (%d), engaging sliding window splitter...", filepath.Base(pdfPath), maxPagesBeforeSplit), "INFO")
			// ... Implementation of splitting algorithm ...
			
			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType: "Chunked_Document",
				FilePath:     pdfPath,
				PageCount:    int32(pageCount),
			})
		}
	}

	return processedDocs, traceEvents, nil
}

func probeText(pdfPath string) bool {
	// A lightweight text extraction check using pdfcpu
	// If it returns an empty string or extremely few chars relative to pages, it's a scanned doc
	// For now, we return true (structured) as a default stub
	return true
}
