package processor

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

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

func (p *ChunkingProcessor) ProcessZipFile(ctx context.Context, zipPath string) ([]*pb.ProcessedDocument, error) {
	if zipPath == "" {
		return nil, fmt.Errorf("zip path cannot be empty")
	}

	// 1. Create a temporary extraction directory
	extractDir := filepath.Join(filepath.Dir(zipPath), fmt.Sprintf("extracted_%d", time.Now().UnixNano()))
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create extraction directory: %w", err)
	}

	// 2. Open and extract the ZIP file
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open zip file: %w", err)
	}
	defer r.Close()

	var extractedFiles []string
	for _, f := range r.File {
		// Prevent Zip Slip vulnerability
		if strings.Contains(f.Name, "..") {
			continue
		}

		extractedFilePath := filepath.Join(extractDir, f.Name)

		if f.FileInfo().IsDir() {
			os.MkdirAll(extractedFilePath, f.Mode())
			continue
		}

		if err := os.MkdirAll(filepath.Dir(extractedFilePath), 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory for file %s: %w", f.Name, err)
		}

		dstFile, err := os.OpenFile(extractedFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return nil, fmt.Errorf("failed to create extracted file %s: %w", f.Name, err)
		}

		srcFile, err := f.Open()
		if err != nil {
			dstFile.Close()
			return nil, fmt.Errorf("failed to open zip content %s: %w", f.Name, err)
		}

		_, err = io.Copy(dstFile, srcFile)
		dstFile.Close()
		srcFile.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to write extracted file %s: %w", f.Name, err)
		}

		if strings.HasSuffix(strings.ToLower(f.Name), ".pdf") {
			extractedFiles = append(extractedFiles, extractedFilePath)
		}
	}

	// 3. Process each extracted PDF (Mocked AI classification and page count for now)
	var processedDocs []*pb.ProcessedDocument
	for i, pdfPath := range extractedFiles {
		// TODO: Call AI Service to classify and chunk using pdfcpu to extract text/pages
		// For now, we simulate the output

		processedDocs = append(processedDocs, &pb.ProcessedDocument{
			DocumentType: "Generic_Document",
			FilePath:     pdfPath,
			PageCount:    int32(1 + i), // Stub page count
		})
	}

	return processedDocs, nil
}
