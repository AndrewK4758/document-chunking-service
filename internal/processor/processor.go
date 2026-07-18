package processor

import (
	"archive/zip"
	"context"
	"encoding/json"
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

func (p *ChunkingProcessor) ProcessZipFile(ctx context.Context, req *pb.ChunkDocumentRequest) ([]*pb.ProcessedDocument, []*pb.ChunkingTraceEvent, error) {
	if req.ZipFilePath == "" {
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

	addTrace(fmt.Sprintf("Starting zip extraction for %s", req.ZipFilePath), "INFO")

	extractDir := filepath.Join(filepath.Dir(req.ZipFilePath), fmt.Sprintf("extracted_%d", time.Now().UnixNano()))
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return nil, traceEvents, fmt.Errorf("failed to create extraction directory: %w", err)
	}

	r, err := zip.OpenReader(req.ZipFilePath)
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
		docTypeCorrupted := "Corrupted_Or_Encrypted_Document"
		if req.OriginalRequest != nil && req.OriginalRequest.Metadata != nil && req.OriginalRequest.Metadata["doc_type_corrupted"] != "" {
			docTypeCorrupted = req.OriginalRequest.Metadata["doc_type_corrupted"]
		}

		if err := api.ValidateFile(pdfPath, model.NewDefaultConfiguration()); err != nil {
			addTrace(fmt.Sprintf("PDF Validation failed for %s (possible encryption or corruption): %v", filepath.Base(pdfPath), err), "WARN")
			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType: docTypeCorrupted,
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
			modelChoice := pb.ModelChoice_MODEL_CHOICE_OCR
			inputBytes, err := os.ReadFile(pdfPath)
			if err != nil {
				addTrace(fmt.Sprintf("Failed to read PDF file %s: %v", filepath.Base(pdfPath), err), "ERROR")
				return nil, traceEvents, fmt.Errorf("failed to read pdf file: %w", err)
			}
			addTrace(fmt.Sprintf("%s routing to OCR Vision LLM.", filepath.Base(pdfPath)), "INFO")

			actionName := "ClassifyDocument"
			if req.OriginalRequest != nil && req.OriginalRequest.Metadata != nil && req.OriginalRequest.Metadata["action_name_classify"] != "" {
				actionName = req.OriginalRequest.Metadata["action_name_classify"]
			}

			if p.AiClient != nil && req.OriginalRequest != nil {
				// Construct action definition based on the client's original node config
				sysContext := ""
				if req.OriginalRequest.NodeConfig != nil {
					sysContext = req.OriginalRequest.NodeConfig.SystemInstructions
				}
				
				actionDef := &pb.ActionDefinition{
					Name: actionName,
					Prompt: sysContext,
				}

				resp, err := p.AiClient.PerformAction(ctx, &pb.PerformActionRequest{
					Context:       req.OriginalRequest,
					CurrentAction: actionDef,
					InputBytes:    inputBytes,
					ModelChoice:   modelChoice,
				})
				
				if err != nil {
					addTrace(fmt.Sprintf("AI Classification failed: %v", err), "ERROR")
				} else {
					addTrace(fmt.Sprintf("AI Classification successful: %s", resp.ActionResultJson), "INFO")
				}
			}

			docTypeClassified := "Classified_Document"
			if req.OriginalRequest != nil && req.OriginalRequest.Metadata != nil && req.OriginalRequest.Metadata["doc_type_classified"] != "" {
				docTypeClassified = req.OriginalRequest.Metadata["doc_type_classified"]
			}

			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType: docTypeClassified,
				FilePath:     pdfPath,
				PageCount:    int32(pageCount),
			})
		} else {
			// Large document, execute sliding window boundary detection
			addTrace(fmt.Sprintf("%s exceeds max pages (%d), engaging sliding window splitter...", filepath.Base(pdfPath), maxPagesBeforeSplit), "INFO")
			
			slidingWindowSize := 5
			if val, err := strconv.Atoi(os.Getenv("SLIDING_WINDOW_SIZE")); err == nil && val > 0 {
				slidingWindowSize = val
			}

			currentDocStart := 1
			windowStart := 1
			var chunkIndex int

			for windowStart <= pageCount {
				windowEnd := windowStart + slidingWindowSize - 1
				if windowEnd > pageCount {
					windowEnd = pageCount
				}

				addTrace(fmt.Sprintf("Evaluating slice pages %d-%d for boundaries", windowStart, windowEnd), "INFO")

				sliceTempPath := filepath.Join(extractDir, fmt.Sprintf("slice_%d_%d.pdf", windowStart, windowEnd))
				err := api.ExtractPagesFile(pdfPath, sliceTempPath, []string{fmt.Sprintf("%d-%d", windowStart, windowEnd)}, model.NewDefaultConfiguration())
				
				if err != nil {
					addTrace(fmt.Sprintf("Failed to extract slice pages %d-%d: %v", windowStart, windowEnd, err), "ERROR")
					windowStart = windowEnd + 1
					continue
				}

				boundaryPageInSlice := 0
				if p.AiClient != nil && req.OriginalRequest != nil {
					sliceBytes, err := os.ReadFile(sliceTempPath)
					if err != nil {
						addTrace(fmt.Sprintf("Failed to read slice temp file %s: %v", sliceTempPath, err), "ERROR")
						return nil, traceEvents, fmt.Errorf("failed to read slice file: %w", err)
					}
					
					prompt := "Analyze this document slice. Does a NEW document boundary begin somewhere within these pages? A document boundary is typically a new header, title page, or completely different formatting indicating a new file. Return a JSON object: {\"boundary_page_in_slice\": 3} where 3 is the page number (relative to this slice) the new document starts. If it is all one continuous document, return {\"boundary_page_in_slice\": 0}."
					if req.OriginalRequest.Metadata != nil && req.OriginalRequest.Metadata["boundary_detection_prompt"] != "" {
						prompt = req.OriginalRequest.Metadata["boundary_detection_prompt"]
					}

					actionName := "InternalBoundaryDetection"
					if req.OriginalRequest.Metadata != nil && req.OriginalRequest.Metadata["action_name_boundary"] != "" {
						actionName = req.OriginalRequest.Metadata["action_name_boundary"]
					}

					actionDef := &pb.ActionDefinition{
						Name: actionName,
						Prompt: prompt,
					}

					resp, err := p.AiClient.PerformAction(ctx, &pb.PerformActionRequest{
						Context:       req.OriginalRequest,
						CurrentAction: actionDef,
						InputBytes:    sliceBytes,
						ModelChoice:   pb.ModelChoice_MODEL_CHOICE_OCR,
					})
					
					if err == nil && resp.Success {
						var boundResp struct {
							BoundaryPageInSlice int `json:"boundary_page_in_slice"`
						}
						if json.Unmarshal([]byte(resp.ActionResultJson), &boundResp) == nil {
							boundaryPageInSlice = boundResp.BoundaryPageInSlice
						}
					} else {
						addTrace(fmt.Sprintf("AI Boundary Detection failed for slice %d-%d: %v", windowStart, windowEnd, err), "WARN")
					}
				}
				
				os.Remove(sliceTempPath)

				docTypeChunked := "Chunked_Document"
				if req.OriginalRequest != nil && req.OriginalRequest.Metadata != nil && req.OriginalRequest.Metadata["doc_type_chunked"] != "" {
					docTypeChunked = req.OriginalRequest.Metadata["doc_type_chunked"]
				}

				if boundaryPageInSlice > 0 {
					absoluteBoundary := windowStart + boundaryPageInSlice - 1
					if absoluteBoundary > currentDocStart && absoluteBoundary <= pageCount {
						// Split!
						chunkPath := filepath.Join(extractDir, fmt.Sprintf("chunk_%d_%d.pdf", currentDocStart, absoluteBoundary-1))
						err = api.ExtractPagesFile(pdfPath, chunkPath, []string{fmt.Sprintf("%d-%d", currentDocStart, absoluteBoundary-1)}, model.NewDefaultConfiguration())
						if err != nil {
							addTrace(fmt.Sprintf("Failed to extract chunk pages %d-%d: %v", currentDocStart, absoluteBoundary-1, err), "ERROR")
							return nil, traceEvents, fmt.Errorf("failed to extract chunk: %w", err)
						}
						
						addTrace(fmt.Sprintf("Boundary found at absolute page %d. Created chunk %d (pages %d-%d).", absoluteBoundary, chunkIndex, currentDocStart, absoluteBoundary-1), "INFO")
						
						processedDocs = append(processedDocs, &pb.ProcessedDocument{
							DocumentType: docTypeChunked,
							FilePath:     chunkPath,
							PageCount:    int32(absoluteBoundary - currentDocStart),
						})
						chunkIndex++
						
						currentDocStart = absoluteBoundary
						windowStart = absoluteBoundary
					} else {
						// Invalid boundary returned, ignore and move on
						windowStart = windowEnd + 1
					}
				} else {
					addTrace(fmt.Sprintf("No boundary found in pages %d-%d. Continuing document.", windowStart, windowEnd), "INFO")
					windowStart = windowEnd + 1
				}
			}

			// Handle EOF trailing document
			if currentDocStart <= pageCount {
				docTypeChunked := "Chunked_Document"
				if req.OriginalRequest != nil && req.OriginalRequest.Metadata != nil && req.OriginalRequest.Metadata["doc_type_chunked"] != "" {
					docTypeChunked = req.OriginalRequest.Metadata["doc_type_chunked"]
				}

				chunkPath := filepath.Join(extractDir, fmt.Sprintf("chunk_%d_%d.pdf", currentDocStart, pageCount))
				err := api.ExtractPagesFile(pdfPath, chunkPath, []string{fmt.Sprintf("%d-%d", currentDocStart, pageCount)}, model.NewDefaultConfiguration())
				if err != nil {
					addTrace(fmt.Sprintf("Failed to extract final chunk pages %d-%d: %v", currentDocStart, pageCount, err), "ERROR")
					return nil, traceEvents, fmt.Errorf("failed to extract final chunk: %w", err)
				}
				
				addTrace(fmt.Sprintf("Created final chunk %d (pages %d-%d).", chunkIndex, currentDocStart, pageCount), "INFO")
				processedDocs = append(processedDocs, &pb.ProcessedDocument{
					DocumentType: docTypeChunked,
					FilePath:     chunkPath,
					PageCount:    int32(pageCount - currentDocStart + 1),
				})
			}
		}
	}

	return processedDocs, traceEvents, nil
}

// probeText is not needed right now since we are deferring OCR vs Structured routing
// directly to the AI service depending on the payload.
