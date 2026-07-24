package processor

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"document_processing_service/internal/archive"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/pdfcpu/pdfcpu/pkg/api"
	"github.com/pdfcpu/pdfcpu/pkg/pdfcpu/model"
)

type ChunkingProcessor struct {
	AiClient pb.DocumentProcessorClient
	Store    storage.Manager
}

func NewChunkingProcessor(aiClient pb.DocumentProcessorClient, store storage.Manager) *ChunkingProcessor {
	return &ChunkingProcessor{
		AiClient: aiClient,
		Store:    store,
	}
}

func (p *ChunkingProcessor) ProcessZipFile(ctx context.Context, req *pb.ChunkDocumentRequest) ([]*pb.ProcessedDocument, []*pb.ChunkingTraceEvent, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	var traceEvents []*pb.ChunkingTraceEvent
	addTrace := func(msg string, evtType string) {
		traceEvents = append(traceEvents, &pb.ChunkingTraceEvent{
			Timestamp: time.Now().Format(time.RFC3339),
			Message:   msg,
			EventType: evtType,
		})
	}

	var zipBytes []byte
	var err error
	if len(req.RawFileBytes) > 0 {
		addTrace("Using raw file bytes from request", "INFO")
		zipBytes = req.RawFileBytes
	} else if req.DocumentChunking != nil && req.DocumentChunking.DocumentStorageUri != "" {
		addTrace(fmt.Sprintf("Fetching %s from store", req.DocumentChunking.DocumentStorageUri), "INFO")
		if p.Store == nil {
			return nil, traceEvents, fmt.Errorf("storage manager is not configured")
		}
		zipBytes, err = p.Store.FetchFile(ctx, req.DocumentChunking.DocumentStorageUri, req.Identity)
		if err != nil {
			addTrace(fmt.Sprintf("Failed to fetch file: %v", err), "ERROR")
			return nil, traceEvents, fmt.Errorf("failed to fetch from store: %w", err)
		}
	} else {
		return nil, traceEvents, fmt.Errorf("no file data provided in request")
	}

	extractDir := filepath.Join(os.TempDir(), fmt.Sprintf("chunking_job_%s_%d", req.GetIdentity().GetJobId(), time.Now().UnixNano()))
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return nil, traceEvents, fmt.Errorf("failed to create extraction directory: %w", err)
	}
	defer os.RemoveAll(extractDir)

	// Determine original filename to preserve extension
	originalFilename := "input.pdf"
	if req.DocumentChunking != nil && req.DocumentChunking.DocumentStorageUri != "" {
		originalFilename = filepath.Base(req.DocumentChunking.DocumentStorageUri)
	}

	tempInputPath := filepath.Join(extractDir, originalFilename)
	if err := os.WriteFile(tempInputPath, zipBytes, 0644); err != nil {
		return nil, traceEvents, fmt.Errorf("failed to write temp input file: %w", err)
	}

	extractedPaths, err := archive.Extract(tempInputPath, extractDir)
	if err != nil {
		addTrace(fmt.Sprintf("Failed to extract archive: %v", err), "ERROR")
		return nil, traceEvents, fmt.Errorf("failed to extract archive: %w", err)
	}

	var extractedFiles []string
	extractedFiles = append(extractedFiles, extractedPaths...)

	addTrace(fmt.Sprintf("Extracted %d documents", len(extractedFiles)), "INFO")

	maxPagesEnv := os.Getenv("MAX_PAGES_BEFORE_SPLIT")
	maxPagesBeforeSplit := 20
	if val, err := strconv.Atoi(maxPagesEnv); err == nil && val > 0 {
		maxPagesBeforeSplit = val
	}

	var processedDocs []*pb.ProcessedDocument

	for _, pdfPath := range extractedFiles {
		var docTypeCorrupted string
		if req.DocumentChunking != nil && req.DocumentChunking.DocTypeCorrupted != "" {
			docTypeCorrupted = req.DocumentChunking.DocTypeCorrupted
		}

		if err := api.ValidateFile(pdfPath, model.NewDefaultConfiguration()); err != nil {
			addTrace(fmt.Sprintf("PDF Validation failed for %s (possible encryption or corruption): %v", filepath.Base(pdfPath), err), "WARN")
			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType:     docTypeCorrupted,
				RelativeFilePath: pdfPath,
				PageCount:        0,
			})
			continue
		}

		pageCount, err := api.PageCountFile(pdfPath)
		if err != nil {
			addTrace(fmt.Sprintf("Failed to get page count for %s: %v", filepath.Base(pdfPath), err), "ERROR")
			continue
		}

		addTrace(fmt.Sprintf("Analyzed %s (Pages: %d)", filepath.Base(pdfPath), pageCount), "INFO")

		if err := ctx.Err(); err != nil {
			return processedDocs, traceEvents, err
		}

		// Probe text vs OCR
		err = api.ExtractContentFile(pdfPath, extractDir, []string{"1"}, model.NewDefaultConfiguration())
		isStructuredText := err == nil

		modelChoice := pb.ModelChoice_MODEL_CHOICE_OCR
		if isStructuredText {
			modelChoice = pb.ModelChoice_MODEL_CHOICE_CLASSIFICATION
			addTrace(fmt.Sprintf("%s has embedded text. Routing to Standard LLM.", filepath.Base(pdfPath)), "INFO")
		} else {
			addTrace(fmt.Sprintf("%s appears to be a scanned document. Routing to OCR Vision LLM.", filepath.Base(pdfPath)), "INFO")
		}

		if pageCount <= maxPagesBeforeSplit {
			inputBytes, err := os.ReadFile(pdfPath)
			if err != nil {
				addTrace(fmt.Sprintf("Failed to read PDF file %s: %v", filepath.Base(pdfPath), err), "ERROR")
				return nil, traceEvents, fmt.Errorf("failed to read pdf file: %w", err)
			}

			var actionName string
			if req.DocumentChunking != nil && req.DocumentChunking.ActionNameClassify != "" {
				actionName = req.DocumentChunking.ActionNameClassify
			}

			if p.AiClient != nil && req.OriginalRequest != nil {
				// Construct action definition based on the client's original node config
				sysContext := ""
				if req.OriginalRequest.NodeConfig != nil {
					sysContext = req.OriginalRequest.NodeConfig.SystemInstructions
				}

				actionDef := &pb.ActionDefinition{
					Name:   actionName,
					Prompt: sysContext,
				}

				stream, err := p.AiClient.PerformAction(ctx)
				var resp *pb.PerformActionResponse
				if err == nil {
					err = stream.Send(&pb.PerformActionRequest{
						Context:       req.OriginalRequest,
						CurrentAction: actionDef,
						InputBytes:    inputBytes,
						ModelChoice:   modelChoice,
						Identity:      req.GetIdentity(),
					})
					if err == nil {
						stream.CloseSend()
						resp, err = stream.Recv()
					}
				}

				if err != nil {
					addTrace(fmt.Sprintf("AI Classification failed: %v", err), "ERROR")
				} else if resp != nil {
					addTrace(fmt.Sprintf("AI Classification successful: %s", resp.ActionResultJson), "INFO")
				}
			}

			// Upload the single document back to storage
			var relativePath string
			if p.Store != nil {
				uploadedPath, err := p.Store.SaveFile(ctx, fmt.Sprintf("%s/%s", req.GetIdentity().GetJobId(), filepath.Base(pdfPath)), req.GetIdentity(), inputBytes)
				if err != nil {
					addTrace(fmt.Sprintf("Failed to upload classified doc: %v", err), "ERROR")
				} else {
					relativePath = uploadedPath
				}
			}

			var docTypeClassified string
			if req.DocumentChunking != nil && req.DocumentChunking.DocTypeClassified != "" {
				docTypeClassified = req.DocumentChunking.DocTypeClassified
			}

			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType:     docTypeClassified,
				RelativeFilePath: relativePath,
				PageCount:        int32(pageCount),
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
				if err := ctx.Err(); err != nil {
					return processedDocs, traceEvents, err
				}

				windowEnd := min(windowStart+slidingWindowSize-1, pageCount)

				addTrace(fmt.Sprintf("Evaluating slice pages %d-%d for boundaries", windowStart, windowEnd), "INFO")

				sliceTempPath := filepath.Join(extractDir, fmt.Sprintf("slice_%d_%d.pdf", windowStart, windowEnd))
				err := api.TrimFile(pdfPath, sliceTempPath, []string{fmt.Sprintf("%d-%d", windowStart, windowEnd)}, model.NewDefaultConfiguration())

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
					if req.DocumentChunking != nil && req.DocumentChunking.BoundaryDetectionPrompt != "" {
						prompt = req.DocumentChunking.BoundaryDetectionPrompt
					}

					var actionName string
					if req.DocumentChunking != nil && req.DocumentChunking.ActionNameBoundary != "" {
						actionName = req.DocumentChunking.ActionNameBoundary
					}

					actionDef := &pb.ActionDefinition{
						Name:   actionName,
						Prompt: prompt,
					}

					stream, err := p.AiClient.PerformAction(ctx)
					var resp *pb.PerformActionResponse
					if err == nil {
						err = stream.Send(&pb.PerformActionRequest{
							Context:       req.OriginalRequest,
							CurrentAction: actionDef,
							InputBytes:    sliceBytes,
							ModelChoice:   pb.ModelChoice_MODEL_CHOICE_OCR,
							Identity:      req.GetIdentity(),
						})
						if err == nil {
							stream.CloseSend()
							resp, err = stream.Recv()
						}
					}

					if err == nil && resp != nil && resp.Success {
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

				var docTypeChunked string
				if req.DocumentChunking != nil && req.DocumentChunking.DocTypeChunked != "" {
					docTypeChunked = req.DocumentChunking.DocTypeChunked
				}

				if boundaryPageInSlice > 0 {
					absoluteBoundary := windowStart + boundaryPageInSlice - 1
					if absoluteBoundary > currentDocStart && absoluteBoundary <= pageCount {
						// Split!
						chunkPath := filepath.Join(extractDir, fmt.Sprintf("chunk_%d_%d.pdf", currentDocStart, absoluteBoundary-1))
						err = api.TrimFile(pdfPath, chunkPath, []string{fmt.Sprintf("%d-%d", currentDocStart, absoluteBoundary-1)}, model.NewDefaultConfiguration())
						if err != nil {
							addTrace(fmt.Sprintf("Failed to extract chunk pages %d-%d: %v", currentDocStart, absoluteBoundary-1, err), "ERROR")
							return nil, traceEvents, fmt.Errorf("failed to extract chunk: %w", err)
						}

						addTrace(fmt.Sprintf("Boundary found at absolute page %d. Created chunk %d (pages %d-%d).", absoluteBoundary, chunkIndex, currentDocStart, absoluteBoundary-1), "INFO")

						var relativePath string
						if p.Store != nil {
							chunkBytes, _ := os.ReadFile(chunkPath)
							uploadedPath, err := p.Store.SaveFile(ctx, fmt.Sprintf("%s/chunk_%d.pdf", req.GetIdentity().GetJobId(), chunkIndex), req.GetIdentity(), chunkBytes)
							if err == nil {
								relativePath = uploadedPath
							}
						}

						processedDocs = append(processedDocs, &pb.ProcessedDocument{
							DocumentType:     docTypeChunked,
							RelativeFilePath: relativePath,
							PageCount:        int32(absoluteBoundary - currentDocStart),
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
				var docTypeChunked string
				if req.DocumentChunking != nil && req.DocumentChunking.DocTypeChunked != "" {
					docTypeChunked = req.DocumentChunking.DocTypeChunked
				}

				chunkPath := filepath.Join(extractDir, fmt.Sprintf("chunk_%d_%d.pdf", currentDocStart, pageCount))
				err := api.TrimFile(pdfPath, chunkPath, []string{fmt.Sprintf("%d-%d", currentDocStart, pageCount)}, model.NewDefaultConfiguration())
				if err != nil {
					addTrace(fmt.Sprintf("Failed to extract final chunk pages %d-%d: %v", currentDocStart, pageCount, err), "ERROR")
					return nil, traceEvents, fmt.Errorf("failed to extract final chunk: %w", err)
				}

				addTrace(fmt.Sprintf("Created final chunk %d (pages %d-%d).", chunkIndex, currentDocStart, pageCount), "INFO")

				var relativePath string
				if p.Store != nil {
					chunkBytes, _ := os.ReadFile(chunkPath)
					uploadedPath, err := p.Store.SaveFile(ctx, fmt.Sprintf("%s/chunk_%d.pdf", req.GetIdentity().GetJobId(), chunkIndex), req.GetIdentity(), chunkBytes)
					if err == nil {
						relativePath = uploadedPath
					}
				}

				processedDocs = append(processedDocs, &pb.ProcessedDocument{
					DocumentType:     docTypeChunked,
					RelativeFilePath: relativePath,
					PageCount:        int32(pageCount - currentDocStart + 1),
				})
			}
		}
	}

	return processedDocs, traceEvents, nil
}

func (p *ChunkingProcessor) GenerateBlankDocument(ctx context.Context, docUri string, identity *pb.InfrastructureIdentity) (string, error) {
	if p.Store == nil {
		return "", fmt.Errorf("storage manager is not configured")
	}

	pdfBytes, err := p.Store.FetchFile(ctx, docUri, identity)
	if err != nil {
		return "", fmt.Errorf("failed to fetch from store: %w", err)
	}

	// [MOCK] In a real implementation, we would shell out to Ghostscript or ImageMagick
	// to strip all text rendering streams and produce a purely visual 'blank' layout PDF.
	// We simulate this for now by uploading the original bytes with a 'blanked_' prefix.
	blankedFileName := fmt.Sprintf("blanked_%d.pdf", time.Now().UnixNano())
	blankUri, err := p.Store.SaveFile(ctx, fmt.Sprintf("%s/%s", identity.GetJobId(), blankedFileName), identity, pdfBytes)
	if err != nil {
		return "", fmt.Errorf("failed to save blank document to store: %w", err)
	}

	return blankUri, nil
}
