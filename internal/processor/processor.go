package processor

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/AndrewK4758/shared_utils/envutil"
	"os"
	"path/filepath"
	"time"

	"document_processing_service/internal/archive"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/workflowparser"
	"github.com/pdfcpu/pdfcpu/pkg/api"
	"github.com/pdfcpu/pdfcpu/pkg/pdfcpu/model"
)

type BoundaryDetectionResponse struct {
	BoundaryPageInSlice int `json:"boundary_page_in_slice"`
}

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

	parser := workflowparser.NewNodeRequestParser(req.GetOriginalRequest())

	if len(req.RawFileBytes) > 0 {
		addTrace("Using raw file bytes from request", "INFO")
		zipBytes = req.RawFileBytes
	} else if parser.GetDocumentUri() != "" {
		addTrace(fmt.Sprintf("Fetching %s from store", parser.GetDocumentUri()), "INFO")
		if p.Store == nil {
			return nil, traceEvents, fmt.Errorf("storage manager is not configured")
		}
		zipBytes, err = p.Store.FetchFile(ctx, parser.GetDocumentUri(), req.Identity)
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
	defer func(path string) {
		err := os.RemoveAll(path)
		if err != nil {

		}
	}(extractDir)

	originalFilename := "input.pdf"
	if parser.GetDocumentUri() != "" {
		originalFilename = filepath.Base(parser.GetDocumentUri())
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

	maxPagesBeforeSplit := envutil.MustGetInt("MAX_PAGES_BEFORE_SPLIT")

	var processedDocs []*pb.ProcessedDocument

	for _, pdfPath := range extractedFiles {
		docTypeCorrupted := parser.GetDocumentProcessorContract().GetDocTypeCorrupted()
		if docTypeCorrupted == "" {
			return nil, traceEvents, fmt.Errorf("DocTypeCorrupted must be configured in DocumentProcessorContract")
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

			actionName := parser.GetDocumentProcessorContract().GetActionNameClassify()
			if actionName == "" {
				return nil, traceEvents, fmt.Errorf("ActionNameClassify must be configured in DocumentProcessorContract")
			}

			if p.AiClient != nil && req.OriginalRequest != nil {
				// Construct action definition based on the client's original node config
				sysContext := parser.GetSystemInstructions()

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

			docTypeClassified := parser.GetDocumentProcessorContract().GetDocTypeClassified()
			if docTypeClassified == "" {
				return nil, traceEvents, fmt.Errorf("DocTypeClassified must be configured in DocumentProcessorContract")
			}

			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType:     docTypeClassified,
				RelativeFilePath: relativePath,
				PageCount:        int32(pageCount),
			})
		} else {
			// Large document, execute sliding window boundary detection
			addTrace(fmt.Sprintf("%s exceeds max pages (%d), engaging sliding window splitter...", filepath.Base(pdfPath), maxPagesBeforeSplit), "INFO")

			slidingWindowSize := envutil.MustGetInt("SLIDING_WINDOW_SIZE")

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

					prompt := parser.GetDocumentProcessorContract().GetBoundaryDetectionPrompt()
					if prompt == "" {
						addTrace("Boundary detection prompt missing or empty in request", "ERROR")
						err := os.Remove(sliceTempPath)
						if err != nil {
							return nil, nil, err
						}
						return nil, traceEvents, fmt.Errorf("boundary detection prompt is required but not provided in request")
					}

					actionName := parser.GetDocumentProcessorContract().GetActionNameBoundary()
					if actionName == "" {
						return nil, traceEvents, fmt.Errorf("ActionNameBoundary must be configured in DocumentProcessorContract")
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
							err := stream.CloseSend()
							if err != nil {
								return nil, nil, err
							}
							resp, err = stream.Recv()
						}
					}

					if err == nil && resp != nil && resp.Success {
						var boundResp BoundaryDetectionResponse
						if json.Unmarshal([]byte(resp.ActionResultJson), &boundResp) == nil {
							boundaryPageInSlice = boundResp.BoundaryPageInSlice
						}
					} else {
						addTrace(fmt.Sprintf("AI Boundary Detection failed for slice %d-%d: %v", windowStart, windowEnd, err), "WARN")
					}
				}

				os.Remove(sliceTempPath)

				docTypeChunked := parser.GetDocumentProcessorContract().GetDocTypeChunked()
				if docTypeChunked == "" {
					return nil, traceEvents, fmt.Errorf("DocTypeChunked must be configured in DocumentProcessorContract")
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
				docTypeChunked := parser.GetDocumentProcessorContract().GetDocTypeChunked()
				if docTypeChunked == "" {
					return nil, traceEvents, fmt.Errorf("DocTypeChunked must be configured in DocumentProcessorContract")
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
