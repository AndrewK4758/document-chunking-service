package processor

import (
	"context"
	"encoding/json"
	"fmt"
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

func cleanupDir(path string, addTrace func(string, string)) {
	err := os.RemoveAll(path)
	if err != nil {
		addTrace(fmt.Sprintf("Failed to clean up extraction directory %s: %v", path, err), "WARN")
	}
}

func (p *ChunkingProcessor) executeAiAction(ctx context.Context, origReq *pb.ExecuteWorkflowNodeRequest, actionDef *pb.ActionDefinition, inputBytes []byte, modelChoice pb.ModelChoice, identity *pb.InfrastructureIdentity) (*pb.PerformActionResponse, error) {
	if p.AiClient == nil {
		return nil, nil
	}
	if origReq == nil {
		return nil, nil
	}

	stream, err := p.AiClient.PerformAction(ctx)
	if err != nil {
		return nil, err
	}

	err = stream.Send(&pb.PerformActionRequest{
		Context:       origReq,
		CurrentAction: actionDef,
		InputBytes:    inputBytes,
		ModelChoice:   modelChoice,
		Identity:      identity,
	})
	if err != nil {
		return nil, err
	}

	err = stream.CloseSend()
	if err != nil {
		return nil, err
	}

	return stream.Recv()
}

func parseBoundaryResponse(resp *pb.PerformActionResponse) (int, error) {
	if resp == nil {
		return 0, nil
	}
	if !resp.Success {
		return 0, nil
	}
	if resp.ActionResult == nil {
		return 0, nil
	}

	jsonBytes, err := resp.ActionResult.MarshalJSON()
	if err != nil {
		return 0, err
	}

	var boundResp BoundaryDetectionResponse
	err = json.Unmarshal(jsonBytes, &boundResp)
	if err != nil {
		return 0, err
	}

	return boundResp.BoundaryPageInSlice, nil
}

func (p *ChunkingProcessor) processSingleDoc(ctx context.Context, req *pb.ChunkDocumentRequest, pdfPath string, pageCount int, docUri string, parser *workflowparser.NodeRequestParser, modelChoice pb.ModelChoice, addTrace func(string, string)) (*pb.ProcessedDocument, error) {
	inputBytes, err := os.ReadFile(pdfPath)
	if err != nil {
		addTrace(fmt.Sprintf("Failed to read PDF file %s: %v", filepath.Base(pdfPath), err), "ERROR")
		return nil, fmt.Errorf("[Step: SINGLE_DOC_PROCESSING, URI: %s] failed to read pdf file: %w", docUri, err)
	}

	actionName := parser.GetDocumentProcessorContract().GetActionNameClassify()
	if actionName == "" {
		inputBytes = nil
		return nil, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] ActionNameClassify must be configured in DocumentProcessorContract", docUri)
	}

	resp, aiErr := p.executeAiAction(ctx, req.OriginalRequest, &pb.ActionDefinition{
		Name:   actionName,
		Prompt: parser.GetSystemInstructions(),
	}, inputBytes, modelChoice, req.GetIdentity())

	if aiErr != nil {
		addTrace(fmt.Sprintf("AI Classification failed: %v", aiErr), "ERROR")
	}

	if resp != nil && aiErr == nil {
		jsonBytes, _ := resp.ActionResult.MarshalJSON()
		addTrace(fmt.Sprintf("AI Classification successful: %s", string(jsonBytes)), "INFO")
	}

	var relativePath string
	var uploadErr error
	if p.Store != nil {
		relativePath, uploadErr = p.Store.SaveFile(ctx, fmt.Sprintf("%s/%s", req.GetIdentity().GetJobId(), filepath.Base(pdfPath)), req.GetIdentity(), inputBytes)
	}
	if uploadErr != nil {
		addTrace(fmt.Sprintf("Failed to upload classified doc: %v", uploadErr), "ERROR")
		relativePath = ""
	}

	inputBytes = nil

	docTypeClassified := parser.GetDocumentProcessorContract().GetDocTypeClassified()
	if docTypeClassified == "" {
		return nil, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] DocTypeClassified must be configured in DocumentProcessorContract", docUri)
	}

	return &pb.ProcessedDocument{
		DocumentType:     docTypeClassified,
		RelativeFilePath: relativePath,
		PageCount:        int32(pageCount),
	}, nil
}

func (p *ChunkingProcessor) processSingleDocAsSlice(ctx context.Context, req *pb.ChunkDocumentRequest, pdfPath string, pageCount int, docUri string, parser *workflowparser.NodeRequestParser, modelChoice pb.ModelChoice, addTrace func(string, string)) ([]*pb.ProcessedDocument, error) {
	doc, err := p.processSingleDoc(ctx, req, pdfPath, pageCount, docUri, parser, modelChoice, addTrace)
	if err != nil {
		return nil, err
	}
	return []*pb.ProcessedDocument{doc}, nil
}

func (p *ChunkingProcessor) detectBoundaryInSlice(ctx context.Context, req *pb.ChunkDocumentRequest, sliceTempPath string, docUri string, parser *workflowparser.NodeRequestParser, addTrace func(string, string), windowStart int, windowEnd int) (int, error) {
	if p.AiClient == nil {
		return 0, nil
	}
	if req.OriginalRequest == nil {
		return 0, nil
	}

	sliceBytes, err := os.ReadFile(sliceTempPath)
	if err != nil {
		addTrace(fmt.Sprintf("Failed to read slice temp file %s: %v", sliceTempPath, err), "ERROR")
		return 0, fmt.Errorf("[Step: SLICING, URI: %s] failed to read slice file: %w", docUri, err)
	}

	prompt := parser.GetDocumentProcessorContract().GetBoundaryDetectionPrompt()
	if prompt == "" {
		addTrace("Boundary detection prompt missing or empty in request", "ERROR")
		sliceBytes = nil
		return 0, fmt.Errorf("[Step: SLICING, URI: %s] boundary detection prompt is required but not provided in request", docUri)
	}

	actionName := parser.GetDocumentProcessorContract().GetActionNameBoundary()
	if actionName == "" {
		sliceBytes = nil
		return 0, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] ActionNameBoundary must be configured in DocumentProcessorContract", docUri)
	}

	actionDef := &pb.ActionDefinition{
		Name:   actionName,
		Prompt: prompt,
	}

	resp, err := p.executeAiAction(ctx, req.OriginalRequest, actionDef, sliceBytes, pb.ModelChoice_MODEL_CHOICE_OCR, req.GetIdentity())
	sliceBytes = nil

	boundaryPageInSlice, unmarshalErr := parseBoundaryResponse(resp)
	if unmarshalErr != nil {
		addTrace(fmt.Sprintf("Failed to unmarshal boundary detection response: %v", unmarshalErr), "ERROR")
		return 0, fmt.Errorf("[Step: SLICING, URI: %s] failed to unmarshal boundary detection response: %w", docUri, unmarshalErr)
	}

	if resp == nil {
		addTrace(fmt.Sprintf("AI Boundary Detection failed for slice %d-%d: %v", windowStart, windowEnd, err), "WARN")
	}
	if resp != nil && !resp.Success {
		addTrace(fmt.Sprintf("AI Boundary Detection failed for slice %d-%d: %v", windowStart, windowEnd, err), "WARN")
	}

	return boundaryPageInSlice, nil
}

func (p *ChunkingProcessor) extractAndUploadChunk(ctx context.Context, req *pb.ChunkDocumentRequest, pdfPath string, startPage int, endPage int, chunkIndex int, extractDir string, docUri string, docTypeChunked string, addTrace func(string, string)) (*pb.ProcessedDocument, error) {
	chunkPath := filepath.Join(extractDir, fmt.Sprintf("chunk_%d_%d.pdf", startPage, endPage))
	err := api.TrimFile(pdfPath, chunkPath, []string{fmt.Sprintf("%d-%d", startPage, endPage)}, model.NewDefaultConfiguration())
	if err != nil {
		addTrace(fmt.Sprintf("Failed to extract chunk pages %d-%d: %v", startPage, endPage, err), "ERROR")
		return nil, fmt.Errorf("[Step: SLICING, URI: %s] failed to extract chunk: %w", docUri, err)
	}

	addTrace(fmt.Sprintf("Created chunk %d (pages %d-%d).", chunkIndex, startPage, endPage), "INFO")

	var chunkBytes []byte
	var readErr error
	if p.Store != nil {
		chunkBytes, readErr = os.ReadFile(chunkPath)
	}
	if readErr != nil {
		addTrace(fmt.Sprintf("Failed to read chunk file %s: %v", chunkPath, readErr), "ERROR")
		return nil, fmt.Errorf("[Step: SLICING, URI: %s] failed to read chunk file: %w", docUri, readErr)
	}

	var relativePath string
	var uploadErr error
	if p.Store != nil {
		relativePath, uploadErr = p.Store.SaveFile(ctx, fmt.Sprintf("%s/chunk_%d.pdf", req.GetIdentity().GetJobId(), chunkIndex), req.GetIdentity(), chunkBytes)
		chunkBytes = nil
	}
	if uploadErr != nil {
		addTrace(fmt.Sprintf("Failed to upload chunk %d: %v", chunkIndex, uploadErr), "ERROR")
		relativePath = ""
	}

	return &pb.ProcessedDocument{
		DocumentType:     docTypeChunked,
		RelativeFilePath: relativePath,
		PageCount:        int32(endPage - startPage + 1),
	}, nil
}

func (p *ChunkingProcessor) processLargeDoc(ctx context.Context, req *pb.ChunkDocumentRequest, pdfPath string, pageCount int, maxPagesBeforeSplit int, extractDir string, docUri string, parser *workflowparser.NodeRequestParser, addTrace func(string, string)) ([]*pb.ProcessedDocument, error) {
	addTrace(fmt.Sprintf("%s exceeds max pages (%d), engaging sliding window splitter...", filepath.Base(pdfPath), maxPagesBeforeSplit), "INFO")

	slidingWindowSize := int(parser.GetDocumentProcessorContract().GetSlidingWindowSize())
	if slidingWindowSize <= 0 {
		return nil, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] SlidingWindowSize must be configured in DocumentProcessorContract and greater than 0", docUri)
	}

	currentDocStart := 1
	windowStart := 1
	var chunkIndex int
	var processedDocs []*pb.ProcessedDocument

	for windowStart <= pageCount {
		err := ctx.Err()
		if err != nil {
			return processedDocs, err
		}

		windowEnd := min(windowStart+slidingWindowSize-1, pageCount)
		addTrace(fmt.Sprintf("Evaluating slice pages %d-%d for boundaries", windowStart, windowEnd), "INFO")

		sliceTempPath := filepath.Join(extractDir, fmt.Sprintf("slice_%d_%d.pdf", windowStart, windowEnd))
		err = api.TrimFile(pdfPath, sliceTempPath, []string{fmt.Sprintf("%d-%d", windowStart, windowEnd)}, model.NewDefaultConfiguration())
		if err != nil {
			addTrace(fmt.Sprintf("Failed to extract slice pages %d-%d: %v", windowStart, windowEnd, err), "ERROR")
			windowStart = windowEnd + 1
			continue
		}

		boundaryPageInSlice, sliceErr := p.detectBoundaryInSlice(ctx, req, sliceTempPath, docUri, parser, addTrace, windowStart, windowEnd)
		os.Remove(sliceTempPath)

		if sliceErr != nil {
			return nil, sliceErr
		}

		docTypeChunked := parser.GetDocumentProcessorContract().GetDocTypeChunked()
		if docTypeChunked == "" {
			return nil, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] DocTypeChunked must be configured in DocumentProcessorContract", docUri)
		}

		if boundaryPageInSlice <= 0 {
			addTrace(fmt.Sprintf("No boundary found in pages %d-%d. Continuing document.", windowStart, windowEnd), "INFO")
			windowStart = windowEnd + 1
			continue
		}

		absoluteBoundary := windowStart + boundaryPageInSlice - 1
		if absoluteBoundary <= currentDocStart {
			windowStart = windowEnd + 1
			continue
		}
		if absoluteBoundary > pageCount {
			windowStart = windowEnd + 1
			continue
		}

		// Valid boundary split!
		chunkDoc, chunkErr := p.extractAndUploadChunk(ctx, req, pdfPath, currentDocStart, absoluteBoundary-1, chunkIndex, extractDir, docUri, docTypeChunked, addTrace)
		if chunkErr != nil {
			return nil, chunkErr
		}

		processedDocs = append(processedDocs, chunkDoc)
		chunkIndex++
		currentDocStart = absoluteBoundary
		windowStart = absoluteBoundary
	}

	if currentDocStart > pageCount {
		return processedDocs, nil
	}

	docTypeChunked := parser.GetDocumentProcessorContract().GetDocTypeChunked()
	if docTypeChunked == "" {
		return nil, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] DocTypeChunked must be configured in DocumentProcessorContract", docUri)
	}

	chunkDoc, chunkErr := p.extractAndUploadChunk(ctx, req, pdfPath, currentDocStart, pageCount, chunkIndex, extractDir, docUri, docTypeChunked, addTrace)
	if chunkErr != nil {
		return nil, chunkErr
	}

	processedDocs = append(processedDocs, chunkDoc)
	return processedDocs, nil
}

func (p *ChunkingProcessor) processDoc(ctx context.Context, req *pb.ChunkDocumentRequest, pdfPath string, pageCount int, maxPagesBeforeSplit int, extractDir string, docUri string, parser *workflowparser.NodeRequestParser, modelChoice pb.ModelChoice, addTrace func(string, string)) ([]*pb.ProcessedDocument, error) {
	isSingleDoc := pageCount <= maxPagesBeforeSplit
	if isSingleDoc {
		return p.processSingleDocAsSlice(ctx, req, pdfPath, pageCount, docUri, parser, modelChoice, addTrace)
	}
	return p.processLargeDoc(ctx, req, pdfPath, pageCount, maxPagesBeforeSplit, extractDir, docUri, parser, addTrace)
}

func (p *ChunkingProcessor) ProcessZipFile(ctx context.Context, req *pb.ChunkDocumentRequest) ([]*pb.ProcessedDocument, []*pb.ChunkingTraceEvent, error) {
	err := ctx.Err()
	if err != nil {
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

	parser := workflowparser.NewNodeRequestParser(req.GetOriginalRequest())

	docUri := parser.GetDocumentUri()
	if docUri == "" {
		docUri = "unknown"
	}

	if len(req.RawFileBytes) > 0 {
		addTrace("Using raw file bytes from request", "INFO")
		zipBytes = req.RawFileBytes
	}

	if len(zipBytes) == 0 && docUri == "unknown" {
		return nil, traceEvents, fmt.Errorf("[Step: JIT_FETCH, URI: %s] no file data provided in request", docUri)
	}

	if len(zipBytes) == 0 && p.Store == nil {
		return nil, traceEvents, fmt.Errorf("[Step: JIT_FETCH, URI: %s] storage manager is not configured", docUri)
	}

	if len(zipBytes) == 0 {
		addTrace(fmt.Sprintf("Fetching %s from store", docUri), "INFO")
		zipBytes, err = p.Store.FetchFile(ctx, docUri, req.Identity)
	}

	if err != nil {
		addTrace(fmt.Sprintf("Failed to fetch file: %v", err), "ERROR")
		return nil, traceEvents, fmt.Errorf("[Step: JIT_FETCH, URI: %s] failed to fetch from store: %w", docUri, err)
	}

	extractDir := filepath.Join(os.TempDir(), fmt.Sprintf("chunking_job_%s_%d", req.GetIdentity().GetJobId(), time.Now().UnixNano()))
	err = os.MkdirAll(extractDir, 0755)
	if err != nil {
		return nil, traceEvents, fmt.Errorf("[Step: TEMP_WRITE, URI: %s] failed to create extraction directory: %w", docUri, err)
	}
	defer cleanupDir(extractDir, addTrace)

	originalFilename := "input.pdf"
	if docUri != "unknown" {
		originalFilename = filepath.Base(docUri)
	}

	tempInputPath := filepath.Join(extractDir, originalFilename)
	err = os.WriteFile(tempInputPath, zipBytes, 0644)
	if err != nil {
		return nil, traceEvents, fmt.Errorf("[Step: TEMP_WRITE, URI: %s] failed to write temp input file: %w", docUri, err)
	}
	// Immediately discard zipBytes from memory to enable Garbage Collection
	zipBytes = nil

	extractedPaths, err := archive.Extract(tempInputPath, extractDir)
	if err != nil {
		addTrace(fmt.Sprintf("Failed to extract archive: %v", err), "ERROR")
		return nil, traceEvents, fmt.Errorf("[Step: ARCHIVE_EXTRACTION, URI: %s] failed to extract archive: %w", docUri, err)
	}

	var extractedFiles []string
	extractedFiles = append(extractedFiles, extractedPaths...)

	addTrace(fmt.Sprintf("Extracted %d documents", len(extractedFiles)), "INFO")

	maxPagesBeforeSplit := int(parser.GetDocumentProcessorContract().GetMaxPagesBeforeSplit())
	if maxPagesBeforeSplit <= 0 {
		return nil, traceEvents, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] MaxPagesBeforeSplit must be configured in DocumentProcessorContract and greater than 0", docUri)
	}

	var processedDocs []*pb.ProcessedDocument

	for _, pdfPath := range extractedFiles {
		docTypeCorrupted := parser.GetDocumentProcessorContract().GetDocTypeCorrupted()
		if docTypeCorrupted == "" {
			return nil, traceEvents, fmt.Errorf("[Step: CONFIG_VALIDATION, URI: %s] DocTypeCorrupted must be configured in DocumentProcessorContract", docUri)
		}

		err = api.ValidateFile(pdfPath, model.NewDefaultConfiguration())
		if err != nil {
			addTrace(fmt.Sprintf("PDF Validation failed for %s (possible encryption or corruption): %v", filepath.Base(pdfPath), err), "WARN")
			processedDocs = append(processedDocs, &pb.ProcessedDocument{
				DocumentType:     docTypeCorrupted,
				RelativeFilePath: pdfPath,
				PageCount:        0,
			})
			continue
		}

		pageCount, pageErr := api.PageCountFile(pdfPath)
		if pageErr != nil {
			addTrace(fmt.Sprintf("Failed to get page count for %s: %v", filepath.Base(pdfPath), pageErr), "ERROR")
			continue
		}

		addTrace(fmt.Sprintf("Analyzed %s (Pages: %d)", filepath.Base(pdfPath), pageCount), "INFO")

		err = ctx.Err()
		if err != nil {
			return processedDocs, traceEvents, err
		}

		// Probe text vs OCR
		err = api.ExtractContentFile(pdfPath, extractDir, []string{"1"}, model.NewDefaultConfiguration())
		isStructuredText := err == nil

		modelChoice := pb.ModelChoice_MODEL_CHOICE_OCR
		if isStructuredText {
			modelChoice = pb.ModelChoice_MODEL_CHOICE_CLASSIFICATION
			addTrace(fmt.Sprintf("%s has embedded text. Routing to Standard LLM.", filepath.Base(pdfPath)), "INFO")
		}
		if !isStructuredText {
			addTrace(fmt.Sprintf("%s appears to be a scanned document. Routing to OCR Vision LLM.", filepath.Base(pdfPath)), "INFO")
		}

		docs, docErr := p.processDoc(ctx, req, pdfPath, pageCount, maxPagesBeforeSplit, extractDir, docUri, parser, modelChoice, addTrace)
		if docErr != nil {
			return nil, traceEvents, docErr
		}
		processedDocs = append(processedDocs, docs...)
	}

	return processedDocs, traceEvents, nil
}
