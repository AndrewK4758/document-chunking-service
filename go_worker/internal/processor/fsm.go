package processor

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	pb "github.com/AndrewK4758/shared_protos"
)

type SubDocumentSlice struct {
	StartPage    int     `json:"start_page"`
	EndPage      int     `json:"end_page"`
	DocumentType string  `json:"document_type"`
	Confidence   float64 `json:"confidence"`
	Reason       string  `json:"reason"`
}

type LogicalBoundaryFSM struct {
	SoloThreshold         float64
	CorroboratedThreshold float64
	ActiveSlice           *SubDocumentSlice
	CompletedSlices       []SubDocumentSlice
	TotalPagesProcessed   int
}

func ResolveThresholds() (float64, float64) {
	solo := 0.70
	if envSolo := os.Getenv("BOUNDARY_TIER3_SOLO_THRESHOLD"); envSolo != "" {
		if val, err := strconv.ParseFloat(envSolo, 64); err == nil && val >= 0.0 && val <= 1.0 {
			solo = val
		}
	}

	corroborated := 0.50
	if envCorrob := os.Getenv("BOUNDARY_TIER3_CORROBORATED_THRESHOLD"); envCorrob != "" {
		if val, err := strconv.ParseFloat(envCorrob, 64); err == nil && val >= 0.0 && val <= 1.0 {
			corroborated = val
		}
	}

	return solo, corroborated
}

func NewLogicalBoundaryFSM() *LogicalBoundaryFSM {
	solo, corrob := ResolveThresholds()
	return &LogicalBoundaryFSM{
		SoloThreshold:         solo,
		CorroboratedThreshold: corrob,
		CompletedSlices:       make([]SubDocumentSlice, 0),
	}
}

func NormalizeDocumentType(docType string) string {
	trimmed := strings.TrimSpace(docType)
	if trimmed == "" {
		return "UNKNOWN"
	}
	upper := strings.ToUpper(trimmed)
	upper = strings.ReplaceAll(upper, "-", "_")
	upper = strings.ReplaceAll(upper, " ", "_")
	for strings.Contains(upper, "__") {
		upper = strings.ReplaceAll(upper, "__", "_")
	}
	return upper
}

func (fsm *LogicalBoundaryFSM) ArbitrateDecision(page *pb.PageEvaluation) *pb.LogicalBoundaryDecision {
	if page == nil {
		panic(fmt.Errorf("action failed: page evaluation cannot be nil"))
	}

	// Tier 1: Deterministic Pattern / Header cues
	if page.IsFirstPage {
		return &pb.LogicalBoundaryDecision{
			PageNumber: page.PageNumber,
			Decision:   "START_NEW_DOCUMENT",
			Confidence: 1.0,
			Reason:     "TIER_1_EXPLICIT_FIRST_PAGE",
		}
	}

	if normType := NormalizeDocumentType(page.DocumentType); normType != "UNKNOWN" && normType != "GENERIC_DOCUMENT" {
		return &pb.LogicalBoundaryDecision{
			PageNumber: page.PageNumber,
			Decision:   "START_NEW_DOCUMENT",
			Confidence: 0.95,
			Reason:     "TIER_1_EXPLICIT_DOCUMENT_HEADER",
		}
	}

	// Tier 2: Spatial Layout Prominence Corroboration
	hasSpatialHeader := page.SpatialDensity > 0.75
	if hasSpatialHeader && float64(page.BoundaryConfidence) >= fsm.CorroboratedThreshold {
		return &pb.LogicalBoundaryDecision{
			PageNumber: page.PageNumber,
			Decision:   "START_NEW_DOCUMENT",
			Confidence: page.BoundaryConfidence,
			Reason:     "TIER_2_SPATIAL_HEADER_CORROBORATED",
		}
	}

	// Tier 3: GLiNER Zero-Shot Entity Classification
	if float64(page.BoundaryConfidence) >= fsm.SoloThreshold {
		return &pb.LogicalBoundaryDecision{
			PageNumber: page.PageNumber,
			Decision:   "START_NEW_DOCUMENT",
			Confidence: page.BoundaryConfidence,
			Reason:     "TIER_3_GLINER_HIGH_CONFIDENCE",
		}
	}

	// Fallback: Continue current slice
	return &pb.LogicalBoundaryDecision{
		PageNumber: page.PageNumber,
		Decision:   "CONTINUE_CURRENT",
		Confidence: 0.50,
		Reason:     "DEFAULT_CONTINUE_CURRENT",
	}
}

func (fsm *LogicalBoundaryFSM) EvaluatePage(page *pb.PageEvaluation) *pb.LogicalBoundaryDecision {
	if page == nil {
		panic(fmt.Errorf("action failed: page evaluation cannot be nil"))
	}
	expectedPage := fsm.TotalPagesProcessed + 1
	if int(page.PageNumber) != expectedPage {
		panic(fmt.Errorf("action failed: invalid page sequence: expected %d, got %d", expectedPage, page.PageNumber))
	}

	decision := fsm.ArbitrateDecision(page)
	normDocType := NormalizeDocumentType(page.DocumentType)

	if decision.Decision == "START_NEW_DOCUMENT" {
		if fsm.ActiveSlice != nil {
			fsm.ActiveSlice.EndPage = int(page.PageNumber) - 1
			fsm.CompletedSlices = append(fsm.CompletedSlices, *fsm.ActiveSlice)
		}

		fsm.ActiveSlice = &SubDocumentSlice{
			StartPage:    int(page.PageNumber),
			EndPage:      int(page.PageNumber),
			DocumentType: normDocType,
			Confidence:   float64(decision.Confidence),
			Reason:       decision.Reason,
		}
	} else {
		if fsm.ActiveSlice == nil {
			fsm.ActiveSlice = &SubDocumentSlice{
				StartPage:    int(page.PageNumber),
				EndPage:      int(page.PageNumber),
				DocumentType: "UNKNOWN",
				Confidence:   float64(decision.Confidence),
				Reason:       decision.Reason,
			}
		} else {
			fsm.ActiveSlice.EndPage = int(page.PageNumber)
		}
	}

	fsm.TotalPagesProcessed++
	return decision
}

func (fsm *LogicalBoundaryFSM) Finalize() []SubDocumentSlice {
	if fsm.ActiveSlice != nil {
		fsm.CompletedSlices = append(fsm.CompletedSlices, *fsm.ActiveSlice)
		fsm.ActiveSlice = nil
	}

	totalPagesInSlices := 0
	expectedNextStart := 1
	for _, slice := range fsm.CompletedSlices {
		if slice.StartPage != expectedNextStart {
			panic(fmt.Errorf("action failed: page gap detected: expected start %d, got %d", expectedNextStart, slice.StartPage))
		}
		if slice.EndPage < slice.StartPage {
			panic(fmt.Errorf("action failed: inverted slice bounds: start %d, end %d", slice.StartPage, slice.EndPage))
		}
		totalPagesInSlices += slice.EndPage - slice.StartPage + 1
		expectedNextStart = slice.EndPage + 1
	}

	if totalPagesInSlices != fsm.TotalPagesProcessed {
		panic(fmt.Errorf("action failed: page count invariant broken: processed %d pages, sliced %d pages", fsm.TotalPagesProcessed, totalPagesInSlices))
	}

	return fsm.CompletedSlices
}
