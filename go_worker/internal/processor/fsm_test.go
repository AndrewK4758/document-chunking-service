package processor

import (
	"os"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/stretchr/testify/assert"
)

func TestLogicalBoundaryFSM_MultiDocumentPackage(t *testing.T) {
	fsm := NewLogicalBoundaryFSM()

	// 10-page document stream:
	// Pages 1-3: Closing Disclosure (Boundary on 1)
	// Pages 4-7: Settlement Statement (Boundary on 4)
	// Pages 8-10: Deed of Trust (Boundary on 8)
	evals := []*pb.PageEvaluation{
		{PageNumber: 1, IsFirstPage: true, DocumentType: "Closing Disclosure", BoundaryConfidence: 0.95},
		{PageNumber: 2, IsFirstPage: false, DocumentType: "", BoundaryConfidence: 0.10},
		{PageNumber: 3, IsFirstPage: false, DocumentType: "", BoundaryConfidence: 0.15},
		{PageNumber: 4, IsFirstPage: false, DocumentType: "Settlement Statement", BoundaryConfidence: 0.90},
		{PageNumber: 5, IsFirstPage: false, DocumentType: "", BoundaryConfidence: 0.05},
		{PageNumber: 6, IsFirstPage: false, DocumentType: "", BoundaryConfidence: 0.12},
		{PageNumber: 7, IsFirstPage: false, DocumentType: "", BoundaryConfidence: 0.08},
		{PageNumber: 8, IsFirstPage: false, DocumentType: "Deed-Of-Trust", BoundaryConfidence: 0.88},
		{PageNumber: 9, IsFirstPage: false, DocumentType: "", BoundaryConfidence: 0.10},
		{PageNumber: 10, IsFirstPage: false, DocumentType: "", BoundaryConfidence: 0.05, IsLastPage: true},
	}

	for _, page := range evals {
		decision := fsm.EvaluatePage(page)
		assert.NotNil(t, decision)
	}

	slices := fsm.Finalize()
	assert.Len(t, slices, 3)

	assert.Equal(t, 1, slices[0].StartPage)
	assert.Equal(t, 3, slices[0].EndPage)
	assert.Equal(t, "CLOSING_DISCLOSURE", slices[0].DocumentType)

	assert.Equal(t, 4, slices[1].StartPage)
	assert.Equal(t, 7, slices[1].EndPage)
	assert.Equal(t, "SETTLEMENT_STATEMENT", slices[1].DocumentType)

	assert.Equal(t, 8, slices[2].StartPage)
	assert.Equal(t, 10, slices[2].EndPage)
	assert.Equal(t, "DEED_OF_TRUST", slices[2].DocumentType)
}

func TestLogicalBoundaryFSM_Arbitrate3Tiers(t *testing.T) {
	fsm := NewLogicalBoundaryFSM()

	tests := []struct {
		name             string
		page             *pb.PageEvaluation
		expectedDecision string
		expectedReason   string
	}{
		{
			name: "Tier 1: Explicit First Page",
			page: &pb.PageEvaluation{
				PageNumber:         1,
				IsFirstPage:        true,
				BoundaryConfidence: 0.20,
			},
			expectedDecision: "START_NEW_DOCUMENT",
			expectedReason:   "TIER_1_EXPLICIT_FIRST_PAGE",
		},
		{
			name: "Tier 1: Explicit Document Type Header",
			page: &pb.PageEvaluation{
				PageNumber:         2,
				IsFirstPage:        false,
				DocumentType:       "Closing Disclosure",
				BoundaryConfidence: 0.30,
			},
			expectedDecision: "START_NEW_DOCUMENT",
			expectedReason:   "TIER_1_EXPLICIT_DOCUMENT_HEADER",
		},
		{
			name: "Tier 2: Spatial Header Corroborated with moderate GLiNER confidence",
			page: &pb.PageEvaluation{
				PageNumber:         3,
				IsFirstPage:        false,
				SpatialDensity:     0.85,
				BoundaryConfidence: 0.55,
			},
			expectedDecision: "START_NEW_DOCUMENT",
			expectedReason:   "TIER_2_SPATIAL_HEADER_CORROBORATED",
		},
		{
			name: "Tier 3: High Confidence GLiNER solo inference",
			page: &pb.PageEvaluation{
				PageNumber:         4,
				IsFirstPage:        false,
				SpatialDensity:     0.20,
				BoundaryConfidence: 0.75,
			},
			expectedDecision: "START_NEW_DOCUMENT",
			expectedReason:   "TIER_3_GLINER_HIGH_CONFIDENCE",
		},
		{
			name: "Fallback: Low confidence without cues continues current slice",
			page: &pb.PageEvaluation{
				PageNumber:         5,
				IsFirstPage:        false,
				SpatialDensity:     0.30,
				BoundaryConfidence: 0.40,
			},
			expectedDecision: "CONTINUE_CURRENT",
			expectedReason:   "DEFAULT_CONTINUE_CURRENT",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision := fsm.ArbitrateDecision(tt.page)
			assert.Equal(t, tt.expectedDecision, decision.Decision)
			assert.Equal(t, tt.expectedReason, decision.Reason)
		})
	}
}

func TestLogicalBoundaryFSM_FailFastOnNonContiguousSequence(t *testing.T) {
	fsm := NewLogicalBoundaryFSM()

	fsm.EvaluatePage(&pb.PageEvaluation{PageNumber: 1, IsFirstPage: true})

	assert.Panics(t, func() {
		fsm.EvaluatePage(&pb.PageEvaluation{PageNumber: 3})
	})
}

func TestLogicalBoundaryFSM_FailFastOnNilPage(t *testing.T) {
	fsm := NewLogicalBoundaryFSM()

	assert.Panics(t, func() {
		fsm.EvaluatePage(nil)
	})
	assert.Panics(t, func() {
		fsm.ArbitrateDecision(nil)
	})
}

func TestLogicalBoundaryFSM_DocumentTypeNormalization(t *testing.T) {
	assert.Equal(t, "UNKNOWN", NormalizeDocumentType(""))
	assert.Equal(t, "CLOSING_DISCLOSURE", NormalizeDocumentType("closing disclosure"))
	assert.Equal(t, "SETTLEMENT_STATEMENT", NormalizeDocumentType("settlement-statement"))
	assert.Equal(t, "DEED_OF_TRUST", NormalizeDocumentType("  deed   of   trust  "))
	assert.Equal(t, "MORTGAGE_NOTE", NormalizeDocumentType("MORTGAGE__NOTE"))
}

func TestLogicalBoundaryFSM_DynamicThresholds(t *testing.T) {
	os.Setenv("BOUNDARY_TIER3_SOLO_THRESHOLD", "0.85")
	os.Setenv("BOUNDARY_TIER3_CORROBORATED_THRESHOLD", "0.65")
	defer func() {
		os.Unsetenv("BOUNDARY_TIER3_SOLO_THRESHOLD")
		os.Unsetenv("BOUNDARY_TIER3_CORROBORATED_THRESHOLD")
	}()

	solo, corrob := ResolveThresholds()
	assert.InDelta(t, 0.85, solo, 0.001)
	assert.InDelta(t, 0.65, corrob, 0.001)
}
