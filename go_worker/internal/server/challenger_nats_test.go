package server

import (
	"encoding/json"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/protobuf/encoding/protojson"
)

func TestChallenger_NatsStructUnmarshaling_PreservesIsScanned(t *testing.T) {
	docs := []*pb.ProcessedDocument{
		{
			DocumentType:     "classified",
			RelativeFilePath: "scanned_doc.pdf",
			PageCount:        5,
			IsScanned:        true,
		},
		{
			DocumentType:     "classified",
			RelativeFilePath: "text_doc.pdf",
			PageCount:        3,
			IsScanned:        false,
		},
	}

	structResult, err := buildDocsStructResult(docs)
	if err != nil {
		t.Fatalf("buildDocsStructResult error: %v", err)
	}
	if structResult == nil {
		t.Fatalf("buildDocsStructResult returned nil")
	}

	// 1. Verify protojson marshaling of structResult
	jsonBytes, err := protojson.Marshal(structResult)
	if err != nil {
		t.Fatalf("protojson.Marshal(structResult) error: %v", err)
	}

	// 2. Unmarshal into generic map to check JSON payload structure
	var rawMap map[string]any
	err = json.Unmarshal(jsonBytes, &rawMap)
	if err != nil {
		t.Fatalf("json.Unmarshal rawMap error: %v", err)
	}

	docList, ok := rawMap["documents"].([]any)
	if !ok || len(docList) != 2 {
		t.Fatalf("Expected 'documents' array with 2 items in rawMap, got: %v", rawMap)
	}

	doc0, ok := docList[0].(map[string]any)
	if !ok {
		t.Fatalf("Expected doc 0 to be a map")
	}

	if isScanned, exists := doc0["is_scanned"]; !exists || isScanned != true {
		t.Errorf("Doc 0 (scanned): expected is_scanned=true, got exists=%v, val=%v", exists, isScanned)
	}

	doc1, ok := docList[1].(map[string]any)
	if !ok {
		t.Fatalf("Expected doc 1 to be a map")
	}
	_ = doc1 // doc1 exists in map (for is_scanned=false, field may be omitted or false)

	// Note: for false, json.Marshal on pb struct with `omitempty` omits false fields,
	// or in structpb.Struct protojson, false bools are omitted or false.
	// Check if unmarshaled into pb.ProcessedDocument struct:
	var reconstructedDocs struct {
		Documents []*pb.ProcessedDocument `json:"documents"`
	}

	reconstructedBytes, err := json.Marshal(structResult.AsMap())
	if err != nil {
		t.Fatalf("json.Marshal(structResult.AsMap()) error: %v", err)
	}

	err = json.Unmarshal(reconstructedBytes, &reconstructedDocs)
	if err != nil {
		t.Fatalf("json.Unmarshal reconstructedDocs error: %v", err)
	}

	if len(reconstructedDocs.Documents) != 2 {
		t.Fatalf("Expected 2 reconstructed docs, got %d", len(reconstructedDocs.Documents))
	}

	if reconstructedDocs.Documents[0].IsScanned != true {
		t.Errorf("Reconstructed doc 0 IsScanned expected true, got false")
	}

	if reconstructedDocs.Documents[1].IsScanned != false {
		t.Errorf("Reconstructed doc 1 IsScanned expected false, got true")
	}
}
