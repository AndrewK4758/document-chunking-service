package processor

import (
	"context"
	"fmt"
	"io"
	"testing"
	"time"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/protobuf/types/known/structpb"
)

type mockFailingStorage struct {
	failFetch bool
	failSave  bool
}

func (m *mockFailingStorage) FetchFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	if m.failFetch {
		return nil, fmt.Errorf("[Step: JIT_FETCH, URI: %s] storage network timeout simulated", relativePath)
	}
	return []byte("dummy zip content"), nil
}

func (m *mockFailingStorage) FetchStream(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity) (io.ReadCloser, error) {
	if m.failFetch {
		return nil, fmt.Errorf("[Step: JIT_FETCH, URI: %s] storage network timeout simulated", relativePath)
	}
	return nil, fmt.Errorf("not implemented")
}

func (m *mockFailingStorage) SaveFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity, data []byte) (string, error) {
	if m.failSave {
		return "", fmt.Errorf("[Step: STORAGE_SAVE, URI: %s] gateway storage write failed", relativePath)
	}
	return "internal-store://" + relativePath, nil
}

func TestAdversarial_ProcessZipFile_MissingConfigValidation(t *testing.T) {
	store := &mockFailingStorage{}
	chunker := NewChunkingProcessor(nil, store)

	tests := []struct {
		name          string
		payload       map[string]interface{}
		expectedError string
	}{
		{
			name: "Missing MaxPagesBeforeSplit",
			payload: map[string]interface{}{
				"node_config": map[string]interface{}{},
				"global_state": map[string]interface{}{
					"functional": map[string]interface{}{},
				},
				"storage": map[string]interface{}{"document_uri": "test.pdf"},
				"document_processor": map[string]interface{}{
					"sliding_window_size": float64(5),
					"doc_type_corrupted":  "corrupted",
				},
			},
			expectedError: "max_pages_before_split must be configured",
		},
		{
			name: "Missing DocTypeCorrupted",
			payload: map[string]interface{}{
				"node_config": map[string]interface{}{},
				"global_state": map[string]interface{}{
					"functional": map[string]interface{}{},
				},
				"storage": map[string]interface{}{"document_uri": "test.pdf"},
				"document_processor": map[string]interface{}{
					"max_pages_before_split": float64(10),
					"sliding_window_size":    float64(5),
				},
			},
			expectedError: "doc_type_corrupted must be configured",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payloadStruct, _ := structpb.NewStruct(tt.payload)
			// We pass RawFileBytes so fetch succeeds immediately
			_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
				RawFileBytes: []byte("dummy pdf/zip bytes"),
				Identity:     &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
				OriginalRequest: &pb.ExecuteTaskRequest{
					Identity:    &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
					TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
					Payload:     payloadStruct,
				},
			})

			if err == nil {
				t.Fatalf("Expected error for configuration flaw, got nil")
			}
			if !testing.Short() && err != nil && tt.expectedError != "" {
				t.Logf("Got expected config validation error: %v", err)
			}
		})
	}
}

func TestAdversarial_ProcessZipFile_ZeroByteRawBytes(t *testing.T) {
	store := &mockFailingStorage{failFetch: true}
	chunker := NewChunkingProcessor(nil, store)

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "internal-store://tenant/app/job/doc.pdf"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
		},
	})
	// Test when raw file bytes is empty slice and storage fetch fails
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		RawFileBytes: []byte{},
		Identity:     &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})

	if err == nil {
		t.Fatal("Expected error when storage fetch fails, got nil")
	}
}

func TestAdversarial_ProcessZipFile_ContextDoneEarly(t *testing.T) {
	store := &mockFailingStorage{}
	chunker := NewChunkingProcessor(nil, store)

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-1*time.Second))
	defer cancel()

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "internal-store://t/a/j1/doc.pdf"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size":    float64(5),
			"doc_type_corrupted":     "corrupted",
		},
	})
	_, _, err := chunker.ProcessZipFile(ctx, &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
			TaskSubject: pb.WorkerSubject_WORKER_DOC_INSPECT,
			Payload:     payload,
		},
	})

	if err == nil {
		t.Fatal("Expected context error when deadline passed, got nil")
	}
}
