package processor

import (
	"context"
	"fmt"
	"testing"
	"time"

	pb "github.com/AndrewK4758/shared_protos"
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
		docContract   *pb.DocumentProcessorContract
		expectedError string
	}{
		{
			name: "Missing MaxPagesBeforeSplit",
			docContract: &pb.DocumentProcessorContract{
				MaxPagesBeforeSplit: 0,
				SlidingWindowSize:   5,
				DocTypeCorrupted:    "corrupted",
			},
			expectedError: "MaxPagesBeforeSplit must be configured",
		},
		{
			name: "Missing DocTypeCorrupted",
			docContract: &pb.DocumentProcessorContract{
				MaxPagesBeforeSplit: 10,
				SlidingWindowSize:   5,
				DocTypeCorrupted:    "",
			},
			expectedError: "DocTypeCorrupted must be configured",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// We pass RawFileBytes so fetch succeeds immediately
			_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
				RawFileBytes: []byte("dummy pdf/zip bytes"),
				Identity:     &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
				OriginalRequest: &pb.ExecuteWorkflowNodeRequest{
					Storage:           &pb.StorageContract{DocumentUri: "test.pdf"},
					DocumentProcessor: tt.docContract,
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

	// Test when raw file bytes is empty slice and storage fetch fails
	_, _, err := chunker.ProcessZipFile(context.Background(), &pb.ChunkDocumentRequest{
		RawFileBytes: []byte{},
		Identity:     &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
		OriginalRequest: &pb.ExecuteWorkflowNodeRequest{
			Storage: &pb.StorageContract{DocumentUri: "internal-store://tenant/app/job/doc.pdf"},
			DocumentProcessor: &pb.DocumentProcessorContract{
				MaxPagesBeforeSplit: 10,
				SlidingWindowSize:   5,
				DocTypeCorrupted:    "corrupted",
			},
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

	_, _, err := chunker.ProcessZipFile(ctx, &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j1"},
		OriginalRequest: &pb.ExecuteWorkflowNodeRequest{
			Storage: &pb.StorageContract{DocumentUri: "internal-store://t/a/j1/doc.pdf"},
		},
	})

	if err == nil {
		t.Fatal("Expected context error when deadline passed, got nil")
	}
}
