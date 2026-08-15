package server

import (
	"context"
	"fmt"
	"io"
	"sync"
	"testing"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/structpb"
)

type mockStreamWithCounter struct {
	grpc.ServerStream
	mu         sync.Mutex
	requests   []*pb.ChunkDocumentRequest
	recvIdx    int
	responses  []*pb.ChunkDocumentResponse
	sendErr    error
	recvErr    error
	contextVal context.Context
}

func (m *mockStreamWithCounter) Context() context.Context {
	if m.contextVal != nil {
		return m.contextVal
	}
	return context.Background()
}

func (m *mockStreamWithCounter) Recv() (*pb.ChunkDocumentRequest, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.recvErr != nil {
		return nil, m.recvErr
	}
	if m.recvIdx >= len(m.requests) {
		return nil, io.EOF
	}
	req := m.requests[m.recvIdx]
	m.recvIdx++
	return req, nil
}

func (m *mockStreamWithCounter) Send(resp *pb.ChunkDocumentResponse) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.sendErr != nil {
		return m.sendErr
	}
	m.responses = append(m.responses, resp)
	return nil
}

func TestAdversarial_GRPC_StreamReceiveError(t *testing.T) {
	srv := NewDocumentProcessingServer(nil, nil)
	stream := &mockStreamWithCounter{
		recvErr: fmt.Errorf("gRPC transport connection broken"),
	}

	err := srv.ChunkDocument(stream)
	if err == nil {
		t.Fatal("Expected error on stream Recv failure, got nil")
	}
}

func TestAdversarial_GRPC_StreamSendError(t *testing.T) {
	tempDir := t.TempDir()
	store := &mockStorage{basePath: tempDir}
	srv := NewDocumentProcessingServer(nil, store)

	payload, _ := structpb.NewStruct(map[string]interface{}{
		"node_config": map[string]interface{}{},
		"global_state": map[string]interface{}{
			"functional": map[string]interface{}{},
		},
		"storage": map[string]interface{}{"document_uri": "nonexistent.zip"},
		"document_processor": map[string]interface{}{
			"max_pages_before_split": float64(10),
			"sliding_window_size": float64(5),
			"doc_type_corrupted": "corrupted",
		},
	})

	req := &pb.ChunkDocumentRequest{
		Identity: &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_send_err"},
		OriginalRequest: &pb.ExecuteTaskRequest{
			Identity:    &pb.InfrastructureIdentity{TenantId: "t1", AppId: "a1", JobId: "job_send_err"},
			TaskSubject: "worker.doc.process",
			Payload:     payload,
		},
	}

	stream := &mockStreamWithCounter{
		requests: []*pb.ChunkDocumentRequest{req},
		sendErr:  fmt.Errorf("client reset stream"),
	}

	err := srv.ChunkDocument(stream)
	if err == nil {
		t.Fatal("Expected error when stream Send fails, got nil")
	}
}

func TestAdversarial_NATS_SendErrorReply_NilReplySubj(t *testing.T) {
	server := &NatsServer{}
	// Calling sendErrorReply with empty replySubj should return early safely without panic
	server.sendErrorReply(context.Background(), "", "CODE", "msg", &pb.InfrastructureIdentity{TenantId: "t", AppId: "a", JobId: "j"})
}
