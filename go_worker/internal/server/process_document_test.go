package server

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/grpc"
)

func TestProcessDocument_Success(t *testing.T) {
	tempDir := t.TempDir()
	store := &mockStorage{basePath: tempDir}
	os.MkdirAll(filepath.Join(tempDir, "path/to"), 0755)
	os.WriteFile(filepath.Join(tempDir, "path/to/doc.pdf"), []byte("dummy pdf content"), 0644)

	cppCalled := false
	netCalled := false

	cppClient := &MockCppMlWorkerServiceClient{
		RedactDocumentFunc: func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
			cppCalled = true
			return &pb.RedactDocumentResponse{Success: true}, nil
		},
	}

	netClient := &MockMlWorkerServiceClient{
		RedactPiiFunc: func(ctx context.Context, in *pb.PiiRedactionRequest, opts ...grpc.CallOption) (*pb.PiiRedactionResponse, error) {
			netCalled = true
			return &pb.PiiRedactionResponse{Success: true}, nil
		},
	}

	srv := NewDocumentProcessingServer(nil, cppClient, netClient, store)

	logDir := t.TempDir()
	os.Setenv("PROJECT_ROOT", logDir)
	defer os.Unsetenv("PROJECT_ROOT")

	req := &pb.ProcessDocRequest{
		DocumentUri: "path/to/doc.pdf",
		Identity: &pb.InfrastructureIdentity{
			TenantId: "t1",
			AppId:    "a1",
			JobId:    "job_456",
		},
	}

	resp, err := srv.ProcessDocument(context.Background(), req)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if !resp.Success {
		t.Errorf("Expected success, got false. ErrorMessage: %s", resp.ErrorMessage)
	}
	if !cppCalled {
		t.Error("Expected cpp client to be called")
	}
	if !netCalled {
		t.Error("Expected net client to be called")
	}

	time.Sleep(50 * time.Millisecond)

	logFile := filepath.Join(logDir, "logs", "benchmark.log")
	content, err := os.ReadFile(logFile)
	if err != nil {
		t.Fatalf("Failed to read log file: %v", err)
	}
	if !strings.Contains(string(content), "JobId: job_456") {
		t.Errorf("Log file missing JobId: %s", string(content))
	}
	if !strings.Contains(string(content), "CppSuccess: true") {
		t.Errorf("Log file missing CppSuccess: %s", string(content))
	}
}

func TestProcessDocument_MissingIdentity(t *testing.T) {
	srv := NewDocumentProcessingServer(nil, nil, nil, nil)
	
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic due to missing identity")
		}
	}()

	req := &pb.ProcessDocRequest{
		DocumentUri: "path/to/doc.pdf",
	}

	srv.ProcessDocument(context.Background(), req)
}
