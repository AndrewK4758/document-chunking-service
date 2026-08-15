package server

import (
	"context"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/grpc"
)

type MockCppMlWorkerServiceClient struct {
	RedactDocumentFunc func(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error)
}

func (m *MockCppMlWorkerServiceClient) RedactDocument(ctx context.Context, in *pb.RedactDocumentRequest, opts ...grpc.CallOption) (*pb.RedactDocumentResponse, error) {
	if m.RedactDocumentFunc != nil {
		return m.RedactDocumentFunc(ctx, in, opts...)
	}
	return &pb.RedactDocumentResponse{Success: true}, nil
}
