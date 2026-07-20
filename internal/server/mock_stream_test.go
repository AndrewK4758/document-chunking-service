package server

import (
	"context"
	"io"
	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/grpc"
)

type mockChunkDocumentServer struct {
	grpc.ServerStream
	ctx      context.Context
	req      *pb.ChunkDocumentRequest
	resp     *pb.ChunkDocumentResponse
	reqSent  bool
}

func (m *mockChunkDocumentServer) Context() context.Context {
	if m.ctx == nil {
		return context.Background()
	}
	return m.ctx
}

func (m *mockChunkDocumentServer) Send(resp *pb.ChunkDocumentResponse) error {
	m.resp = resp
	return nil
}

func (m *mockChunkDocumentServer) Recv() (*pb.ChunkDocumentRequest, error) {
	if !m.reqSent {
		m.reqSent = true
		return m.req, nil
	}
	return nil, io.EOF
}
