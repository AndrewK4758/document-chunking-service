package storage

import (
	"context"
	"io"

	pb "github.com/AndrewK4758/shared_protos"
)

type Manager interface {
	FetchFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity) ([]byte, error)
	FetchStream(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity) (io.ReadCloser, error)
	SaveFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity, data []byte) (string, error)
}
