package storage

import (
	"context"
	"fmt"
	"io"
	"strings"

	pb "github.com/AndrewK4758/shared_protos"
	"google.golang.org/grpc/metadata"
)

type GatewayStorage struct {
	client pb.StorageGatewayServiceClient
}

func NewGatewayStorage(client pb.StorageGatewayServiceClient) *GatewayStorage {
	return &GatewayStorage{
		client: client,
	}
}

// forwardMetadata takes incoming gRPC metadata from the context and forwards it to the outgoing context.
func forwardMetadata(ctx context.Context) context.Context {
	md, ok := metadata.FromIncomingContext(ctx)
	if ok {
		return metadata.NewOutgoingContext(ctx, md)
	}
	return ctx
}

func (g *GatewayStorage) FetchFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	reqCtx := forwardMetadata(ctx)
	req := &pb.DownloadRequest{
		StorageUri: relativePath, // For gateway fetch, the relative path is the actual internal-store:// URI
		Identity:   identity,
	}

	stream, err := g.client.DownloadDocument(reqCtx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to initiate download stream for %s: %w", relativePath, err)
	}

	var data []byte
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to receive chunk for %s: %w", relativePath, err)
		}
		data = append(data, resp.ChunkData...)
	}

	return data, nil
}

func (g *GatewayStorage) SaveFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	// Parse relativePath which is formatted as "jobID/filename.pdf" by the processor
	parts := strings.SplitN(relativePath, "/", 2)
	if len(parts) != 2 {
		return "", fmt.Errorf("invalid save path format, expected jobID/filename, got: %s", relativePath)
	}
	filename := parts[1]

	reqCtx := forwardMetadata(ctx)
	stream, err := g.client.UploadDocument(reqCtx)
	if err != nil {
		return "", fmt.Errorf("failed to initiate upload stream: %w", err)
	}

	// Send metadata chunk first
	err = stream.Send(&pb.UploadRequest{
		UploadElement: &pb.UploadRequest_Metadata{
			Metadata: &pb.UploadMetadata{
				Identity:         identity,
				OriginalFilename: filename,
			},
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to send upload metadata for %s: %w", filename, err)
	}

	// Send data in chunks (64KB chunks)
	chunkSize := 64 * 1024
	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}

		err = stream.Send(&pb.UploadRequest{
			UploadElement: &pb.UploadRequest_ChunkData{
				ChunkData: data[i:end],
			},
		})
		if err != nil {
			return "", fmt.Errorf("failed to send chunk for %s: %w", filename, err)
		}
	}

	// Close stream and receive the storage URI
	resp, err := stream.CloseAndRecv()
	if err != nil {
		return "", fmt.Errorf("failed to close upload stream for %s: %w", filename, err)
	}

	if !resp.Success {
		return "", fmt.Errorf("gateway rejected upload for %s: %s", filename, resp.ErrorMessage)
	}

	return resp.StorageUri, nil
}
