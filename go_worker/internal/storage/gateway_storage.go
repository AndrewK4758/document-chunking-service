package storage

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	pb "github.com/AndrewK4758/shared_protos"
	"github.com/AndrewK4758/shared_utils/envutil"
	"google.golang.org/grpc/metadata"
)

type GatewayStorage struct {
	client       pb.StorageGatewayServiceClient
	retryCount   int
	retrySleepMs int
}

func NewGatewayStorage(client pb.StorageGatewayServiceClient) *GatewayStorage {
	return &GatewayStorage{
		client:       client,
		retryCount:   envutil.MustGetInt("STORAGE_FETCH_RETRY_COUNT"),
		retrySleepMs: envutil.MustGetInt("STORAGE_FETCH_RETRY_SLEEP_MS"),
	}
}

func NewGatewayStorageWithRetry(client pb.StorageGatewayServiceClient, retryCount, retrySleepMs int) *GatewayStorage {
	return &GatewayStorage{
		client:       client,
		retryCount:   retryCount,
		retrySleepMs: retrySleepMs,
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

	retryCount := g.retryCount
	if retryCount < 0 {
		retryCount = envutil.MustGetInt("STORAGE_FETCH_RETRY_COUNT")
	}

	retrySleepMs := g.retrySleepMs
	if retrySleepMs <= 0 {
		retrySleepMs = envutil.MustGetInt("STORAGE_FETCH_RETRY_SLEEP_MS")
	}

	reqCtx := forwardMetadata(ctx)
	req := &pb.DownloadRequest{
		StorageUri: relativePath,
		Identity:   identity,
	}

	var lastErr error
	for attempt := 0; attempt <= retryCount; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if attempt > 0 {
			backoffMs := retrySleepMs * (1 << (attempt - 1))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(backoffMs) * time.Millisecond):
			}
		}

		stream, err := g.client.DownloadDocument(reqCtx, req)
		if err != nil {
			lastErr = fmt.Errorf("failed to initiate download stream for %s (attempt %d/%d): %w", relativePath, attempt+1, retryCount+1, err)
			continue
		}

		data, streamErr := g.receiveStreamChunks(stream, relativePath)
		if streamErr != nil {
			lastErr = fmt.Errorf("failed during chunk receive for %s (attempt %d/%d): %w", relativePath, attempt+1, retryCount+1, streamErr)
			continue
		}

		return data, nil
	}

	return nil, fmt.Errorf("storage fetch failed for %s after %d retries: %w", relativePath, retryCount, lastErr)
}

type streamReader struct {
	stream pb.StorageGatewayService_DownloadDocumentClient
	buf    []byte
	cancel context.CancelFunc
}

func (r *streamReader) Read(p []byte) (n int, err error) {
	if len(r.buf) > 0 {
		n = copy(p, r.buf)
		r.buf = r.buf[n:]
		return n, nil
	}
	resp, err := r.stream.Recv()
	if err == io.EOF {
		return 0, io.EOF
	}
	if err != nil {
		return 0, err
	}
	n = copy(p, resp.ChunkData)
	if n < len(resp.ChunkData) {
		r.buf = append(r.buf, resp.ChunkData[n:]...)
	}
	return n, nil
}

func (r *streamReader) Close() error {
	if r.cancel != nil {
		r.cancel()
	}
	return nil
}

func (g *GatewayStorage) FetchStream(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity) (io.ReadCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	reqCtx, cancel := context.WithCancel(forwardMetadata(ctx))
	req := &pb.DownloadRequest{
		StorageUri: relativePath,
		Identity:   identity,
	}

	stream, err := g.client.DownloadDocument(reqCtx, req)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to initiate download stream for %s: %w", relativePath, err)
	}

	return &streamReader{
		stream: stream,
		buf:    nil,
		cancel: cancel,
	}, nil
}

func (g *GatewayStorage) receiveStreamChunks(stream pb.StorageGatewayService_DownloadDocumentClient, relativePath string) ([]byte, error) {
	var data []byte
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			return data, nil
		}
		if err != nil {
			return nil, fmt.Errorf("failed to receive chunk for %s: %w", relativePath, err)
		}
		data = append(data, resp.ChunkData...)
	}
}

func (g *GatewayStorage) SaveFile(ctx context.Context, relativePath string, identity *pb.InfrastructureIdentity, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	// Parse relativePath formatted as "jobID/filename.pdf"
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

	chunkSize := 64 * 1024
	for i := 0; i < len(data); i += chunkSize {
		end := min(i+chunkSize, len(data))

		err = stream.Send(&pb.UploadRequest{
			UploadElement: &pb.UploadRequest_ChunkData{
				ChunkData: data[i:end],
			},
		})
		if err != nil {
			return "", fmt.Errorf("failed to send chunk for %s: %w", filename, err)
		}
	}

	resp, err := stream.CloseAndRecv()
	if err != nil {
		return "", fmt.Errorf("failed to close upload stream for %s: %w", filename, err)
	}

	if !resp.Success {
		return "", fmt.Errorf("gateway rejected upload for %s: %s", filename, resp.ErrorMessage)
	}

	return resp.StorageUri, nil
}
