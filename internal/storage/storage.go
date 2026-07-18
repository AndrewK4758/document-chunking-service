package storage

import (
	"context"
)

type Manager interface {
	FetchFile(ctx context.Context, relativePath string) ([]byte, error)
	SaveFile(ctx context.Context, relativePath string, data []byte) (string, error)
}
