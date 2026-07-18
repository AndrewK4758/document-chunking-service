package storage

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

type StorageManager interface {
	FetchFile(ctx context.Context, relativePath string) ([]byte, error)
	SaveFile(ctx context.Context, relativePath string, data []byte) (string, error)
}

type LocalStorage struct {
	basePath string
}

func NewLocalStorage(basePath string) (*LocalStorage, error) {
	if basePath == "" {
		return nil, fmt.Errorf("basePath cannot be empty for local storage")
	}

	// Ensure the base directory exists
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base storage directory: %w", err)
	}

	return &LocalStorage{basePath: basePath}, nil
}

func (l *LocalStorage) FetchFile(ctx context.Context, relativePath string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	fullPath := filepath.Join(l.basePath, relativePath)
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch file %s from storage: %w", relativePath, err)
	}

	return data, nil
}

func (l *LocalStorage) SaveFile(ctx context.Context, relativePath string, data []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	fullPath := filepath.Join(l.basePath, relativePath)

	// Create directories if they don't exist
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		return "", fmt.Errorf("failed to create directory for file %s: %w", relativePath, err)
	}

	if err := os.WriteFile(fullPath, data, 0644); err != nil {
		return "", fmt.Errorf("failed to write file %s to storage: %w", relativePath, err)
	}

	return relativePath, nil
}
