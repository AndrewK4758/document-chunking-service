package storage

import (
	"context"
	"testing"
)

func TestLocalStorage_FetchAndSave(t *testing.T) {
	tempDir := t.TempDir()
	store, err := NewLocalStorage(tempDir)
	if err != nil {
		t.Fatalf("Failed to create local storage: %v", err)
	}

	ctx := context.Background()
	relativePath := "job_123/file.txt"
	testData := []byte("hello world")

	// Save
	savedPath, err := store.SaveFile(ctx, relativePath, testData)
	if err != nil {
		t.Fatalf("Failed to save file: %v", err)
	}
	if savedPath != relativePath {
		t.Errorf("Expected saved path %s, got %s", relativePath, savedPath)
	}

	// Fetch
	fetchedData, err := store.FetchFile(ctx, relativePath)
	if err != nil {
		t.Fatalf("Failed to fetch file: %v", err)
	}
	if string(fetchedData) != string(testData) {
		t.Errorf("Expected data %s, got %s", testData, fetchedData)
	}
}

func TestLocalStorage_InvalidBase(t *testing.T) {
	_, err := NewLocalStorage("")
	if err == nil {
		t.Fatal("Expected error for empty base path")
	}
}

func TestLocalStorage_ContextCancel(t *testing.T) {
	tempDir := t.TempDir()
	store, _ := NewLocalStorage(tempDir)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := store.FetchFile(ctx, "file.txt")
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}

	_, err = store.SaveFile(ctx, "file.txt", []byte("data"))
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func TestLocalStorage_FetchMissingFile(t *testing.T) {
	tempDir := t.TempDir()
	store, _ := NewLocalStorage(tempDir)

	_, err := store.FetchFile(context.Background(), "missing.txt")
	if err == nil {
		t.Fatal("Expected error fetching missing file")
	}
}
