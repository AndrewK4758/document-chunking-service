package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/AndrewK4758/shared_protos"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <path_to_pdf>")
		os.Exit(1)
	}

	pdfPath := os.Args[1]
	pdfBytes, err := os.ReadFile(pdfPath)
	if err != nil {
		log.Fatalf("Failed to read PDF file: %v", err)
	}

	fileName := filepath.Base(pdfPath)

	// Connect to Storage Gateway
	gatewayConn, err := grpc.NewClient("localhost:11118", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to Storage Gateway (localhost:11118): %v", err)
	}
	defer gatewayConn.Close()

	gatewayClient := pb.NewStorageGatewayServiceClient(gatewayConn)

	// Mock Identity
	identity := &pb.InfrastructureIdentity{
		TenantId: "test-tenant",
		AppId:    "test-ml-script",
		JobId:    "job-12345",
	}

	fmt.Printf("Uploading %s to Storage Gateway...\n", fileName)
	uploadStream, err := gatewayClient.UploadDocument(context.Background())
	if err != nil {
		log.Fatalf("Failed to open upload stream: %v", err)
	}

	// Send metadata
	err = uploadStream.Send(&pb.UploadRequest{
		UploadElement: &pb.UploadRequest_Metadata{
			Metadata: &pb.UploadMetadata{
				Identity:         identity,
				OriginalFilename: fileName,
			},
		},
	})
	if err != nil {
		log.Fatalf("Failed to send metadata: %v", err)
	}

	// Send chunk (we send the whole thing as one chunk for this test script)
	err = uploadStream.Send(&pb.UploadRequest{
		UploadElement: &pb.UploadRequest_ChunkData{
			ChunkData: pdfBytes,
		},
	})
	if err != nil {
		log.Fatalf("Failed to send chunk: %v", err)
	}

	uploadResp, err := uploadStream.CloseAndRecv()
	if err != nil {
		log.Fatalf("Upload failed: %v", err)
	}

	if !uploadResp.Success {
		log.Fatalf("Storage Gateway rejected upload: %s", uploadResp.ErrorMessage)
	}

	storageUri := uploadResp.StorageUri
	fmt.Printf("Uploaded successfully! Storage URI: %s\n", storageUri)

	// Connect to C++ ML Worker
	cppConn, err := grpc.NewClient("127.0.0.1:50060", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to C++ ML Worker (127.0.0.1:50060): %v", err)
	}
	defer cppConn.Close()

	mlClient := pb.NewCppMlWorkerServiceClient(cppConn)

	fmt.Println("Triggering ML Redaction...")
	mlResp, err := mlClient.RedactDocument(context.Background(), &pb.RedactDocumentRequest{
		DocumentUri: storageUri,
		Identity:    identity,
	})
	if err != nil {
		log.Fatalf("RPC call to ML Worker failed: %v", err)
	}

	if !mlResp.Success {
		log.Fatalf("ML Worker failed to process document: %s", mlResp.ErrorMessage)
	}

	fmt.Println("ML Redaction completed successfully!")

	// Reconstruct the redacted URI
	redactedUri := "internal-store://" + identity.TenantId + "/" + identity.AppId + "/" + identity.JobId + "/redacted_" + fileName

	fmt.Printf("Downloading redacted file from: %s\n", redactedUri)
	downloadStream, err := gatewayClient.DownloadDocument(context.Background(), &pb.DownloadRequest{
		StorageUri: redactedUri,
		Identity:   identity,
	})
	if err != nil {
		log.Fatalf("Failed to start download stream: %v", err)
	}

	var downloadedBytes []byte
	for {
		chunk, err := downloadStream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Error downloading chunk: %v", err)
		}
		downloadedBytes = append(downloadedBytes, chunk.ChunkData...)
	}

	outFile := strings.TrimSuffix(pdfPath, filepath.Ext(pdfPath)) + "_redacted.pdf"
	err = os.WriteFile(outFile, downloadedBytes, 0644)
	if err != nil {
		log.Fatalf("Failed to save redacted PDF: %v", err)
	}

	fmt.Printf("Done! Saved redacted PDF to: %s\n", outFile)
}
