package main

import (
	"context"
	"log"
	"net"
	"os"

	"github.com/joho/godotenv"
	"google.golang.org/grpc"

	"document_processing_service/internal/server"
	"document_processing_service/internal/storage"
	pb "github.com/AndrewK4758/shared_protos"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using environment variables")
	}

	port := os.Getenv("GRPC_PORT")
	if port == "" {
		port = "50054" // Default port for document_processing_service
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()

	// Dial AI Service
	aiServiceUrl := os.Getenv("AI_SERVICE_URL")
	if aiServiceUrl == "" {
		aiServiceUrl = "localhost:50051" // Default AI Service URL
	}

	aiConn, err := grpc.Dial(aiServiceUrl, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial ai_service: %v", err)
	}
	defer aiConn.Close()

	aiClient := pb.NewDocumentProcessorClient(aiConn)

	// Query available models on startup
	modelsResp, err := aiClient.GetAvailableModels(context.Background(), &pb.GetAvailableModelsRequest{})
	if err != nil {
		log.Printf("Warning: failed to query AI service for available models: %v", err)
	} else {
		log.Printf("AI Service Models Loaded - Text: %v, OCR: %v", modelsResp.TextModels, modelsResp.OcrModels)
	}

	// Initialize Storage Gateway Client
	gatewayAddr := os.Getenv("STORAGE_GATEWAY_ADDR")
	if gatewayAddr == "" {
		gatewayAddr = "localhost:50058"
	}
	gatewayConn, err := grpc.Dial(gatewayAddr, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("failed to dial storage_gateway: %v", err)
	}
	defer gatewayConn.Close()

	store := storage.NewGatewayStorage(pb.NewStorageGatewayServiceClient(gatewayConn))

	// Register the DocumentProcessingService
	docServer := server.NewDocumentProcessingServer(aiClient, store)
	pb.RegisterDocumentProcessingServiceServer(s, docServer)

	log.Printf("Document Processing Service listening on %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
