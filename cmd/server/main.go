package main

import (
	"log"
	"net"
	"os"

	"github.com/joho/godotenv"
	"google.golang.org/grpc"

	pb "github.com/AndrewK4758/shared_protos"
	"document_processing_service/internal/server"
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

	// Register the DocumentProcessingService
	docServer := server.NewDocumentProcessingServer(aiClient)
	pb.RegisterDocumentProcessingServiceServer(s, docServer)

	log.Printf("Document Processing Service listening on %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
