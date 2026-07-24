package main

import (
	"context"
	"net"

	"github.com/AndrewK4758/shared_utils/envutil"

	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/AndrewK4758/shared_utils/natsutil"

	"github.com/nats-io/nats.go/jetstream"
	"google.golang.org/grpc"

	"document_processing_service/internal/server"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
)

func main() {
	logger.InitLogger()
	envutil.LoadEnv(".env", "document_processing_service/.env", "../.env", "../../.env")

	port := envutil.GetStringOrDefault("GRPC_PORT", "")
	if port == "" {
		port = "50054" // Default port for document_processing_service
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		logger.Fatal("DocProcessor", "failed to listen: %v", err)
	}

	s := grpc.NewServer()

	// Dial AI Service
	aiServiceUrl := envutil.GetStringOrDefault("AI_SERVICE_URL", "")
	if aiServiceUrl == "" {
		aiServiceUrl = "localhost:50051" // Default AI Service URL
	}

	aiConn, err := grpc.Dial(aiServiceUrl, grpc.WithInsecure())
	if err != nil {
		logger.Fatal("DocProcessor", "failed to dial ai_service: %v", err)
	}
	defer aiConn.Close()

	aiClient := pb.NewDocumentProcessorClient(aiConn)

	// Query available models on startup
	modelsResp, err := aiClient.GetAvailableModels(context.Background(), &pb.GetAvailableModelsRequest{
		Identity: &pb.InfrastructureIdentity{
			TenantId: "system",
			AppId:    "doc_processor_startup",
			JobId:    "startup",
		},
	})
	if err != nil {
		logger.Info("DocProcessor", "Warning: failed to query AI service for available models: %v", err)
	} else {
		logger.Info("DocProcessor", "AI Service Models Loaded - Text: %v, OCR: %v", modelsResp.TextModels, modelsResp.OcrModels)
	}

	// Initialize Storage Gateway Client
	gatewayAddr := envutil.GetStringOrDefault("STORAGE_GATEWAY_ADDR", "")
	if gatewayAddr == "" {
		gatewayAddr = "localhost:50058"
	}
	gatewayConn, err := grpc.Dial(gatewayAddr, grpc.WithInsecure())
	if err != nil {
		logger.Fatal("DocProcessor", "failed to dial storage_gateway: %v", err)
	}
	defer gatewayConn.Close()

	store := storage.NewGatewayStorage(pb.NewStorageGatewayServiceClient(gatewayConn))

	// Initialize NATS
	natsURL := envutil.GetStringOrDefault("NATS_URL", "")
	if natsURL == "" {
		natsURL = "nats://localhost:4222"
	}
	nc, err := natsutil.ConnectWithRetries(natsURL, "DocProcessor")
	if err != nil {
		logger.Fatal("DocProcessor", "failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	js, err := jetstream.New(nc)
	if err != nil {
		logger.Fatal("DocProcessor", "failed to init JetStream: %v", err)
	}

	natsServer := server.NewNatsServer(nc, js, aiClient, store)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := natsServer.Start(ctx); err != nil {
			logger.Fatal("DocProcessor", "NATS Server failed: %v", err)
		}
	}()

	// Register the DocumentProcessingService
	docServer := server.NewDocumentProcessingServer(aiClient, store)
	pb.RegisterDocumentProcessingServiceServer(s, docServer)

	logger.Info("DocProcessor", "Document Processing Service listening on %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		logger.Fatal("DocProcessor", "failed to serve: %v", err)
	}
}
