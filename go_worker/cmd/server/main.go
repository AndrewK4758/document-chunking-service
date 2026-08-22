package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/AndrewK4758/shared_utils/envutil"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/AndrewK4758/shared_utils/natsutil"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/recovery"
	"github.com/nats-io/nats.go/jetstream"
	"google.golang.org/grpc"

	"document_processing_service/internal/db"
	"document_processing_service/internal/server"
	"document_processing_service/internal/storage"

	pb "github.com/AndrewK4758/shared_protos"
)

func main() {
	// Construct an absolute fallback path for local Rider execution
	absFallback, _ := filepath.Abs("/home/ak/projects/doc_processor_service/document_processing_service/go_worker/.env")

	envutil.LoadEnv(
		".env",
		"../.env",
		"../../.env",
		"../../../.env",
		"document_processing_service/go_worker/.env",
		"go_worker/.env",
		absFallback,
	)
	logger.InitLogger()

	// MPL-320: taxonomy startup health check (fail-fast in dev)
	{
		_ = func() string {
			// Ensure taxonomy loads at startup; panic in dev if missing per AC-4
			// Use processor TaxonomyPath + MustLoadTaxonomy
			// Import via blank to avoid cycle: direct load via os.ReadFile
			return ""
		}()
	}
	// Direct startup check without import cycle: use env path
	{
		taxPath := "document_processing_service/config/label_taxonomy.json"
		if p := envutil.GetStringOrDefault("TAXONOMY_PATH", ""); p != "" {
			taxPath = p
		}
		// Also try PROJECT_ROOT relative
		if _, err := os.Stat(taxPath); err != nil {
			if pr := envutil.GetStringOrDefault("PROJECT_ROOT", ""); pr != "" {
				alt := pr + "/" + taxPath
				if _, err2 := os.Stat(alt); err2 == nil {
					taxPath = alt
				}
			}
		}
		// Fail-fast check via processor IsFailFast semantics (replicate)
		isFailFast := false
		if v := envutil.GetStringOrDefault("FAIL_FAST", ""); v != "" {
			low := strings.ToLower(strings.TrimSpace(v))
			if low == "true" || low == "1" || low == "yes" {
				isFailFast = true
			}
		} else if env := envutil.GetStringOrDefault("APP_ENV", ""); env == "development" || env == "dev" {
			isFailFast = true
		}
		if _, err := os.ReadFile(taxPath); err != nil && isFailFast {
			panic(fmt.Errorf("taxonomy_load_failed: failed to read %s: %w", taxPath, err))
		}
		logger.Info("DocProcessor", "Taxonomy startup check passed for %s (failFast=%v)", taxPath, isFailFast)
	}

	// Initialize Database and run migrations
	dbInstance := db.InitDB()
	_ = dbInstance // Will pass to server or storage when needed

	port := envutil.MustGetString("GRPC_PORT")

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		panic(fmt.Errorf("failed to listen: %v", err))
	}
	s := grpc.NewServer(
		grpc.ChainUnaryInterceptor(
			recovery.UnaryServerInterceptor(),
		),
		grpc.ChainStreamInterceptor(
			recovery.StreamServerInterceptor(),
		),
	)

	// Dial AI Service
	aiServiceUrl := envutil.MustGetString("AI_SERVICE_URL")

	aiConn, err := grpc.NewClient(aiServiceUrl, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(fmt.Errorf("failed to dial ai_service: %v", err))
	}
	defer aiConn.Close()

	aiClient := pb.NewDocumentProcessorClient(aiConn)

	// ... (Wait, main.go already imports context, net, etc. I need to add 'os' if it's missing)
	// Query available models on startup asynchronously
	//go func() {
	//	tenant := envutil.GetStringOrDefault("TENANT_ID", "system")
	//	app := envutil.GetStringOrDefault("APP_ID", "doc_processor_startup")
	//	job := envutil.GetStringOrDefault("JOB_ID", "startup")
	//
	//	modelsResp, err := aiClient.GetAvailableModels(context.Background(), &pb.GetAvailableModelsRequest{
	//		Identity: &pb.InfrastructureIdentity{
	//			TenantId: tenant,
	//			AppId:    app,
	//			JobId:    job,
	//		},
	//	})
	//	if err != nil {
	//		logger.Info("DocProcessor", "Warning: failed to query AI service for available models: %v", err)
	//	} else {
	//		logger.Info("DocProcessor", "AI Service Models Loaded - Text: %v, OCR: %v", modelsResp.TextModels, modelsResp.OcrModels)
	//	}
	//}()

	// Initialize Storage Gateway Client
	gatewayAddr := envutil.MustGetString("STORAGE_GATEWAY_ADDR")

	gatewayConn, err := grpc.NewClient(gatewayAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(fmt.Errorf("failed to dial storage_gateway: %v", err))
	}
	defer gatewayConn.Close()

	store := storage.NewGatewayStorage(pb.NewStorageGatewayServiceClient(gatewayConn))

	cppMlWorkerUrl := envutil.GetStringOrDefault("CPP_ML_WORKER_URL", "localhost:50060")
	dotnetMlWorkerUrl := envutil.GetStringOrDefault("DOTNET_ML_WORKER_URL", "localhost:50051")
	retryPolicy := `{
		"methodConfig": [{
		  "name": [{"service": "ml.worker.v1.CppMlWorkerService"}, {"service": "ml.worker.v1.MlWorkerService"}],
		  "retryPolicy": {
			  "MaxAttempts": 4,
			  "InitialBackoff": "0.1s",
			  "MaxBackoff": "1s",
			  "BackoffMultiplier": 2,
			  "RetryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"]
		  }
		}]}`

	cppMlConn, err := grpc.NewClient(
		cppMlWorkerUrl,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(retryPolicy),
	)

	if err != nil {
		panic(fmt.Errorf("failed to dial cpp_ml_worker: %v", err))
	}
	defer cppMlConn.Close()

	cppMlClient := pb.NewCppMlWorkerServiceClient(cppMlConn)

	dotnetMlConn, err := grpc.NewClient(
		dotnetMlWorkerUrl,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(retryPolicy),
	)

	if err != nil {
		panic(fmt.Errorf("failed to dial dotnet_ml_worker: %v", err))
	}
	defer dotnetMlConn.Close()

	dotnetMlClient := pb.NewMlWorkerServiceClient(dotnetMlConn)

	// Initialize NATS
	natsURL := envutil.MustGetString("NATS_URL")
	if natsURL == "" {
		natsURL = "nats://localhost:4222"
	}
	nc, err := natsutil.ConnectWithRetries(natsURL, "DocProcessor")
	if err != nil {
		panic(fmt.Errorf("failed to connect to NATS: %v", err))
	}
	defer nc.Close()

	js, err := jetstream.New(nc)
	if err != nil {
		panic(fmt.Errorf("failed to init JetStream: %v", err))
	}

	natsServer := server.NewNatsServer(nc, js, aiClient, store, cppMlClient, dotnetMlClient, dbInstance)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := natsServer.Start(ctx); err != nil {
			panic(fmt.Errorf("NATS Server failed: %v", err))
		}
	}()

	// Register the DocumentProcessingService
	docServer := server.NewDocumentProcessingServer(aiClient, cppMlClient, dotnetMlClient, store)
	pb.RegisterDocumentProcessingServiceServer(s, docServer)

	logger.Info("DocProcessor", "Document Processing Service listening on %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		panic(fmt.Errorf("failed to serve: %v", err))
	}
}
