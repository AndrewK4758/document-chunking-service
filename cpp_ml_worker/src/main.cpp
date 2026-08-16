#include <iostream>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <string>
#include <cstdlib>

#include <grpcpp/grpcpp.h>
#include "ml_worker.grpc.pb.h"
#include "StorageGatewayClient.h"
#include "MuPDFWrapper.h"
#include "MLInferenceEngine.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using ml::worker::v1::CppMlWorkerService;
using ml::worker::v1::RedactDocumentRequest;
using ml::worker::v1::RedactDocumentResponse;

static void load_env_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open environment file: " << path << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (size_t pos = line.find('='); pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            setenv(key.c_str(), value.c_str(), 0);
        }
    }
}

#include <csignal>

namespace {
    std::unique_ptr<grpc::Server> g_server;

    void SignalHandler(int signum) {
        std::cout << "\nReceived signal " << signum << ". Shutting down server..." << std::endl;
        if (g_server) {
            g_server->Shutdown();
        }
    }
    
    class CppMlWorkerServiceImpl final : public CppMlWorkerService::Service {
    public:
        CppMlWorkerServiceImpl(const std::string& model_path, const std::string& tokenizer_path, const std::string& storage_gateway_uri) {
            storage_client_ = std::make_unique<StorageGatewayClient>(
                grpc::CreateChannel(storage_gateway_uri, grpc::InsecureChannelCredentials()));

            auto tokenizer = CreateRealTokenizer(tokenizer_path);
            auto onnx_runner = CreateRealOnnxRunner(model_path);
            engine_ = std::make_unique<MLInferenceEngine>(std::move(tokenizer), std::move(onnx_runner));
        }

        Status RedactDocument(ServerContext* context, const RedactDocumentRequest* request, RedactDocumentResponse* response) override {
            try {
                if (request->document_uri().empty()) {
                    throw std::runtime_error("Missing document_uri in request");
                }
                if (!request->has_identity()) {
                    throw std::runtime_error("Missing identity in request");
                }

                std::string job_id = request->identity().job_id();
                std::cout << "[JOB: " << job_id << "] Starting ML Redaction for URI: " << request->document_uri() << std::endl;

                std::cout << "[JOB: " << job_id << "] 1. Downloading document from Storage Gateway..." << std::endl;
                const std::string pdf_data = storage_client_->Download(request->document_uri(), request->identity());

                std::cout << "[JOB: " << job_id << "] 2. Initializing MuPDF context..." << std::endl;
                const MuPDFWrapper mupdf(pdf_data);
                const int page_count = mupdf.GetPageCount();
                std::cout << "[JOB: " << job_id << "]    -> Document loaded successfully with " << page_count << " pages." << std::endl;

                std::cout << "[JOB: " << job_id << "] 3. Beginning page-by-page ML inference and redaction..." << std::endl;
                for (int i = 0; i < page_count; i++) {
                    std::cout << "[JOB: " << job_id << "]    -> [Page " << (i+1) << "/" << page_count << "] Extracting text and coordinates..." << std::endl;
                    auto [text, char_bboxes] = mupdf.ExtractPageText(i);
                    
                    std::cout << "[JOB: " << job_id << "]    -> [Page " << (i+1) << "/" << page_count << "] Running GLiNER ML Inference..." << std::endl;
                    std::vector<fz_rect> bboxes = engine_->CalculateRedactionBBoxes(text, char_bboxes);
                    
                    std::cout << "[JOB: " << job_id << "]    -> [Page " << (i+1) << "/" << page_count << "] Identified " << bboxes.size() << " PII elements. Applying redaction blocks..." << std::endl;
                    mupdf.ApplyRedactions(i, bboxes);
                }

                std::cout << "[JOB: " << job_id << "] 4. Serializing redacted PDF document back to memory..." << std::endl;
                std::string redacted_pdf_data = mupdf.GetRedactedPdf();
                
                std::cout << "[JOB: " << job_id << "] 5. Uploading final redacted document to Storage Gateway..." << std::endl;
                std::string storage_uri = storage_client_->Upload("redacted_" + request->document_uri(), request->identity(), redacted_pdf_data);

                std::cout << "[JOB: " << job_id << "] ML Redaction Process Completed Successfully! Saved as: " << storage_uri << std::endl;

                response->set_success(true);
                response->set_error_message("");
                return Status::OK;

            } catch (const std::exception& ex) {
                std::cerr << "Fatal error during execution: " << ex.what() << std::endl;
                response->set_success(false);
                response->set_error_message(ex.what());
                throw; // Fail fast principle
            }
        }

    private:
        std::unique_ptr<StorageGatewayClient> storage_client_;
        std::unique_ptr<IMLInferenceEngine> engine_;
    };
}

int main(int argc, char** argv) {
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    load_env_file(".env");
    std::cout << "Starting C++ ML Worker..." << std::endl;

    // Provide default paths, these could be overridden by env vars
    std::string model_path = "/home/ak/projects/ml/model.onnx";
    std::string tokenizer_path = "/home/ak/projects/ml/tokenizer.json";
    
    if (const char* env_m = std::getenv("ONNX_MODEL_PATH")) {
        model_path = env_m;
    }
    if (const char* env_t = std::getenv("TOKENIZER_PATH")) {
        tokenizer_path = env_t;
    }

    std::string storage_gateway_uri = "storage_gateway:50051";
    if (const char* env_s = std::getenv("STORAGE_GATEWAY_URI")) {
        storage_gateway_uri = env_s;
    }

    std::string server_address("0.0.0.0:50060");
    if (const char* env_addr = std::getenv("SERVER_ADDRESS")) {
        server_address = env_addr;
    }
    CppMlWorkerServiceImpl service(model_path, tokenizer_path, storage_gateway_uri);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    g_server = builder.BuildAndStart();
    if (!g_server) {
        std::cerr << "Failed to bind and start gRPC server on " << server_address << std::endl;
        return 1;
    }
    
    std::cout << "Server listening on " << server_address << std::endl;
    g_server->Wait();
    
    std::cout << "Server stopped cleanly." << std::endl;
    return 0;
}
