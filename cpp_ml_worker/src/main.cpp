#include <iostream>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <string>
#include <cstdlib>

#include <spdlog/spdlog.h>

#include <grpcpp/grpcpp.h>
#include "ml_worker.grpc.pb.h"
#include "StorageGatewayClient.h"
#include "MuPDFWrapper.h"
#include "MLInferenceEngine.h"
#include "FzContextPool.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using ml::worker::v1::CppMlWorkerService;
using ml::worker::v1::RedactDocumentRequest;
using ml::worker::v1::RedactDocumentResponse;

static bool load_env_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
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
    return true;
}

#include <csignal>
#include <thread>

namespace {
    class CppMlWorkerServiceImpl final : public CppMlWorkerService::Service {
    public:
        CppMlWorkerServiceImpl(std::shared_ptr<FzContextPool> pool,
                               const std::string& model_path, const std::string& tokenizer_path, const std::string& storage_gateway_uri)
            : pool_(std::move(pool)) {
            if (!pool_) {
                throw std::runtime_error("FzContextPool is null");
            }
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
                spdlog::info("[JOB: {}] Starting ML Redaction for URI: {}", job_id, request->document_uri());

                spdlog::info("[JOB: {}] 1. Downloading document from Storage Gateway...", job_id);
                const std::string pdf_data = storage_client_->Download(request->document_uri(), request->identity());

                spdlog::info("[JOB: {}] 2. Initializing MuPDF context...", job_id);
                const MuPDFWrapper mupdf(pdf_data, pool_);
                const int page_count = mupdf.GetPageCount();
                spdlog::info("[JOB: {}]    -> Document loaded successfully with {} pages.", job_id, page_count);

                std::vector<std::string> labels;
                labels.reserve(request->dynamic_labels_size());
                for (const auto& l : request->dynamic_labels()) {
                    labels.push_back(l);
                }

                spdlog::info("[JOB: {}] 3. Beginning page-by-page ML inference and redaction with {} dynamic labels...", job_id, labels.size());
                for (int i = 0; i < page_count; i++) {
                    spdlog::info("[JOB: {}]    -> [Page {}/{}] Extracting text and coordinates...", job_id, i+1, page_count);
                    auto [text, char_bboxes] = mupdf.ExtractPageText(i);
                    
                    spdlog::info("[JOB: {}]    -> [Page {}/{}] Running GLiNER ML Inference...", job_id, i+1, page_count);
                    std::vector<fz_rect> bboxes = engine_->CalculateRedactionBBoxes(text, char_bboxes, labels);
                    
                    spdlog::info("[JOB: {}]    -> [Page {}/{}] Identified {} PII elements. Applying redaction blocks...", job_id, i+1, page_count, bboxes.size());
                    mupdf.ApplyRedactions(i, bboxes);
                }

                spdlog::info("[JOB: {}] 4. Serializing redacted PDF document back to memory...", job_id);
                std::string redacted_pdf_data = mupdf.GetRedactedPdf();
                
                spdlog::info("[JOB: {}] 5. Uploading final redacted document to Storage Gateway...", job_id);
                std::string storage_uri = storage_client_->Upload("redacted_" + request->document_uri(), request->identity(), redacted_pdf_data);

                spdlog::info("[JOB: {}] ML Redaction Process Completed Successfully! Saved as: {}", job_id, storage_uri);

                response->set_success(true);
                response->set_error_message("");
                return Status::OK;

            } catch (const std::invalid_argument& ex) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string detail = job_id.empty() ? ex.what() : "[JOB: " + job_id + "] " + ex.what();
                std::string traceparent;
                if (context) {
                    auto meta = context->client_metadata();
                    auto it = meta.find("traceparent");
                    if (it != meta.end()) traceparent = std::string(it->second.data(), it->second.size());
                    auto it2 = meta.find("grpc-trace-bin");
                    if (traceparent.empty() && it2 != meta.end()) traceparent = std::string(it2->second.data(), it2->second.size());
                }
                try {
                    spdlog::error("[JOB: {}] gRPC RedactDocument failed code=INVALID_ARGUMENT traceparent={} what={}", job_id.empty() ? "unknown" : job_id, traceparent.empty() ? "-" : traceparent, ex.what());
                } catch (...) {}
                return Status(grpc::StatusCode::INVALID_ARGUMENT, detail);
            } catch (const std::out_of_range& ex) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string detail = job_id.empty() ? ex.what() : "[JOB: " + job_id + "] " + ex.what();
                std::string traceparent;
                if (context) {
                    auto meta = context->client_metadata();
                    auto it = meta.find("traceparent");
                    if (it != meta.end()) traceparent = std::string(it->second.data(), it->second.size());
                    auto it2 = meta.find("grpc-trace-bin");
                    if (traceparent.empty() && it2 != meta.end()) traceparent = std::string(it2->second.data(), it2->second.size());
                }
                try {
                    spdlog::error("[JOB: {}] gRPC RedactDocument failed code=INVALID_ARGUMENT traceparent={} what={}", job_id.empty() ? "unknown" : job_id, traceparent.empty() ? "-" : traceparent, ex.what());
                } catch (...) {}
                return Status(grpc::StatusCode::INVALID_ARGUMENT, detail);
            } catch (const std::exception& ex) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string detail = job_id.empty() ? ex.what() : "[JOB: " + job_id + "] " + ex.what();
                std::string traceparent;
                if (context) {
                    auto meta = context->client_metadata();
                    auto it = meta.find("traceparent");
                    if (it != meta.end()) traceparent = std::string(it->second.data(), it->second.size());
                    auto it2 = meta.find("grpc-trace-bin");
                    if (traceparent.empty() && it2 != meta.end()) traceparent = std::string(it2->second.data(), it2->second.size());
                }
                try {
                    spdlog::error("[JOB: {}] gRPC RedactDocument failed code=INTERNAL traceparent={} what={}", job_id.empty() ? "unknown" : job_id, traceparent.empty() ? "-" : traceparent, ex.what());
                } catch (...) {}
                return Status(grpc::StatusCode::INTERNAL, detail);
            } catch (...) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string detail = job_id.empty() ? "Unknown exception" : "[JOB: " + job_id + "] Unknown exception";
                std::string traceparent;
                if (context) {
                    auto meta = context->client_metadata();
                    auto it = meta.find("traceparent");
                    if (it != meta.end()) traceparent = std::string(it->second.data(), it->second.size());
                    auto it2 = meta.find("grpc-trace-bin");
                    if (traceparent.empty() && it2 != meta.end()) traceparent = std::string(it2->second.data(), it2->second.size());
                }
                try {
                    spdlog::error("[JOB: {}] gRPC RedactDocument failed code=INTERNAL traceparent={} what=Unknown exception", job_id.empty() ? "unknown" : job_id, traceparent.empty() ? "-" : traceparent);
                } catch (...) {}
                return Status(grpc::StatusCode::INTERNAL, detail);
            }
        }

    private:
        std::shared_ptr<FzContextPool> pool_;
        std::unique_ptr<StorageGatewayClient> storage_client_;
        std::unique_ptr<IMLInferenceEngine> engine_;
    };
}

int main(int argc, char** argv) {
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

    if (!load_env_file(".env") && !load_env_file("../.env")) {
        spdlog::warn("Failed to open environment file (.env or ../.env)");
    }
    
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Starting C++ ML Worker...");

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
    auto pool = std::make_shared<FzContextPool>(FzContextPool::ResolveCapacity());
    CppMlWorkerServiceImpl service(pool, model_path, tokenizer_path, storage_gateway_uri);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server = builder.BuildAndStart();
    if (!server) {
        spdlog::error("Failed to bind and start gRPC server on {}", server_address);
        return 1;
    }

    std::jthread signal_handler_thread([&sigset, server_ptr = server.get()]() {
        int signum = 0;
        if (sigwait(&sigset, &signum) == 0) {
            spdlog::info("Received signal {}. Shutting down server cleanly...", signum);
            server_ptr->Shutdown();
        }
    });

    spdlog::info("Server listening on {}", server_address);
    server->Wait();
    
    spdlog::info("Server stopped cleanly.");
    return 0;
}
