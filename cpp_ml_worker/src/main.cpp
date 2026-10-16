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
#include "GrpcStatusHelper.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using ml::worker::v1::CppMlWorkerService;
using ml::worker::v1::RedactDocumentRequest;
using ml::worker::v1::RedactDocumentResponse;

#include "Logger.h"

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
                    throw std::invalid_argument("Missing document_uri in request");
                }
                if (!request->has_identity()) {
                    throw std::invalid_argument("Missing identity in request");
                }
                if (request->pass_count() < 0) {
                    throw std::invalid_argument("[SEAM: IMPOSSIBLE_STATE] pass_count must be >= 1");
                }
                int current_pass = request->pass_count() > 0 ? request->pass_count() : 1;

                std::string job_id = request->identity().job_id();
                spdlog::info("[JOB: {}] Starting ML Redaction for URI: {} (Pass: {})", job_id, request->document_uri(), current_pass);

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
                std::map<int, std::vector<fz_rect>> per_page_boxes;
                std::vector<std::string> target_plaintexts;
                for (int i = 0; i < page_count; i++) {
                    spdlog::info("[JOB: {}]    -> [Page {}/{}] Extracting text and coordinates...", job_id, i+1, page_count);
                    auto [text, char_bboxes] = mupdf.ExtractPageText(i);
                    
                    spdlog::info("[JOB: {}]    -> [Page {}/{}] Running GLiNER ML Inference...", job_id, i+1, page_count);
                    float threshold_override = request->confidence_threshold_override() > 0.0f ? request->confidence_threshold_override() : -1.0f;
                    std::vector<fz_rect> bboxes = engine_->CalculateRedactionBBoxes(text, char_bboxes, labels, threshold_override);
                    
                    for (const auto& zone : request->spatial_wipe_zones()) {
                        if (zone.page_number() == i + 1 || zone.page_number() == i) {
                            if (zone.x0() < 0 || zone.y0() < 0 || zone.x1() < 0 || zone.y1() < 0) {
                                spdlog::error("[JOB: {}] SpatialWipeZone negative coordinate on page {}: [{},{},{},{}]", job_id, zone.page_number(), zone.x0(), zone.y0(), zone.x1(), zone.y1());
                                throw std::invalid_argument("SpatialWipeZone contains negative coordinate: [" + std::to_string(zone.x0()) + "," + std::to_string(zone.y0()) + "," + std::to_string(zone.x1()) + "," + std::to_string(zone.y1()) + "]");
                            }
                            if (zone.x1() <= zone.x0() || zone.y1() <= zone.y0()) {
                                spdlog::error("[JOB: {}] SpatialWipeZone invalid rect on page {}: x1<=x0 or y1<=y0", job_id, zone.page_number());
                                throw std::invalid_argument("SpatialWipeZone invalid rect: x1 must be > x0 and y1 > y0");
                            }
                            fz_rect r = { zone.x0(), zone.y0(), zone.x1(), zone.y1() };
                            bboxes.push_back(r);
                        }
                    }
                    for (const auto& box : request->custom_boxes()) {
                        if (box.page_number() == i + 1 || box.page_number() == i) {
                            fz_rect r = { box.x0(), box.y0(), box.x1(), box.y1() };
                            bboxes.push_back(r);
                        }
                    }
                    
                    spdlog::info("[JOB: {}]    -> [Page {}/{}] Identified {} PII elements. Applying redaction blocks...", job_id, i+1, page_count, bboxes.size());
                    if (!bboxes.empty()) {
                        per_page_boxes[i] = bboxes;
                        if (!text.empty()) {
                            size_t ws = text.find(' ');
                            std::string preview = (ws == std::string::npos) ? text.substr(0, 16) : text.substr(0, std::min<size_t>(ws, 16));
                            if (!preview.empty()) target_plaintexts.push_back(preview);
                        }
                    }
                    mupdf.ApplyRedactions(i, bboxes);
                }

                spdlog::info("[JOB: {}] 4. Serializing redacted PDF document back to memory...", job_id);
                std::string redacted_pdf_data = mupdf.GetRedactedPdf();
                if (redacted_pdf_data.empty()) {
                    spdlog::error("[JOB: {}] Serialized redacted PDF data is empty (0 bytes), rejecting upload to Storage Gateway", job_id);
                    throw std::runtime_error("[SEAM: LINEAGE_GAP] RedactDocument: serialized PDF data is empty, rejecting 0-byte upload");
                }

                std::string traceparent;
                auto meta = context->client_metadata();
                auto it = meta.find("traceparent");
                if (it != meta.end()) traceparent = std::string(it->second.data(), it->second.size());
                else {
                    auto it2 = meta.find("grpc-trace-bin");
                    if (it2 != meta.end()) traceparent = std::string(it2->second.data(), it2->second.size());
                }

                auto report = mupdf.VerifyPostRedaction(redacted_pdf_data, per_page_boxes, target_plaintexts, job_id, traceparent, current_pass);

                spdlog::info("[JOB: {}] 5. Uploading redacted document to Storage Gateway...", job_id);
                // Extract basename for the storage key so URI schemes ("internal-store://...") don't cause path traversal
                const std::string& _uri = request->document_uri();
                size_t _pos = _uri.find_last_of("/\\");
                std::string _base = (_pos==std::string::npos? _uri : _uri.substr(_pos+1));
                if(_base.empty()) _base = "redacted.pdf";
                std::string storage_uri = storage_client_->Upload("redacted_" + _base, request->identity(), redacted_pdf_data);

                response->set_success(true);
                response->set_error_message("");
                {
                    int total_boxes = 0;
                    for (auto &val: per_page_boxes | std::views::values) total_boxes += val.size();
                    response->set_redacted_boxes_count(total_boxes);

                    auto* rep = response->mutable_residual_report();
                    rep->set_is_converged(report.is_converged);
                    rep->set_residual_value_count(report.residual_value_count);
                    rep->set_pass_count(report.pass_count);
                    for (const auto& sample : report.residual_samples) {
                        rep->add_residual_samples(sample);
                    }
                }

                if (report.is_converged) {
                    spdlog::info("[JOB: {}] ML Redaction Process Completed Successfully! Saved as: {}", job_id, storage_uri);
                } else {
                    spdlog::warn("[JOB: {}] ML Redaction Pass {} completed with {} residuals. Saved as: {}", job_id, current_pass, report.residual_value_count, storage_uri);
                }
                return Status::OK;

            } catch (const std::invalid_argument& ex) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
                std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
                std::string traceparent = doc_processor::ExtractTraceparent(context);
                return doc_processor::GrpcStatusFromException(ex, job_id, traceparent, "CppMlWorkerService/RedactDocument", tenant_id, app_id);
            } catch (const std::out_of_range& ex) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
                std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
                std::string traceparent = doc_processor::ExtractTraceparent(context);
                return doc_processor::GrpcStatusFromException(ex, job_id, traceparent, "CppMlWorkerService/RedactDocument", tenant_id, app_id);
            } catch (const std::exception& ex) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
                std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
                std::string traceparent = doc_processor::ExtractTraceparent(context);
                return doc_processor::GrpcStatusFromException(ex, job_id, traceparent, "CppMlWorkerService/RedactDocument", tenant_id, app_id);
            } catch (...) {
                std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
                std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
                std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
                std::string traceparent = doc_processor::ExtractTraceparent(context);
                return doc_processor::GrpcStatusFromUnknown(job_id, traceparent, "CppMlWorkerService/RedactDocument", tenant_id, app_id);
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

    doc_processor::InitEnv();
    doc_processor::InitLogger("cpp_ml_worker");
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
