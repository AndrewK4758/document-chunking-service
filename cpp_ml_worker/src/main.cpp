#include <iostream>
#include <memory>
#include <stdexcept>

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

namespace {
    class CppMlWorkerServiceImpl final : public CppMlWorkerService::Service {
    public:
        CppMlWorkerServiceImpl(const std::string& model_path, const std::string& tokenizer_path) {
            storage_client_ = std::make_unique<StorageGatewayClient>(
                grpc::CreateChannel("storage_gateway:50051", grpc::InsecureChannelCredentials()));

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

                std::string pdf_data = storage_client_->Download(request->document_uri(), request->identity());

                MuPDFWrapper mupdf(pdf_data);
                int page_count = mupdf.GetPageCount();

                for (int i = 0; i < page_count; i++) {
                    PageTextData ptd = mupdf.ExtractPageText(i);
                    std::vector<fz_rect> bboxes = engine_->CalculateRedactionBBoxes(ptd.text, ptd.char_bboxes);
                    mupdf.ApplyRedactions(i, bboxes);
                }

                std::string redacted_pdf_data = mupdf.GetRedactedPdf();
                std::string storage_uri = storage_client_->Upload("redacted_" + request->document_uri(), request->identity(), redacted_pdf_data);

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
    std::cout << "Starting C++ ML Worker..." << std::endl;

    // Provide default paths, these could be overridden by env vars
    std::string model_path = "model.onnx";
    std::string tokenizer_path = "tokenizer.json";
    
    if (const char* env_m = std::getenv("ONNX_MODEL_PATH")) {
        model_path = env_m;
    }
    if (const char* env_t = std::getenv("TOKENIZER_PATH")) {
        tokenizer_path = env_t;
    }

    std::string server_address("0.0.0.0:50060");
    CppMlWorkerServiceImpl service(model_path, tokenizer_path);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    server->Wait();
    return 0;
}
