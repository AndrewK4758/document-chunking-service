#include "MLInferenceEngine.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "tokenizers_cpp/tokenizers_cpp.h"

MLInferenceEngine::MLInferenceEngine(std::unique_ptr<ITokenizer> tokenizer, std::unique_ptr<IOnnxRunner> onnx_runner)
    : tokenizer_(std::move(tokenizer)), onnx_runner_(std::move(onnx_runner)) {
    if (!tokenizer_ || !onnx_runner_) {
        throw std::runtime_error("MLInferenceEngine initialized with null components");
    }
}

std::vector<fz_rect> MLInferenceEngine::CalculateRedactionBBoxes(const std::string& text, const std::vector<fz_rect>& char_bboxes) {
    if (text.size() != char_bboxes.size()) {
        throw std::runtime_error("Text length does not match character bounding boxes length");
    }

    std::vector<int32_t> tokens = tokenizer_->Encode(text);
    std::vector<int> tags = onnx_runner_->RunInference(tokens);

    if (tokens.size() != tags.size()) {
        throw std::runtime_error("Tokens and tags size mismatch");
    }

    std::vector<fz_rect> redaction_bboxes;
    size_t char_idx = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string decoded = tokenizer_->Decode({tokens[i]});
        if (decoded.empty()) continue;

        size_t pos = text.find(decoded, char_idx);
        if (pos != std::string::npos) {
            if (tags[i] > 0) { // If tagged as PII
                fz_rect r = fz_empty_rect;
                for (size_t c = pos; c < pos + decoded.length(); ++c) {
                    if (fz_is_empty_rect(char_bboxes[c])) continue;
                    if (fz_is_empty_rect(r)) {
                        r = char_bboxes[c];
                    } else {
                        r = fz_union_rect(r, char_bboxes[c]);
                    }
                }
                if (!fz_is_empty_rect(r)) {
                    redaction_bboxes.push_back(r);
                }
            }
            char_idx = pos + decoded.length();
        }
    }

    return redaction_bboxes;
}

namespace {
    class RealTokenizer : public ITokenizer {
    public:
        explicit RealTokenizer(const std::string& path) {
            std::ifstream t(path);
            if (!t.is_open()) {
                throw std::runtime_error("Failed to open tokenizer.json at " + path);
            }
            std::stringstream buffer;
            buffer << t.rdbuf();
            tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(buffer.str());
            if (!tokenizer_) {
                throw std::runtime_error("Failed to parse tokenizer.json");
            }
        }

        std::vector<int32_t> Encode(const std::string& text) override {
            return tokenizer_->Encode(text);
        }

        std::string Decode(const std::vector<int32_t>& ids) override {
            return tokenizer_->Decode(ids);
        }

    private:
        std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    };
}

namespace {
    class RealOnnxRunner : public IOnnxRunner {
    public:
        explicit RealOnnxRunner(const std::string& path) : env_(ORT_LOGGING_LEVEL_WARNING, "MLWorkerEnv") {
            try {
                Ort::SessionOptions session_options;
                session_ = std::make_unique<Ort::Session>(env_, path.c_str(), session_options);
            } catch (const Ort::Exception& e) {
                throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
            }
        }

        std::vector<int> RunInference(const std::vector<int32_t>& tokens) override {
            if (tokens.empty()) return {};

            const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};
            std::vector<int64_t> token_ids(tokens.begin(), tokens.end());

            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, token_ids.data(), token_ids.size(), input_shape.data(), input_shape.size());

            const char* input_names[] = {"input_ids"};
            const char* output_names[] = {"logits"};

            auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

            const auto* logits = output_tensors[0].GetTensorMutableData<float>();
            std::vector<int> tags(tokens.size(), 0);
            for (size_t i = 0; i < tokens.size(); ++i) {
                tags[i] = (logits[i] > 0.5f) ? 1 : 0;
            }
            return tags;
        }

    private:
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
    };
}

std::unique_ptr<ITokenizer> CreateRealTokenizer(const std::string& path) {
    return std::make_unique<RealTokenizer>(path);
}

std::unique_ptr<IOnnxRunner> CreateRealOnnxRunner(const std::string& path) {
    return std::make_unique<RealOnnxRunner>(path);
}
