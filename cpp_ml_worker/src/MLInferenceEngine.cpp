#include "MLInferenceEngine.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "tokenizers_cpp/tokenizers_cpp.h"
#include "SpanGenerator.h"

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

    // Sanitize UTF-8 to prevent Rust tokenizers panic
    std::string sanitized_text;
    sanitized_text.reserve(text.size());
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];
        if (c <= 0x7F) {
            sanitized_text += text[i++];
        } else if (c >= 0xC2 && c <= 0xDF) {
            if (i + 1 < text.length() && (text[i+1] & 0xC0) == 0x80) {
                sanitized_text += text[i++]; sanitized_text += text[i++];
            } else { sanitized_text += '?'; i++; }
        } else if (c >= 0xE0 && c <= 0xEF) {
            if (i + 2 < text.length() && (text[i+1] & 0xC0) == 0x80 && (text[i+2] & 0xC0) == 0x80) {
                sanitized_text += text[i++]; sanitized_text += text[i++]; sanitized_text += text[i++];
            } else { sanitized_text += '?'; i++; }
        } else if (c >= 0xF0 && c <= 0xF4) {
            if (i + 3 < text.length() && (text[i+1] & 0xC0) == 0x80 && (text[i+2] & 0xC0) == 0x80 && (text[i+3] & 0xC0) == 0x80) {
                sanitized_text += text[i++]; sanitized_text += text[i++]; sanitized_text += text[i++]; sanitized_text += text[i++];
            } else { sanitized_text += '?'; i++; }
        } else {
            sanitized_text += '?'; i++;
        }
    }

    std::vector<int32_t> tokens = tokenizer_->Encode(sanitized_text);
    std::vector<int> tags = onnx_runner_->RunInference(tokens);

    if (tokens.size() != tags.size()) {
        throw std::runtime_error("Tokens and tags size mismatch");
    }

    std::vector<fz_rect> redaction_bboxes;
    size_t char_idx = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string decoded = tokenizer_->Decode({tokens[i]});
        if (decoded.empty()) continue;

        size_t pos = sanitized_text.find(decoded, char_idx);
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

            auto spans = doc_processor::SpanGenerator::Generate(tokens.size(), 12);
            if (spans.num_spans == 0) return std::vector<int>(tokens.size(), 0);

            const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};
            std::vector<int64_t> token_ids(tokens.begin(), tokens.end());
            std::vector<int64_t> attention_mask(tokens.size(), 1); // 1s for all tokens
            std::vector<int64_t> words_mask(tokens.size(), 1); // 1s for all tokens
            std::vector<int64_t> text_lengths = { static_cast<int64_t>(tokens.size()) };

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, token_ids.data(), token_ids.size(), input_shape.data(), input_shape.size()));
            
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()));

            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, words_mask.data(), words_mask.size(), input_shape.data(), input_shape.size()));

            std::vector<int64_t> len_shape = {1, 1};
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, text_lengths.data(), text_lengths.size(), len_shape.data(), len_shape.size()));

            std::vector<int64_t> span_idx_shape = {1, static_cast<int64_t>(spans.num_spans), 2};
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, spans.span_idx.data(), spans.span_idx.size(), span_idx_shape.data(), span_idx_shape.size()));

            std::vector<int64_t> span_mask_shape = {1, static_cast<int64_t>(spans.num_spans)};
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, spans.span_mask.data(), spans.span_mask.size(), span_mask_shape.data(), span_mask_shape.size()));

            const char* input_names[] = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"};
            const char* output_names[] = {"logits"};

            auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 6, output_names, 1);

            const auto* logits = output_tensors[0].GetTensorMutableData<float>();
            auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto out_shape = type_info.GetShape(); // Usually [1, num_spans, num_classes]
            size_t num_classes = (out_shape.size() == 3) ? out_shape[2] : 1;

            std::vector<int> tags(tokens.size(), 0);
            
            for (size_t i = 0; i < spans.num_spans; ++i) {
                // If it's multi-class, we check if any class > threshold (or specific class)
                // We'll check the first class or max class over 0.5f for simplicity
                float max_score = -1.0f;
                for (size_t c = 0; c < num_classes; ++c) {
                    float score = logits[i * num_classes + c];
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                if (max_score > 0.0f) { // Sigmoid > 0.5 means logit > 0.0
                    int64_t start = spans.span_idx[i * 2];
                    int64_t end = spans.span_idx[i * 2 + 1];
                    for (int64_t t = start; t <= end && t < static_cast<int64_t>(tags.size()); ++t) {
                        tags[t] = 1;
                    }
                }
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
