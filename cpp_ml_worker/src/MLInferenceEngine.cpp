#include "MLInferenceEngine.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <tuple>
#include <onnxruntime_cxx_api.h>
#include <ranges>
#include "tokenizers_cpp/tokenizers_cpp.h"
#include "SpanGenerator.h"

MLInferenceEngine::MLInferenceEngine(std::unique_ptr<ITokenizer> tokenizer, std::unique_ptr<IOnnxRunner> onnx_runner)
    : tokenizer_(std::move(tokenizer)), onnx_runner_(std::move(onnx_runner)) {
    if (!tokenizer_ || !onnx_runner_) {
        throw std::runtime_error("MLInferenceEngine initialized with null components");
    }
}

namespace {
    struct WordSpan {
        std::string text;
        size_t start_char;
        size_t end_char;
        fz_rect bbox;
    };

    std::string SanitizeUtf8(const std::string& input) {
        std::string out;
        out.reserve(input.size());
        for (size_t i = 0; i < input.size(); ) {
            if (const auto c = static_cast<unsigned char>(input[i]); c <= 0x7F) {
                out += input[i++];
            } else if (c >= 0xC2 && c <= 0xDF) {
                if (i + 1 < input.size() && (static_cast<unsigned char>(input[i+1]) & 0xC0) == 0x80) {
                    out += input[i++]; out += input[i++];
                } else { out += ' '; i++; }
            } else if (c >= 0xE0 && c <= 0xEF) {
                if (i + 2 < input.size() && (static_cast<unsigned char>(input[i+1]) & 0xC0) == 0x80 && (static_cast<unsigned char>(input[i+2]) & 0xC0) == 0x80) {
                    out += input[i++]; out += input[i++]; out += input[i++];
                } else { out += ' '; i++; }
            } else if (c >= 0xF0 && c <= 0xF4) {
                if (i + 3 < input.size() && (static_cast<unsigned char>(input[i+1]) & 0xC0) == 0x80 && (static_cast<unsigned char>(input[i+2]) & 0xC0) == 0x80 && (static_cast<unsigned char>(input[i+3]) & 0xC0) == 0x80) {
                    out += input[i++]; out += input[i++]; out += input[i++]; out += input[i++];
                } else { out += ' '; i++; }
            } else {
                out += ' '; i++;
            }
        }
        return out;
    }

    struct CandidateKey {
        size_t start_word;
        size_t end_word;
        std::string label;

        bool operator<(const CandidateKey& other) const {
            return std::tie(start_word, end_word, label) < std::tie(other.start_word, other.end_word, other.label);
        }
    };
}

float MLInferenceEngine::ResolveConfidenceThreshold(float threshold_override) {
    if (threshold_override >= 0.0f && threshold_override <= 1.0f) {
        return threshold_override;
    }
    if (const char* env_val = std::getenv("GLINER_CONFIDENCE_THRESHOLD")) {
        try {
            float val = std::stof(env_val);
            if (val >= 0.0f && val <= 1.0f) {
                return val;
            }
        } catch (...) {}
    }
    return 0.50f;
}

std::vector<ml::worker::v1::ExtractedSpan> MLInferenceEngine::CalculateExtractedSpans(
    const std::string& text,
    const std::vector<fz_rect>& char_bboxes,
    std::span<const std::string> dynamic_labels,
    int page_number,
    float page_width,
    float page_height,
    float threshold_override
) {
    if (text.size() != char_bboxes.size()) {
        throw std::invalid_argument("Text length does not match character bounding boxes length");
    }
    if (page_width <= 0.0f || page_height <= 0.0f) {
        throw std::invalid_argument("Page dimensions must be positive");
    }
    if (text.empty() || dynamic_labels.empty()) {
        return {};
    }

    float threshold = ResolveConfidenceThreshold(threshold_override);

    // 1. Extract words and corresponding bounding boxes using punctuation-aware tokenization (\w+(?:[-_]\w+)*|\S)
    std::vector<WordSpan> words;
    size_t n = text.size();
    size_t char_pos = 0;

    auto is_word_char = [](char c) -> bool {
        unsigned char uc = static_cast<unsigned char>(c);
        return std::isalnum(uc) || uc == '_' || uc >= 0x80;
    };

    while (char_pos < n) {
        while (char_pos < n && (std::isspace(static_cast<unsigned char>(text[char_pos])) || text[char_pos] == '\n' || text[char_pos] == '\r')) {
            char_pos++;
        }
        if (char_pos >= n) break;

        size_t start = char_pos;
        if (is_word_char(text[char_pos])) {
            while (char_pos < n) {
                if (is_word_char(text[char_pos])) {
                    char_pos++;
                } else if ((text[char_pos] == '-' || text[char_pos] == '_') &&
                           char_pos + 1 < n && is_word_char(text[char_pos + 1])) {
                    char_pos += 2;
                } else {
                    break;
                }
            }
        } else {
            char_pos++;
        }
        size_t end = char_pos;

        fz_rect word_box = fz_empty_rect;
        for (size_t c = start; c < end; ++c) {
            if (c < char_bboxes.size() && !fz_is_empty_rect(char_bboxes[c])) {
                if (fz_is_empty_rect(word_box)) {
                    word_box = char_bboxes[c];
                } else {
                    word_box = fz_union_rect(word_box, char_bboxes[c]);
                }
            }
        }

        words.push_back({text.substr(start, end - start), start, end, word_box});
    }

    if (words.empty()) {
        return {};
    }

    constexpr size_t WINDOW_SIZE = 384;
    constexpr size_t WINDOW_OVERLAP = 64;
    constexpr size_t WINDOW_STEP = WINDOW_SIZE - WINDOW_OVERLAP;

    std::map<CandidateKey, float> candidate_map;

    size_t num_total_words = words.size();
    for (size_t window_start = 0; window_start < num_total_words; ) {
        size_t window_end = std::min(window_start + WINDOW_SIZE, num_total_words);
        size_t window_word_count = window_end - window_start;

        // Build GLiNER prompt sequence: [CLS] <<ENT>> label1 <<ENT>> label2 ... <<SEP>> word1 word2 ... [SEP]
        std::vector<int64_t> input_ids;
        input_ids.push_back(1); // [CLS]

        for (const auto& label : dynamic_labels) {
            input_ids.push_back(128002); // <<ENT>>
            std::vector<int32_t> l_tokens = tokenizer_->Encode(SanitizeUtf8(label));
            for (int32_t t : l_tokens) {
                input_ids.push_back(t);
            }
        }
        input_ids.push_back(128003); // <<SEP>>

        std::vector<int64_t> words_mask(input_ids.size(), 0);

        for (size_t wi = window_start; wi < window_end; ++wi) {
            const auto& w = words[wi];
            std::vector<int32_t> w_tokens = tokenizer_->Encode(SanitizeUtf8(w.text));
            int64_t word_idx = static_cast<int64_t>(wi - window_start + 1);

            if (w_tokens.empty()) {
                input_ids.push_back(0);
                words_mask.push_back(word_idx);
            } else {
                input_ids.push_back(w_tokens[0]);
                words_mask.push_back(word_idx);
                for (size_t ti = 1; ti < w_tokens.size(); ++ti) {
                    input_ids.push_back(w_tokens[ti]);
                    words_mask.push_back(0);
                }
            }
        }

        input_ids.push_back(2); // [SEP]
        words_mask.push_back(0);

        std::vector<int64_t> attention_mask(input_ids.size(), 1);

        auto spans = doc_processor::SpanGenerator::Generate(window_word_count, 12);

        std::vector<float> logits = onnx_runner_->RunInference(
            input_ids,
            attention_mask,
            words_mask,
            window_word_count,
            dynamic_labels.size(),
            spans
        );

        if (!logits.empty()) {
            size_t num_classes = dynamic_labels.size();
            for (size_t k = 0; k < window_word_count; ++k) {
                for (size_t l = 0; l < 12; ++l) {
                    if (k + l >= window_word_count) continue;
                    for (size_t c = 0; c < num_classes; ++c) {
                        size_t idx = (k * 12 + l) * num_classes + c;
                        if (idx < logits.size()) {
                            float logit = logits[idx];
                            float conf = 1.0f / (1.0f + std::exp(-logit));
                            if (conf >= threshold) {
                                CandidateKey key{window_start + k, window_start + k + l, dynamic_labels[c]};
                                auto it = candidate_map.find(key);
                                if (it == candidate_map.end() || conf > it->second) {
                                    candidate_map[key] = conf;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (window_end >= num_total_words) {
            break;
        }
        window_start += (num_total_words > WINDOW_SIZE ? WINDOW_STEP : num_total_words);
    }

    std::vector<ml::worker::v1::ExtractedSpan> results;
    results.reserve(candidate_map.size());

    for (const auto& [cand_key, conf] : candidate_map) {
        fz_rect union_box = fz_empty_rect;
        for (size_t wi = cand_key.start_word; wi <= cand_key.end_word; ++wi) {
            if (!fz_is_empty_rect(words[wi].bbox)) {
                if (fz_is_empty_rect(union_box)) {
                    union_box = words[wi].bbox;
                } else {
                    union_box = fz_union_rect(union_box, words[wi].bbox);
                }
            }
        }

        float rx0 = std::min(union_box.x0, union_box.x1);
        float rx1 = std::max(union_box.x0, union_box.x1);
        float ry0 = std::min(union_box.y0, union_box.y1);
        float ry1 = std::max(union_box.y0, union_box.y1);

        float norm_x0 = std::clamp(rx0 / page_width, 0.0f, 1.0f);
        float norm_x1 = std::clamp(rx1 / page_width, 0.0f, 1.0f);
        float norm_y0 = std::clamp(ry0 / page_height, 0.0f, 1.0f);
        float norm_y1 = std::clamp(ry1 / page_height, 0.0f, 1.0f);

        size_t text_start = words[cand_key.start_word].start_char;
        size_t text_end = words[cand_key.end_word].end_char;
        std::string span_text = text.substr(text_start, text_end - text_start);

        ml::worker::v1::ExtractedSpan span;
        span.set_text(span_text);
        span.set_x0(norm_x0);
        span.set_y0(norm_y0);
        span.set_x1(norm_x1);
        span.set_y1(norm_y1);
        span.set_label(cand_key.label);
        span.set_confidence(conf);
        span.set_page_number(page_number);

        results.push_back(std::move(span));
    }

    return results;
}

std::vector<fz_rect> MLInferenceEngine::CalculateRedactionBBoxes(
    const std::string& text,
    const std::vector<fz_rect>& char_bboxes,
    std::span<const std::string> dynamic_labels
) {
    if (text.size() != char_bboxes.size()) {
        throw std::invalid_argument("Text length does not match character bounding boxes length");
    }
    if (text.empty() || dynamic_labels.empty()) {
        return {};
    }

    auto extracted_spans = CalculateExtractedSpans(
        text,
        char_bboxes,
        dynamic_labels,
        1,
        1000.0f,
        1000.0f
    );

    std::vector<fz_rect> redaction_bboxes;
    redaction_bboxes.reserve(extracted_spans.size());

    for (const auto& span : extracted_spans) {
        fz_rect rect;
        rect.x0 = span.x0() * 1000.0f;
        rect.y0 = span.y0() * 1000.0f;
        rect.x1 = span.x1() * 1000.0f;
        rect.y1 = span.y1() * 1000.0f;
        redaction_bboxes.push_back(rect);
    }

    return redaction_bboxes;
}

namespace {
    class RealTokenizer : public ITokenizer {
    public:
        explicit RealTokenizer(const std::string& path) {
            std::ifstream file(path);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open tokenizer json file: " + path);
            }
            std::stringstream buffer;
            buffer << file.rdbuf();
            tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(buffer.str());
            if (!tokenizer_) {
                throw std::runtime_error("Failed to create tokenizer from blob json");
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

#include <spdlog/spdlog.h>

namespace {
    class RealOnnxRunner : public IOnnxRunner {
    public:
        explicit RealOnnxRunner(const std::string& path) : env_(ORT_LOGGING_LEVEL_WARNING, "MLWorkerEnv") {
            bool session_created = false;
            try {
                Ort::SessionOptions cuda_session_options;
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                cuda_session_options.AppendExecutionProvider_CUDA(cuda_options);
                cuda_session_options.SetIntraOpNumThreads(4);
                cuda_session_options.SetInterOpNumThreads(2);
                session_ = std::make_unique<Ort::Session>(env_, path.c_str(), cuda_session_options);
                session_created = true;
                spdlog::info("Successfully initialized ONNX Runtime with CUDA Execution Provider (GPU device 0).");
            } catch (const std::exception& e) {
                spdlog::warn("CUDA Execution Provider initialization unavailable: {}. Falling back to CPU Execution Provider.", e.what());
            }

            if (!session_created) {
                try {
                    Ort::SessionOptions cpu_session_options;
                    cpu_session_options.SetIntraOpNumThreads(4);
                    cpu_session_options.SetInterOpNumThreads(2);
                    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), cpu_session_options);
                    spdlog::info("Successfully initialized ONNX Runtime with CPU Execution Provider.");
                } catch (const Ort::Exception& e) {
                    throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
                }
            }
        }

        std::vector<float> RunInference(
            const std::vector<int64_t>& input_ids,
            const std::vector<int64_t>& attention_mask,
            const std::vector<int64_t>& words_mask,
            size_t num_words,
            size_t num_classes,
            const doc_processor::GeneratedSpans& spans
        ) override {
            if (input_ids.empty() || num_words == 0 || num_classes == 0) return {};

            Ort::AllocatorWithDefaultOptions allocator;
            
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
            auto token_tensor = Ort::Value::CreateTensor<int64_t>(allocator, input_shape.data(), input_shape.size());
            std::copy(input_ids.begin(), input_ids.end(), token_tensor.GetTensorMutableData<int64_t>());
            
            auto attn_tensor = Ort::Value::CreateTensor<int64_t>(allocator, input_shape.data(), input_shape.size());
            std::copy(attention_mask.begin(), attention_mask.end(), attn_tensor.GetTensorMutableData<int64_t>());
            
            auto words_tensor = Ort::Value::CreateTensor<int64_t>(allocator, input_shape.data(), input_shape.size());
            std::copy(words_mask.begin(), words_mask.end(), words_tensor.GetTensorMutableData<int64_t>());

            std::vector<int64_t> len_shape = {1, 1};
            auto len_tensor = Ort::Value::CreateTensor<int64_t>(allocator, len_shape.data(), len_shape.size());
            len_tensor.GetTensorMutableData<int64_t>()[0] = static_cast<int64_t>(num_words);

            std::vector<int64_t> span_idx_shape = {1, static_cast<int64_t>(spans.num_spans), 2};
            auto span_idx_tensor = Ort::Value::CreateTensor<int64_t>(allocator, span_idx_shape.data(), span_idx_shape.size());
            std::copy(spans.span_idx_storage.begin(), spans.span_idx_storage.end(), span_idx_tensor.GetTensorMutableData<int64_t>());

            std::vector<int64_t> span_mask_shape = {1, static_cast<int64_t>(spans.num_spans)};
            auto span_mask_tensor = Ort::Value::CreateTensor<bool>(allocator, span_mask_shape.data(), span_mask_shape.size());
            bool* mask_ptr = span_mask_tensor.GetTensorMutableData<bool>();
            for (size_t i = 0; i < spans.num_spans; ++i) {
                mask_ptr[i] = spans.span_mask_storage[i] != 0;
            }

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(token_tensor));
            input_tensors.push_back(std::move(attn_tensor));
            input_tensors.push_back(std::move(words_tensor));
            input_tensors.push_back(std::move(len_tensor));
            input_tensors.push_back(std::move(span_idx_tensor));
            input_tensors.push_back(std::move(span_mask_tensor));

            const char* input_names[] = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"};
            const char* output_names[] = {"logits"};

            auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 6, output_names, 1);

            const auto* logits_raw = output_tensors[0].GetTensorMutableData<float>();
            auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t total_elements = type_info.GetElementCount();

            return std::vector<float>(logits_raw, logits_raw + total_elements);
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
