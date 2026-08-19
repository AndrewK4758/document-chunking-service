#include "MLInferenceEngine.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
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
}

std::vector<fz_rect> MLInferenceEngine::CalculateRedactionBBoxes(const std::string& text, const std::vector<fz_rect>& char_bboxes, std::span<const std::string> dynamic_labels) {
    if (text.size() != char_bboxes.size()) {
        throw std::runtime_error("Text length does not match character bounding boxes length");
    }
    if (text.empty() || dynamic_labels.empty()) {
        return {};
    }

    // 1. Extract words and corresponding bounding boxes
    std::vector<WordSpan> words;
    size_t n = text.size();
    size_t char_pos = 0;

    while (char_pos < n) {
        while (char_pos < n && (std::isspace(static_cast<unsigned char>(text[char_pos])) || text[char_pos] == '\n' || text[char_pos] == '\r')) {
            char_pos++;
        }
        if (char_pos >= n) break;

        size_t start = char_pos;
        while (char_pos < n && !std::isspace(static_cast<unsigned char>(text[char_pos])) && text[char_pos] != '\n' && text[char_pos] != '\r') {
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

    // 2. Build GLiNER prompt sequence: [CLS] <<ENT>> label1 <<ENT>> label2 ... <<SEP>>
    std::vector<int64_t> input_ids;
    input_ids.push_back(1); // [CLS]

    for (const auto& label : dynamic_labels) {
        input_ids.push_back(128002); // <<ENT>>
        std::vector<int32_t> l_tokens = tokenizer_->Encode(SanitizeUtf8(label));
        for (int32_t id : l_tokens) {
            input_ids.push_back(static_cast<int64_t>(id));
        }
    }
    input_ids.push_back(128003); // <<SEP>>

    size_t prompt_len = input_ids.size();
    std::vector<int64_t> words_mask(prompt_len, 0);

    // 3. Tokenize words and map into words_mask
    for (size_t w_idx = 0; w_idx < words.size(); ++w_idx) {
        std::string safe_word = SanitizeUtf8(words[w_idx].text);
        std::vector<int32_t> w_subwords = tokenizer_->Encode(safe_word);
        if (w_subwords.empty()) {
            w_subwords.push_back(0);
        }
        for (int32_t id : w_subwords) {
            input_ids.push_back(static_cast<int64_t>(id));
            words_mask.push_back(static_cast<int64_t>(w_idx + 1));
        }
    }

    input_ids.push_back(2); // [SEP]
    words_mask.push_back(0);

    std::vector<int64_t> attention_mask(input_ids.size(), 1);

    // 4. Generate spans
    size_t num_words = words.size();
    size_t max_span_len = 12;
    auto spans = doc_processor::SpanGenerator::Generate(num_words, max_span_len);

    // 5. Run ONNX Inference
    std::vector<float> logits = onnx_runner_->RunInference(
        input_ids, attention_mask, words_mask, num_words, dynamic_labels.size(), spans
    );

    if (logits.empty()) {
        return {};
    }

    // 6. Decode predictions using mdspan
    auto logits_md = std::mdspan<const float, std::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>>(
        logits.data(), num_words, max_span_len, dynamic_labels.size()
    );

    std::vector<fz_rect> redaction_bboxes;
    const float threshold = 0.4f;

    for (size_t w = 0; w < num_words; ++w) {
        for (size_t span_len = 0; span_len < max_span_len; ++span_len) {
            size_t end_word = w + span_len;
            if (end_word < num_words) {
                for (size_t c = 0; c < dynamic_labels.size(); ++c) {
                    float logit = logits_md[w, span_len, c];
                    float prob = 1.0f / (1.0f + std::exp(-logit));
                    if (prob > threshold) {
                        fz_rect entity_box = fz_empty_rect;
                        for (size_t k = w; k <= end_word; ++k) {
                            if (!fz_is_empty_rect(words[k].bbox)) {
                                if (fz_is_empty_rect(entity_box)) {
                                    entity_box = words[k].bbox;
                                } else {
                                    entity_box = fz_union_rect(entity_box, words[k].bbox);
                                }
                            }
                        }
                        if (!fz_is_empty_rect(entity_box)) {
                            redaction_bboxes.push_back(entity_box);
                        }
                    }
                }
            }
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
