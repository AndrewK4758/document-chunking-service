#pragma once

#include <string>
#include <vector>
#include <memory>
#include <span>
#include <cstdint>
#include <stdexcept>

#include <stdckdint.h>

#include "mupdf/fitz.h"
#include "SpanGenerator.h"
#include "ml_worker.pb.h"

// Interface for Tokenizer to allow mocking in tests
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int32_t> Encode(const std::string& text) = 0;
    virtual std::string Decode(const std::vector<int32_t>& ids) = 0;
};

// Interface for ONNX inference to allow mocking in tests
class IOnnxRunner {
public:
    virtual ~IOnnxRunner() = default;
    virtual std::vector<float> RunInference(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        const std::vector<int64_t>& words_mask,
        size_t num_words,
        size_t num_classes,
        const doc_processor::GeneratedSpans& spans
    ) = 0;
};

// Abstract interface for the engine
class IMLInferenceEngine {
public:
    virtual ~IMLInferenceEngine() = default;
    virtual std::vector<fz_rect> CalculateRedactionBBoxes(
        const std::string& text,
        const std::vector<fz_rect>& char_bboxes,
        std::span<const std::string> dynamic_labels
    ) = 0;

    virtual std::vector<ml::worker::v1::ExtractedSpan> CalculateExtractedSpans(
        const std::string& text,
        const std::vector<fz_rect>& char_bboxes,
        std::span<const std::string> dynamic_labels,
        int page_number,
        float page_width,
        float page_height,
        float threshold_override = -1.0f
    ) = 0;
};

class MLInferenceEngine : public IMLInferenceEngine {
public:
    MLInferenceEngine(std::unique_ptr<ITokenizer> tokenizer, std::unique_ptr<IOnnxRunner> onnx_runner);

    std::vector<fz_rect> CalculateRedactionBBoxes(
        const std::string& text,
        const std::vector<fz_rect>& char_bboxes,
        std::span<const std::string> dynamic_labels
    ) override;

    std::vector<ml::worker::v1::ExtractedSpan> CalculateExtractedSpans(
        const std::string& text,
        const std::vector<fz_rect>& char_bboxes,
        std::span<const std::string> dynamic_labels,
        int page_number,
        float page_width,
        float page_height,
        float threshold_override = -1.0f
    ) override;

    [[nodiscard]] static float ResolveConfidenceThreshold(float threshold_override);

private:
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<IOnnxRunner> onnx_runner_;
};

// Factory functions that can fail-fast if files are missing
std::unique_ptr<ITokenizer> CreateRealTokenizer(const std::string& tokenizer_json_path);
std::unique_ptr<IOnnxRunner> CreateRealOnnxRunner(const std::string& model_path);
