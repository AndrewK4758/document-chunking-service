#pragma once

#include <string>
#include <vector>
#include <memory>

#include <stdckdint.h>

#include "mupdf/fitz.h"

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
    virtual std::vector<int> RunInference(const std::vector<int32_t>& tokens) = 0;
};

// Abstract interface for the engine
class IMLInferenceEngine {
public:
    virtual ~IMLInferenceEngine() = default;
    virtual std::vector<fz_rect> CalculateRedactionBBoxes(const std::string& text, const std::vector<fz_rect>& char_bboxes) = 0;
};

class MLInferenceEngine : public IMLInferenceEngine {
public:
    MLInferenceEngine(std::unique_ptr<ITokenizer> tokenizer, std::unique_ptr<IOnnxRunner> onnx_runner);

    std::vector<fz_rect> CalculateRedactionBBoxes(const std::string& text, const std::vector<fz_rect>& char_bboxes) override;

private:
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<IOnnxRunner> onnx_runner_;
};

// Factory functions that can fail-fast if files are missing
std::unique_ptr<ITokenizer> CreateRealTokenizer(const std::string& tokenizer_json_path);
std::unique_ptr<IOnnxRunner> CreateRealOnnxRunner(const std::string& model_path);
