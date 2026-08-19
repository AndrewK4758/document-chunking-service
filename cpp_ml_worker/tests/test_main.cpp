#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include "MLInferenceEngine.h"
#include "MuPDFWrapper.h"

extern "C" {
#include "mupdf/fitz.h"
}

namespace {
    class MockTokenizer : public ITokenizer {
    public:
        std::vector<int32_t> Encode(const std::string& text) override {
            std::vector<int32_t> tokens;
            int pos = 0;
            // Simple tokenization: every 5 characters is a token
            while (pos < text.size()) {
                tokens.push_back(pos); // Use position as token ID
                pos += 5;
            }
            return tokens;
        }

        std::string Decode(const std::vector<int32_t>& ids) override {
            // We know each token spans 5 characters, just return 5 spaces
            // OR return a known substring for mapping.
            // For our test, let's just make it return a 5 character long string
            return "12345";
        }
    };
}

#include "SpanGenerator.h"

namespace {
    class MockOnnxRunner : public IOnnxRunner {
    public:
        std::vector<float> RunInference(
            const std::vector<int64_t>& input_ids,
            const std::vector<int64_t>& attention_mask,
            const std::vector<int64_t>& words_mask,
            size_t num_words,
            size_t num_classes,
            const doc_processor::GeneratedSpans& spans
        ) override {
            size_t total = num_words * 12 * num_classes;
            std::vector<float> logits(total, -10.0f);
            if (num_words > 1 && num_classes > 0) {
                size_t idx = (1 * 12 + 0) * num_classes + 0;
                if (idx < logits.size()) {
                    logits[idx] = 10.0f;
                }
            }
            return logits;
        }
    };
}

TEST(WorkerTests, InitializationTest) {
    fz_context *ctx = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);    ASSERT_NE(ctx, nullptr);
    fz_drop_context(ctx);
}

TEST(WorkerTests, InferenceEngineCalculatesCorrectBBoxes) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto onnx_runner = std::make_unique<MockOnnxRunner>();
    
    MLInferenceEngine engine(std::move(tokenizer), std::move(onnx_runner));

    const std::string text = "12345 12345 12345";
    std::vector<fz_rect> char_bboxes;
    for (int i = 0; i < 17; ++i) {
        fz_rect r = { static_cast<float>(i)*10, 0, static_cast<float>(i + 1)*10, 10 };        char_bboxes.push_back(r);
    }

    std::vector<std::string> labels = {"TestLabel"};
    auto bboxes = engine.CalculateRedactionBBoxes(text, char_bboxes, labels);
    
    // The second word (indices 6-10) was tagged as PII.
    // Bbox should cover x from 60 to 110.
    ASSERT_EQ(bboxes.size(), 1);
    EXPECT_FLOAT_EQ(bboxes[0].x0, 60.0f);
    EXPECT_FLOAT_EQ(bboxes[0].y0, 0.0f);
    EXPECT_FLOAT_EQ(bboxes[0].x1, 110.0f);
    EXPECT_FLOAT_EQ(bboxes[0].y1, 10.0f);
}

#include <fstream>
#include <sstream>

TEST(WorkerTests, MuPDFWrapperExtractsText) {
    try {
        std::ifstream file("tests/dummy.pdf", std::ios::binary);
        if (!file.is_open()) {
            file.open("../tests/dummy.pdf", std::ios::binary);
        }
        if (!file.is_open()) {
            file.open("../../tests/dummy.pdf", std::ios::binary);
        }
        ASSERT_TRUE(file.is_open()) << "dummy.pdf not found in test dir";
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string pdf_data = buffer.str();

        MuPDFWrapper mupdf(pdf_data);
        ASSERT_EQ(mupdf.GetPageCount(), 1);

        auto [text, char_bboxes] = mupdf.ExtractPageText(0);
        EXPECT_GT(text.size(), 0);
        EXPECT_GT(char_bboxes.size(), 0);

        // Test apply redaction
        std::vector<fz_rect> redactions;
        if (!char_bboxes.empty()) {
            redactions.push_back(char_bboxes[0]);
        }
        mupdf.ApplyRedactions(0, redactions);
        
        std::string redacted = mupdf.GetRedactedPdf();
        EXPECT_GT(redacted.size(), 0);
    } catch (const std::exception& e) {
        FAIL() << "Uncaught exception: " << e.what();
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
