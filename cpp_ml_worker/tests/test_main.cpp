#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <stdckdint.h>
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
            return "12345";
        }
    };

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
                    logits[idx] = 10.0f; // High confidence logit for second word
                }
            }
            return logits;
        }
    };
}

TEST(WorkerTests, InitializationTest) {
    fz_context *ctx = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);
    ASSERT_NE(ctx, nullptr);
    fz_drop_context(ctx);
}

TEST(WorkerTests, InferenceEngineCalculatesCorrectBBoxes) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto onnx_runner = std::make_unique<MockOnnxRunner>();
    
    MLInferenceEngine engine(std::move(tokenizer), std::move(onnx_runner));

    const std::string text = "12345 12345 12345";
    std::vector<fz_rect> char_bboxes;
    for (int i = 0; i < 17; ++i) {
        fz_rect r = { static_cast<float>(i)*10, 0, static_cast<float>(i + 1)*10, 10 };
        char_bboxes.push_back(r);
    }

    std::vector<std::string> labels = {"TestLabel"};
    auto bboxes = engine.CalculateRedactionBBoxes(text, char_bboxes, labels);
    
    ASSERT_EQ(bboxes.size(), 1);
    EXPECT_FLOAT_EQ(bboxes[0].x0, 60.0f);
    EXPECT_FLOAT_EQ(bboxes[0].y0, 0.0f);
    EXPECT_FLOAT_EQ(bboxes[0].x1, 110.0f);
    EXPECT_FLOAT_EQ(bboxes[0].y1, 10.0f);
}

TEST(WorkerTests, CalculateExtractedSpans_StandardAndNormalized) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto onnx_runner = std::make_unique<MockOnnxRunner>();
    
    MLInferenceEngine engine(std::move(tokenizer), std::move(onnx_runner));

    const std::string text = "12345 12345 12345";
    std::vector<fz_rect> char_bboxes;
    for (int i = 0; i < 17; ++i) {
        fz_rect r = { static_cast<float>(i)*10, 0, static_cast<float>(i + 1)*10, 10 };
        char_bboxes.push_back(r);
    }

    std::vector<std::string> labels = {"TestLabel"};
    auto spans = engine.CalculateExtractedSpans(
        text,
        char_bboxes,
        labels,
        1,
        1000.0f,
        500.0f
    );

    ASSERT_EQ(spans.size(), 1);
    EXPECT_EQ(spans[0].text(), "12345");
    EXPECT_EQ(spans[0].label(), "TestLabel");
    EXPECT_EQ(spans[0].page_number(), 1);
    EXPECT_GT(spans[0].confidence(), 0.99f);
    EXPECT_FLOAT_EQ(spans[0].x0(), 60.0f / 1000.0f);
    EXPECT_FLOAT_EQ(spans[0].x1(), 110.0f / 1000.0f);
    EXPECT_FLOAT_EQ(spans[0].y0(), 0.0f);
    EXPECT_FLOAT_EQ(spans[0].y1(), 10.0f / 500.0f);
}

TEST(WorkerTests, CalculateExtractedSpans_FailFastOnInvalidInput) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto onnx_runner = std::make_unique<MockOnnxRunner>();
    
    MLInferenceEngine engine(std::move(tokenizer), std::move(onnx_runner));

    const std::string text = "12345 12345";
    std::vector<fz_rect> mismatched_bboxes(5, fz_empty_rect);
    std::vector<std::string> labels = {"TestLabel"};

    EXPECT_THROW(
        (void)engine.CalculateExtractedSpans(text, mismatched_bboxes, labels, 1, 1000.0f, 1000.0f),
        std::invalid_argument
    );

    std::vector<fz_rect> valid_bboxes(text.size(), fz_empty_rect);
    EXPECT_THROW(
        (void)engine.CalculateExtractedSpans(text, valid_bboxes, labels, 1, 0.0f, 1000.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        (void)engine.CalculateExtractedSpans(text, valid_bboxes, labels, 1, 1000.0f, -10.0f),
        std::invalid_argument
    );
}

TEST(WorkerTests, ResolveConfidenceThreshold_Hierarchy) {
    EXPECT_FLOAT_EQ(MLInferenceEngine::ResolveConfidenceThreshold(0.85f), 0.85f);
    EXPECT_FLOAT_EQ(MLInferenceEngine::ResolveConfidenceThreshold(-1.0f), 0.50f);

    setenv("GLINER_CONFIDENCE_THRESHOLD", "0.75", 1);
    EXPECT_FLOAT_EQ(MLInferenceEngine::ResolveConfidenceThreshold(-1.0f), 0.75f);
    unsetenv("GLINER_CONFIDENCE_THRESHOLD");
}

#include <fstream>
#include <sstream>

TEST(WorkerTests, MuPDFWrapper_ApplyRedactions_PermanentGlyphErasure) {
    const std::string text_pdf = 
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "5 0 obj << /Length 73 >> stream\n"
        "BT\n"
        "/F1 24 Tf\n"
        "50 720 Td\n"
        "(CONFIDENTIAL_KEY) Tj\n"
        "/F1 12 Tf\n"
        "0 -200 Td\n"
        "(Public body info) Tj\n"
        "ET\n"
        "endstream\n"
        "endobj\n"
        "xref\n"
        "0 6\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "0000000236 00000 n \n"
        "0000000307 00000 n \n"
        "trailer << /Size 6 /Root 1 0 R >>\n"
        "startxref\n"
        "432\n"
        "%%EOF\n";

    MuPDFWrapper mupdf(text_pdf);
    ASSERT_EQ(mupdf.GetPageCount(), 1);

    auto [initial_text, char_bboxes] = mupdf.ExtractPageText(0);
    size_t key_pos = initial_text.find("CONFIDENTIAL_KEY");
    ASSERT_NE(key_pos, std::string::npos);

    // Compute bounding box from actual extracted character positions
    fz_rect key_rect = fz_empty_rect;
    for (size_t i = 0; i < 16; ++i) {
        if (key_pos + i < char_bboxes.size() && !fz_is_empty_rect(char_bboxes[key_pos + i])) {
            if (fz_is_empty_rect(key_rect)) {
                key_rect = char_bboxes[key_pos + i];
            } else {
                key_rect = fz_union_rect(key_rect, char_bboxes[key_pos + i]);
            }
        }
    }
    ASSERT_FALSE(fz_is_empty_rect(key_rect));

    std::vector<fz_rect> redactions = {key_rect};
    mupdf.ApplyRedactions(0, redactions);

    std::string redacted_pdf_bytes = mupdf.GetRedactedPdf();
    EXPECT_GT(redacted_pdf_bytes.size(), 0);

    // Verify re-extraction from redacted PDF
    MuPDFWrapper redacted_mupdf(redacted_pdf_bytes);
    auto [after_text, after_bboxes] = redacted_mupdf.ExtractPageText(0);
    EXPECT_EQ(after_text.find("CONFIDENTIAL_KEY"), std::string::npos);
    EXPECT_NE(after_text.find("Public body info"), std::string::npos);
}

TEST(WorkerTests, MuPDFWrapper_ApplyNormalizedRedactions_MultiPageAndValidation) {
    const std::string multi_page_pdf = 
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "5 0 obj << /Length 50 >> stream\n"
        "BT\n/F1 12 Tf\n50 700 Td\n(PageOneSecret) Tj\nET\n"
        "endstream\nendobj\n"
        "6 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >> endobj\n"
        "7 0 obj << /Length 50 >> stream\n"
        "BT\n/F1 12 Tf\n50 700 Td\n(PageTwoSecret) Tj\nET\n"
        "endstream\nendobj\n"
        "xref\n"
        "0 8\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000121 00000 n \n"
        "0000000242 00000 n \n"
        "0000000313 00000 n \n"
        "0000000412 00000 n \n"
        "0000000533 00000 n \n"
        "trailer << /Size 8 /Root 1 0 R >>\n"
        "startxref\n"
        "632\n"
        "%%EOF\n";

    MuPDFWrapper mupdf(multi_page_pdf);
    ASSERT_EQ(mupdf.GetPageCount(), 2);

    ml::worker::v1::NormalizedRedactionBox box1;
    box1.set_page_number(1);
    box1.set_x0(0.0f); box1.set_y0(0.0f); box1.set_x1(1.0f); box1.set_y1(0.5f);

    ml::worker::v1::NormalizedRedactionBox box2;
    box2.set_page_number(2);
    box2.set_x0(0.0f); box2.set_y0(0.0f); box2.set_x1(1.0f); box2.set_y1(0.5f);

    // Degenerate box should be ignored without error
    ml::worker::v1::NormalizedRedactionBox degenerate_box;
    degenerate_box.set_page_number(1);
    degenerate_box.set_x0(0.5f); degenerate_box.set_y0(0.5f); degenerate_box.set_x1(0.5f); degenerate_box.set_y1(0.5f);

    std::vector<ml::worker::v1::NormalizedRedactionBox> custom_boxes = {box1, box2, degenerate_box};
    mupdf.ApplyNormalizedRedactions(custom_boxes);

    std::string redacted_bytes = mupdf.GetRedactedPdf();
    EXPECT_GT(redacted_bytes.size(), 0);

    // Out of bounds page check throws std::out_of_range
    EXPECT_THROW(mupdf.ApplyRedactions(-1, {}), std::out_of_range);
    EXPECT_THROW(mupdf.ApplyRedactions(10, {}), std::out_of_range);

    ml::worker::v1::NormalizedRedactionBox out_of_range_box;
    out_of_range_box.set_page_number(99);
    std::vector<ml::worker::v1::NormalizedRedactionBox> bad_boxes = {out_of_range_box};
    EXPECT_THROW(mupdf.ApplyNormalizedRedactions(bad_boxes), std::out_of_range);
}

TEST(WorkerTests, MuPDFWrapper_GetPageRotation_StandardAndRotated) {
    const std::string standard_pdf = 
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj\n"
        "xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        "trailer << /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n";

    MuPDFWrapper standard_mupdf(standard_pdf);
    EXPECT_EQ(standard_mupdf.GetPageRotation(0), 0);

    const std::string rotated_pdf = 
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 90 >> endobj\n"
        "xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        "trailer << /Size 4 /Root 1 0 R >>\nstartxref\n202\n%%EOF\n";

    MuPDFWrapper rotated_mupdf(rotated_pdf);
    EXPECT_EQ(rotated_mupdf.GetPageRotation(0), 90);
    EXPECT_THROW((void)rotated_mupdf.GetPageRotation(-1), std::out_of_range);
    EXPECT_THROW((void)rotated_mupdf.GetPageRotation(5), std::out_of_range);
}

TEST(WorkerTests, MuPDFWrapper_ApplyNormalizedRedactions_RotatedPageTextErasure) {
    const std::string rotated_text_pdf = 
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 90 /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "5 0 obj << /Length 50 >> stream\n"
        "BT\n/F1 12 Tf\n50 700 Td\n(RotatedSecret) Tj\nET\n"
        "endstream\nendobj\n"
        "xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000247 00000 n \n0000000318 00000 n \n"
        "trailer << /Size 6 /Root 1 0 R >>\nstartxref\n443\n%%EOF\n";

    MuPDFWrapper mupdf(rotated_text_pdf);
    EXPECT_EQ(mupdf.GetPageRotation(0), 90);

    auto [initial_text, char_bboxes] = mupdf.ExtractPageText(0);
    EXPECT_NE(initial_text.find("RotatedSecret"), std::string::npos);

    // Apply normalized redaction box over full visual page
    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0(0.0f);
    box.set_y0(0.0f);
    box.set_x1(1.0f);
    box.set_y1(1.0f);

    std::vector<ml::worker::v1::NormalizedRedactionBox> boxes = {box};
    mupdf.ApplyNormalizedRedactions(boxes);

    std::string redacted_pdf_bytes = mupdf.GetRedactedPdf();
    EXPECT_GT(redacted_pdf_bytes.size(), 0);

    MuPDFWrapper redacted_mupdf(redacted_pdf_bytes);
    auto [after_text, after_bboxes] = redacted_mupdf.ExtractPageText(0);
    EXPECT_EQ(after_text.find("RotatedSecret"), std::string::npos);
}

namespace {
    class MultiSubtokenMockTokenizer : public ITokenizer {
    public:
        std::vector<int32_t> Encode(const std::string& text) override {
            if (text == "first") return {10};
            if (text == "multi") return {20, 21, 22};
            if (text == "third") return {30};
            return {100};
        }
        std::string Decode(const std::vector<int32_t>& ids) override { return ""; }
    };

    class RecordingMockOnnxRunner : public IOnnxRunner {
    public:
        std::vector<std::vector<int64_t>> captured_input_ids;
        std::vector<std::vector<int64_t>> captured_words_masks;
        std::vector<size_t> captured_num_words;

        std::vector<float> RunInference(
            const std::vector<int64_t>& input_ids,
            const std::vector<int64_t>& attention_mask,
            const std::vector<int64_t>& words_mask,
            size_t num_words,
            size_t num_classes,
            const doc_processor::GeneratedSpans& spans
        ) override {
            captured_input_ids.push_back(input_ids);
            captured_words_masks.push_back(words_mask);
            captured_num_words.push_back(num_words);
            return std::vector<float>(num_words * 12 * num_classes, -10.0f);
        }
    };

    class SingleSpanOnnxRunner : public IOnnxRunner {
    public:
        size_t target_word_start = 0;
        size_t target_span_len = 1;
        float target_confidence = 10.0f;

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
            if (target_word_start < num_words && target_span_len <= 12 && num_classes > 0) {
                size_t l = target_span_len - 1;
                size_t idx = (target_word_start * 12 + l) * num_classes + 0;
                if (idx < logits.size()) {
                    logits[idx] = target_confidence;
                }
            }
            return logits;
        }
    };
}

TEST(WorkerTests, WordsMask_1BasedSequentialAndSubtokenMasking) {
    auto tokenizer = std::make_unique<MultiSubtokenMockTokenizer>();
    auto runner_ptr = std::make_unique<RecordingMockOnnxRunner>();
    auto* runner = runner_ptr.get();

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    const std::string text = "first multi third";
    std::vector<fz_rect> char_bboxes(text.size(), fz_rect{0, 0, 10, 10});
    std::vector<std::string> labels = {"TestLabel"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    ASSERT_EQ(runner->captured_words_masks.size(), 1);
    const auto& mask = runner->captured_words_masks[0];
    const auto& ids = runner->captured_input_ids[0];

    ASSERT_EQ(mask.size(), ids.size());
    ASSERT_GE(mask.size(), 4 + 1 + 3 + 1 + 1);
    EXPECT_EQ(mask[0], 0);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 0);
    EXPECT_EQ(mask[3], 0);
    EXPECT_EQ(mask[4], 1);
    EXPECT_EQ(mask[5], 2);
    EXPECT_EQ(mask[6], 0);
    EXPECT_EQ(mask[7], 0);
    EXPECT_EQ(mask[8], 3);
    EXPECT_EQ(mask[9], 0);
}

TEST(WorkerTests, WordsMask_SlidingWindowRelativeReset) {
    auto tokenizer = std::make_unique<MultiSubtokenMockTokenizer>();
    auto runner_ptr = std::make_unique<RecordingMockOnnxRunner>();
    auto* runner = runner_ptr.get();

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    std::string text;
    for (int i = 0; i < 400; ++i) {
        text += "word ";
    }
    std::vector<fz_rect> char_bboxes(text.size(), fz_rect{0, 0, 10, 10});
    std::vector<std::string> labels = {"TestLabel"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    ASSERT_EQ(runner->captured_words_masks.size(), 2);
    const auto& mask1 = runner->captured_words_masks[0];
    EXPECT_EQ(runner->captured_num_words[0], 384);
    EXPECT_EQ(mask1[4], 1);
    EXPECT_EQ(mask1[4 + 383], 384);

    const auto& mask2 = runner->captured_words_masks[1];
    EXPECT_EQ(runner->captured_num_words[1], 80);
    EXPECT_EQ(mask2[4], 1);
    EXPECT_EQ(mask2[4 + 79], 80);
}

TEST(WorkerTests, PunctuationAwareTokenization_PunctuationAndSymbols) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<RecordingMockOnnxRunner>();
    auto* runner = runner_ptr.get();

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    const std::string text = "$125,000 and 9.775% (pre-payment)";
    std::vector<fz_rect> char_bboxes;
    for (size_t i = 0; i < text.size(); ++i) {
        char_bboxes.push_back(fz_rect{static_cast<float>(i * 10), 0.0f, static_cast<float>((i + 1) * 10), 20.0f});
    }
    std::vector<std::string> labels = {"Amount"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 1000.0f, 500.0f);

    ASSERT_EQ(runner->captured_num_words.size(), 1);
    EXPECT_EQ(runner->captured_num_words[0], 12);
}

TEST(WorkerTests, PunctuationAwareTokenization_ExactGlyphBoundingBoxes) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<SingleSpanOnnxRunner>();
    runner_ptr->target_word_start = 0;
    runner_ptr->target_span_len = 1;

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    const std::string text = "$125,000";
    std::vector<fz_rect> char_bboxes;
    for (size_t i = 0; i < text.size(); ++i) {
        char_bboxes.push_back(fz_rect{static_cast<float>(i * 10), 5.0f, static_cast<float>((i + 1) * 10), 25.0f});
    }
    std::vector<std::string> labels = {"CurrencySymbol"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    ASSERT_EQ(spans.size(), 1);
    EXPECT_EQ(spans[0].text(), "$");
    EXPECT_FLOAT_EQ(spans[0].x0(), 0.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].x1(), 10.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].y0(), 5.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].y1(), 25.0f / 100.0f);
}

TEST(WorkerTests, PunctuationAwareTokenization_MultiTokenSpanBoundingUnion) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<SingleSpanOnnxRunner>();
    runner_ptr->target_word_start = 0;
    runner_ptr->target_span_len = 4;

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    const std::string text = "$125,000";
    std::vector<fz_rect> char_bboxes;
    for (size_t i = 0; i < text.size(); ++i) {
        char_bboxes.push_back(fz_rect{static_cast<float>(i * 10), 10.0f, static_cast<float>((i + 1) * 10), 30.0f});
    }
    std::vector<std::string> labels = {"LoanAmount"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    ASSERT_EQ(spans.size(), 1);
    EXPECT_EQ(spans[0].text(), "$125,000");
    EXPECT_FLOAT_EQ(spans[0].x0(), 0.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].x1(), 80.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].y0(), 10.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].y1(), 30.0f / 100.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
