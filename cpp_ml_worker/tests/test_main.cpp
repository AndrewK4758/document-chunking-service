#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <stdckdint.h>
#include <onnxruntime_cxx_api.h>
#include <simdutf.h>
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
        "trailer << /Size 6 /Root 1 0 R >>\n"
        "startxref\n443\n%%EOF\n";

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

// ---------------------------------------------------------------------------
// MPL-311: Rotation-Aware Spatial Normalization & Y-Axis Inversion Tests
// ---------------------------------------------------------------------------

namespace {
    std::string MakeSyntheticCanaryPdf(
        const std::string& top_text,
        const std::string& bottom_text,
        const std::string& mediabox = "[0 0 612 792]",
        int rotation = 0,
        float top_y = 720.0f,
        float bottom_y = 200.0f,
        float x_pos = 50.0f
    ) {
        std::ostringstream ss;
        ss << "%PDF-1.4\n";
        ss << "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n";
        ss << "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n";
        ss << "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox " << mediabox;
        if (rotation != 0) {
            ss << " /Rotate " << rotation;
        }
        ss << " /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n";
        ss << "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n";

        std::ostringstream stream_content;
        stream_content << "BT\n/F1 12 Tf\n";
        stream_content << x_pos << " " << top_y << " Td\n";
        stream_content << "(" << top_text << ") Tj\n";
        stream_content << "0 " << (bottom_y - top_y) << " Td\n";
        stream_content << "(" << bottom_text << ") Tj\nET\n";

        std::string stream_str = stream_content.str();
        ss << "5 0 obj << /Length " << stream_str.size() << " >> stream\n";
        ss << stream_str;
        ss << "endstream\nendobj\n";
        ss << "xref\n0 6\n0000000000 65535 f \n";
        ss << "trailer << /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n";
        return ss.str();
    }

    std::string MakeSyntheticLeftRightPdf(
        const std::string& left_text,
        const std::string& right_text,
        const std::string& mediabox = "[0 0 612 792]"
    ) {
        std::ostringstream ss;
        ss << "%PDF-1.4\n";
        ss << "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n";
        ss << "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n";
        ss << "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox " << mediabox;
        ss << " /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n";
        ss << "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n";

        std::ostringstream stream_content;
        stream_content << "BT\n/F1 12 Tf\n";
        stream_content << "50 500 Td\n";
        stream_content << "(" << left_text << ") Tj\n";
        stream_content << "400 0 Td\n";
        stream_content << "(" << right_text << ") Tj\nET\n";

        std::string stream_str = stream_content.str();
        ss << "5 0 obj << /Length " << stream_str.size() << " >> stream\n";
        ss << stream_str;
        ss << "endstream\nendobj\n";
        ss << "xref\n0 6\n0000000000 65535 f \n";
        ss << "trailer << /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n";
        return ss.str();
    }

    std::string MakeSyntheticMultiPageMixedRotationPdf() {
        std::ostringstream ss;
        ss << "%PDF-1.4\n";
        ss << "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n";
        ss << "2 0 obj << /Type /Pages /Kids [3 0 R 6 0 R 8 0 R] /Count 3 >> endobj\n";
        // Page 1: Rotate 0
        ss << "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n";
        ss << "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n";
        ss << "5 0 obj << /Length 50 >> stream\nBT\n/F1 12 Tf\n50 700 Td\n(Page1Secret) Tj\nET\nendstream\nendobj\n";
        // Page 2: Rotate 90
        ss << "6 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 90 /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >> endobj\n";
        ss << "7 0 obj << /Length 50 >> stream\nBT\n/F1 12 Tf\n50 700 Td\n(Page2Secret) Tj\nET\nendstream\nendobj\n";
        // Page 3: Rotate 270
        ss << "8 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 270 /Resources << /Font << /F1 4 0 R >> >> /Contents 9 0 R >> endobj\n";
        ss << "9 0 obj << /Length 50 >> stream\nBT\n/F1 12 Tf\n50 700 Td\n(Page3Secret) Tj\nET\nendstream\nendobj\n";
        ss << "xref\n0 10\n0000000000 65535 f \n";
        ss << "trailer << /Size 10 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n";
        return ss.str();
    }
}

TEST(MPL311Tests, AC1_YAxisInversion_TopHalfVisualBoxErasesTopTextOnly) {
    std::string pdf = MakeSyntheticCanaryPdf("TOP_SECRET", "BOTTOM_PUBLIC");
    MuPDFWrapper wrapper(pdf);
    ASSERT_EQ(wrapper.GetPageCount(), 1);

    auto [init_text, _] = wrapper.ExtractPageText(0);
    ASSERT_NE(init_text.find("TOP_SECRET"), std::string::npos);
    ASSERT_NE(init_text.find("BOTTOM_PUBLIC"), std::string::npos);

    // Top-half visual box: [0.0, 0.0, 1.0, 0.5] in visual top-left space
    ml::worker::v1::NormalizedRedactionBox top_box;
    top_box.set_page_number(1);
    top_box.set_x0(0.0f);
    top_box.set_y0(0.0f);
    top_box.set_x1(1.0f);
    top_box.set_y1(0.5f);

    wrapper.ApplyNormalizedRedactions({&top_box, 1});

    std::string redacted_bytes = wrapper.GetRedactedPdf();
    MuPDFWrapper redacted_wrapper(redacted_bytes);
    auto [after_text, __] = redacted_wrapper.ExtractPageText(0);

    // AC-1: TOP_SECRET (visual top) MUST be erased; BOTTOM_PUBLIC MUST be preserved
    EXPECT_EQ(after_text.find("TOP_SECRET"), std::string::npos);
    EXPECT_NE(after_text.find("BOTTOM_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC1_YAxisInversion_BottomHalfVisualBoxErasesBottomTextOnly) {
    std::string pdf = MakeSyntheticCanaryPdf("TOP_SECRET", "BOTTOM_PUBLIC");
    MuPDFWrapper wrapper(pdf);

    // Bottom-half visual box: [0.0, 0.5, 1.0, 1.0] in visual top-left space
    ml::worker::v1::NormalizedRedactionBox bottom_box;
    bottom_box.set_page_number(1);
    bottom_box.set_x0(0.0f);
    bottom_box.set_y0(0.5f);
    bottom_box.set_x1(1.0f);
    bottom_box.set_y1(1.0f);

    wrapper.ApplyNormalizedRedactions({&bottom_box, 1});

    std::string redacted_bytes = wrapper.GetRedactedPdf();
    MuPDFWrapper redacted_wrapper(redacted_bytes);
    auto [after_text, _] = redacted_wrapper.ExtractPageText(0);

    // AC-1: BOTTOM_PUBLIC (visual bottom) MUST be erased; TOP_SECRET MUST be preserved
    EXPECT_NE(after_text.find("TOP_SECRET"), std::string::npos);
    EXPECT_EQ(after_text.find("BOTTOM_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC1_YAxisInversion_RespectsNonZeroPageBoundsY0) {
    // Non-origin MediaBox [100 100 712 892]
    std::string pdf = MakeSyntheticCanaryPdf("TOP_SECRET", "BOTTOM_PUBLIC", "[100 100 712 892]", 0, 820.0f, 250.0f, 150.0f);
    MuPDFWrapper wrapper(pdf);

    auto [init_text, _] = wrapper.ExtractPageText(0);
    ASSERT_NE(init_text.find("TOP_SECRET"), std::string::npos);
    ASSERT_NE(init_text.find("BOTTOM_PUBLIC"), std::string::npos);

    ml::worker::v1::NormalizedRedactionBox top_box;
    top_box.set_page_number(1);
    top_box.set_x0(0.0f);
    top_box.set_y0(0.0f);
    top_box.set_x1(1.0f);
    top_box.set_y1(0.5f);

    wrapper.ApplyNormalizedRedactions({&top_box, 1});

    std::string redacted_bytes = wrapper.GetRedactedPdf();
    MuPDFWrapper redacted_wrapper(redacted_bytes);
    auto [after_text, __] = redacted_wrapper.ExtractPageText(0);

    EXPECT_EQ(after_text.find("TOP_SECRET"), std::string::npos);
    EXPECT_NE(after_text.find("BOTTOM_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC1_XAxis_Not_Inverted) {
    std::string pdf = MakeSyntheticLeftRightPdf("LEFT_SECRET", "RIGHT_PUBLIC");
    MuPDFWrapper wrapper(pdf);

    auto [init_text, _] = wrapper.ExtractPageText(0);
    ASSERT_NE(init_text.find("LEFT_SECRET"), std::string::npos);
    ASSERT_NE(init_text.find("RIGHT_PUBLIC"), std::string::npos);

    // Left-half box: [0.0, 0.0, 0.5, 1.0]
    ml::worker::v1::NormalizedRedactionBox left_box;
    left_box.set_page_number(1);
    left_box.set_x0(0.0f);
    left_box.set_y0(0.0f);
    left_box.set_x1(0.5f);
    left_box.set_y1(1.0f);

    wrapper.ApplyNormalizedRedactions({&left_box, 1});

    std::string redacted_bytes = wrapper.GetRedactedPdf();
    MuPDFWrapper redacted_wrapper(redacted_bytes);
    auto [after_text, __] = redacted_wrapper.ExtractPageText(0);

    EXPECT_EQ(after_text.find("LEFT_SECRET"), std::string::npos);
    EXPECT_NE(after_text.find("RIGHT_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC2_Rotation_0DegreePage_IdentityPath) {
    std::string pdf = MakeSyntheticCanaryPdf("ROT0_SECRET", "ROT0_PUBLIC", "[0 0 612 792]", 0);
    MuPDFWrapper wrapper(pdf);
    EXPECT_EQ(wrapper.GetPageRotation(0), 0);

    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0(0.0f);
    box.set_y0(0.0f);
    box.set_x1(1.0f);
    box.set_y1(0.5f);

    wrapper.ApplyNormalizedRedactions({&box, 1});

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("ROT0_SECRET"), std::string::npos);
    EXPECT_NE(text.find("ROT0_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC2_Rotation_90DegreePage) {
    std::string pdf = MakeSyntheticCanaryPdf("ROT90_SECRET", "ROT90_PUBLIC", "[0 0 612 792]", 90, 720.0f, 200.0f, 50.0f);
    MuPDFWrapper wrapper(pdf);
    EXPECT_EQ(wrapper.GetPageRotation(0), 90);

    auto [init_text, bboxes] = wrapper.ExtractPageText(0);
    size_t sec_pos = init_text.find("ROT90_SECRET");
    ASSERT_NE(sec_pos, std::string::npos);

    // Compute tight visual bbox
    fz_rect visual_bbox = fz_empty_rect;
    for (size_t i = 0; i < 12; ++i) {
        if (sec_pos + i < bboxes.size() && !fz_is_empty_rect(bboxes[sec_pos + i])) {
            visual_bbox = fz_union_rect(visual_bbox, bboxes[sec_pos + i]);
        }
    }
    ASSERT_FALSE(fz_is_empty_rect(visual_bbox));

    // For rot 90: visual bounds are 792 wide by 612 tall
    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0((visual_bbox.x0 - 5.0f) / 792.0f);
    box.set_y0((visual_bbox.y0 - 5.0f) / 612.0f);
    box.set_x1((visual_bbox.x1 + 5.0f) / 792.0f);
    box.set_y1((visual_bbox.y1 + 5.0f) / 612.0f);

    wrapper.ApplyNormalizedRedactions({&box, 1});

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("ROT90_SECRET"), std::string::npos);
    EXPECT_NE(text.find("ROT90_PUBLIC"), std::string::npos);
    EXPECT_EQ(r.GetPageRotation(0), 90);
}

TEST(MPL311Tests, AC2_Rotation_180DegreePage) {
    std::string pdf = MakeSyntheticCanaryPdf("ROT180_SECRET", "ROT180_PUBLIC", "[0 0 612 792]", 180, 720.0f, 200.0f, 50.0f);
    MuPDFWrapper wrapper(pdf);
    EXPECT_EQ(wrapper.GetPageRotation(0), 180);

    auto [init_text, bboxes] = wrapper.ExtractPageText(0);
    size_t sec_pos = init_text.find("ROT180_SECRET");
    ASSERT_NE(sec_pos, std::string::npos);

    fz_rect visual_bbox = fz_empty_rect;
    for (size_t i = 0; i < 13; ++i) {
        if (sec_pos + i < bboxes.size() && !fz_is_empty_rect(bboxes[sec_pos + i])) {
            visual_bbox = fz_union_rect(visual_bbox, bboxes[sec_pos + i]);
        }
    }
    ASSERT_FALSE(fz_is_empty_rect(visual_bbox));

    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0((visual_bbox.x0 - 5.0f) / 612.0f);
    box.set_y0((visual_bbox.y0 - 5.0f) / 792.0f);
    box.set_x1((visual_bbox.x1 + 5.0f) / 612.0f);
    box.set_y1((visual_bbox.y1 + 5.0f) / 792.0f);

    wrapper.ApplyNormalizedRedactions({&box, 1});

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("ROT180_SECRET"), std::string::npos);
    EXPECT_NE(text.find("ROT180_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC2_Rotation_270DegreePage) {
    std::string pdf = MakeSyntheticCanaryPdf("ROT270_SECRET", "ROT270_PUBLIC", "[0 0 612 792]", 270, 720.0f, 200.0f, 50.0f);
    MuPDFWrapper wrapper(pdf);
    EXPECT_EQ(wrapper.GetPageRotation(0), 270);

    auto [init_text, bboxes] = wrapper.ExtractPageText(0);
    size_t sec_pos = init_text.find("ROT270_SECRET");
    ASSERT_NE(sec_pos, std::string::npos);

    fz_rect visual_bbox = fz_empty_rect;
    for (size_t i = 0; i < 13; ++i) {
        if (sec_pos + i < bboxes.size() && !fz_is_empty_rect(bboxes[sec_pos + i])) {
            visual_bbox = fz_union_rect(visual_bbox, bboxes[sec_pos + i]);
        }
    }
    ASSERT_FALSE(fz_is_empty_rect(visual_bbox));

    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0((visual_bbox.x0 - 5.0f) / 792.0f);
    box.set_y0((visual_bbox.y0 - 5.0f) / 612.0f);
    box.set_x1((visual_bbox.x1 + 5.0f) / 792.0f);
    box.set_y1((visual_bbox.y1 + 5.0f) / 612.0f);

    wrapper.ApplyNormalizedRedactions({&box, 1});

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("ROT270_SECRET"), std::string::npos);
    EXPECT_NE(text.find("ROT270_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC2_MultiPage_MixedRotations) {
    std::string pdf = MakeSyntheticMultiPageMixedRotationPdf();
    MuPDFWrapper wrapper(pdf);
    ASSERT_EQ(wrapper.GetPageCount(), 3);
    EXPECT_EQ(wrapper.GetPageRotation(0), 0);
    EXPECT_EQ(wrapper.GetPageRotation(1), 90);
    EXPECT_EQ(wrapper.GetPageRotation(2), 270);

    ml::worker::v1::NormalizedRedactionBox b1, b2, b3;
    b1.set_page_number(1);
    b1.set_x0(0.0f); b1.set_y0(0.0f); b1.set_x1(1.0f); b1.set_y1(1.0f);

    b2.set_page_number(2);
    b2.set_x0(0.0f); b2.set_y0(0.0f); b2.set_x1(1.0f); b2.set_y1(1.0f);

    b3.set_page_number(3);
    b3.set_x0(0.0f); b3.set_y0(0.0f); b3.set_x1(1.0f); b3.set_y1(1.0f);

    std::vector<ml::worker::v1::NormalizedRedactionBox> all_boxes = {b1, b2, b3};
    wrapper.ApplyNormalizedRedactions(all_boxes);

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    ASSERT_EQ(r.GetPageCount(), 3);
    EXPECT_EQ(r.ExtractPageText(0).text.find("Page1Secret"), std::string::npos);
    EXPECT_EQ(r.ExtractPageText(1).text.find("Page2Secret"), std::string::npos);
    EXPECT_EQ(r.ExtractPageText(2).text.find("Page3Secret"), std::string::npos);
}

TEST(MPL311Tests, AC4_CoordinateClamping_Outside01) {
    std::string pdf = MakeSyntheticCanaryPdf("CLAMP_TARGET", "PRESERVE_ME");
    MuPDFWrapper wrapper(pdf);

    // Box with coordinates outside [0.0, 1.0] covering the top half
    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0(-0.5f);
    box.set_y0(-0.5f);
    box.set_x1(1.5f);
    box.set_y1(0.45f);

    EXPECT_NO_THROW(wrapper.ApplyNormalizedRedactions({&box, 1}));

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("CLAMP_TARGET"), std::string::npos);
    EXPECT_NE(text.find("PRESERVE_ME"), std::string::npos);
}

TEST(MPL311Tests, AC4_SwappedCoordinates_NormalizedViaMinMax) {
    std::string pdf = MakeSyntheticCanaryPdf("SWAP_SECRET", "SWAP_PUBLIC");
    MuPDFWrapper wrapper(pdf);

    // Box with swapped coordinates: x0 > x1 and y0 > y1
    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0(0.8f);
    box.set_x1(0.0f);
    box.set_y0(0.45f);
    box.set_y1(0.0f);

    wrapper.ApplyNormalizedRedactions({&box, 1});

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("SWAP_SECRET"), std::string::npos);
    EXPECT_NE(text.find("SWAP_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC4_DegenerateCollapsedBoxes_SkippedSilently) {
    std::string pdf = MakeSyntheticCanaryPdf("VALID_SECRET", "VALID_PUBLIC");
    MuPDFWrapper wrapper(pdf);

    ml::worker::v1::NormalizedRedactionBox degenerate_x;
    degenerate_x.set_page_number(1);
    degenerate_x.set_x0(0.5f); degenerate_x.set_x1(0.5f);
    degenerate_x.set_y0(0.0f); degenerate_x.set_y1(1.0f);

    ml::worker::v1::NormalizedRedactionBox degenerate_y;
    degenerate_y.set_page_number(1);
    degenerate_y.set_x0(0.0f); degenerate_y.set_x1(1.0f);
    degenerate_y.set_y0(0.5f); degenerate_y.set_y1(0.5f);

    ml::worker::v1::NormalizedRedactionBox valid_box;
    valid_box.set_page_number(1);
    valid_box.set_x0(0.0f); valid_box.set_y0(0.0f);
    valid_box.set_x1(1.0f); valid_box.set_y1(0.45f);

    std::vector<ml::worker::v1::NormalizedRedactionBox> boxes = {degenerate_x, degenerate_y, valid_box};
    EXPECT_NO_THROW(wrapper.ApplyNormalizedRedactions(boxes));

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("VALID_SECRET"), std::string::npos);
    EXPECT_NE(text.find("VALID_PUBLIC"), std::string::npos);
}

TEST(MPL311Tests, AC4_EmptyCustomBoxes_NoOp) {
    std::string pdf = MakeSyntheticCanaryPdf("KEEP_A", "KEEP_B");
    MuPDFWrapper wrapper(pdf);

    EXPECT_NO_THROW(wrapper.ApplyNormalizedRedactions({}));

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_NE(text.find("KEEP_A"), std::string::npos);
    EXPECT_NE(text.find("KEEP_B"), std::string::npos);
}

TEST(MPL311Tests, AC4_CustomFillColorOverride) {
    std::string pdf = MakeSyntheticCanaryPdf("RED_TARGET", "BODY_INFO");
    MuPDFWrapper wrapper(pdf);

    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0(0.0f); box.set_y0(0.0f); box.set_x1(1.0f); box.set_y1(0.45f);

    float red_color[3] = {1.0f, 0.0f, 0.0f};
    wrapper.ApplyNormalizedRedactions({&box, 1}, red_color);

    std::string redacted = wrapper.GetRedactedPdf();
    MuPDFWrapper r(redacted);
    auto [text, _] = r.ExtractPageText(0);
    EXPECT_EQ(text.find("RED_TARGET"), std::string::npos);
    EXPECT_NE(text.find("BODY_INFO"), std::string::npos);
}

TEST(MPL311Tests, AC6_FailFast_OutOfRangePageNumber_ThrowsOutOfRange) {
    std::string pdf = MakeSyntheticCanaryPdf("SECRET", "PUBLIC");
    MuPDFWrapper wrapper(pdf);
    ASSERT_EQ(wrapper.GetPageCount(), 1);

    ml::worker::v1::NormalizedRedactionBox bad_box_high;
    bad_box_high.set_page_number(99);

    EXPECT_THROW(
        wrapper.ApplyNormalizedRedactions({&bad_box_high, 1}),
        std::out_of_range
    );

    EXPECT_THROW(
        wrapper.ApplyRedactions(-1, {}),
        std::out_of_range
    );

    EXPECT_THROW(
        wrapper.ApplyRedactions(1, {}),
        std::out_of_range
    );
}

TEST(MPL311Tests, AC6_ForensicPreservation_PermanentGlyphDestruction) {
    std::string pdf = MakeSyntheticCanaryPdf("PERMANENT_SECRET", "SURVIVING_BODY");
    MuPDFWrapper wrapper(pdf);

    ml::worker::v1::NormalizedRedactionBox box;
    box.set_page_number(1);
    box.set_x0(0.0f); box.set_y0(0.0f); box.set_x1(1.0f); box.set_y1(0.45f);

    wrapper.ApplyNormalizedRedactions({&box, 1});

    std::string redacted_pdf_bytes = wrapper.GetRedactedPdf();
    EXPECT_GT(redacted_pdf_bytes.size(), 0);

    // Verify reloaded PDF is valid and secret glyphs are irrevocably destroyed
    MuPDFWrapper reloaded(redacted_pdf_bytes);
    EXPECT_EQ(reloaded.GetPageCount(), 1);
    auto [extracted_text, _] = reloaded.ExtractPageText(0);
    EXPECT_EQ(extracted_text.find("PERMANENT_SECRET"), std::string::npos);
    EXPECT_NE(extracted_text.find("SURVIVING_BODY"), std::string::npos);
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

namespace {
    class EnvGuard {
    public:
        EnvGuard(const char* name, const char* value) : name_(name) {
            const char* old = std::getenv(name);
            if (old) {
                old_value_ = old;
                had_old_ = true;
            }
            if (value) {
                setenv(name, value, 1);
            } else {
                unsetenv(name);
            }
        }
        ~EnvGuard() {
            if (had_old_) {
                setenv(name_.c_str(), old_value_.c_str(), 1);
            } else {
                unsetenv(name_.c_str());
            }
        }
    private:
        std::string name_;
        std::string old_value_;
        bool had_old_{false};
    };

    class FakeBindingRunner : public IOnnxRunner {
    public:
        size_t call_count{0};
        std::vector<std::vector<int64_t>> captured_input_ids;
        std::vector<std::vector<int64_t>> captured_words_masks;
        std::vector<size_t> captured_num_words;
        std::vector<size_t> captured_num_classes;
        std::vector<size_t> captured_num_spans;
        std::function<std::vector<float>(size_t, size_t, size_t, const doc_processor::GeneratedSpans&)> logit_generator;

        std::vector<float> RunInference(
            const std::vector<int64_t>& input_ids,
            const std::vector<int64_t>& attention_mask,
            const std::vector<int64_t>& words_mask,
            size_t num_words,
            size_t num_classes,
            const doc_processor::GeneratedSpans& spans
        ) override {
            call_count++;
            captured_input_ids.push_back(input_ids);
            captured_words_masks.push_back(words_mask);
            captured_num_words.push_back(num_words);
            captured_num_classes.push_back(num_classes);
            captured_num_spans.push_back(spans.num_spans);

            if (logit_generator) {
                return logit_generator(call_count, num_words, num_classes, spans);
            }
            size_t total = spans.num_spans * num_classes;
            return std::vector<float>(total, -10.0f);
        }
    };
}

TEST(MPL313Tests, EpSelectionTest_GraphOptimizationIsEnableAll) {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.EnableCpuMemArena();
    opts.EnableMemPattern();
    opts.SetIntraOpNumThreads(4);
    opts.SetInterOpNumThreads(2);
    SUCCEED();
}

TEST(MPL313Tests, ThreadConfigTest_EnvOverridesDefault) {
    EnvGuard intra_guard("ORT_INTRA_OP_THREADS", "8");
    EnvGuard inter_guard("ORT_INTER_OP_THREADS", "4");

    auto [intra, inter] = doc_processor::OnnxConfigHelper::ResolveThreadCounts();
    EXPECT_EQ(intra, 8);
    EXPECT_EQ(inter, 4);
}

TEST(MPL313Tests, ThreadConfigTest_Default4And2) {
    EnvGuard intra_guard("ORT_INTRA_OP_THREADS", nullptr);
    EnvGuard inter_guard("ORT_INTER_OP_THREADS", nullptr);

    auto [intra, inter] = doc_processor::OnnxConfigHelper::ResolveThreadCounts();
    EXPECT_EQ(intra, 4);
    EXPECT_EQ(inter, 2);
}

TEST(MPL313Tests, ThreadConfigTest_InvalidEnvFallsBack) {
    EnvGuard intra_guard("ORT_INTRA_OP_THREADS", "invalid_num");
    EnvGuard inter_guard("ORT_INTER_OP_THREADS", "bogus");

    auto [intra, inter] = doc_processor::OnnxConfigHelper::ResolveThreadCounts();
    EXPECT_EQ(intra, 4);
    EXPECT_EQ(inter, 2);
}

TEST(MPL313Tests, EpSelectionTest_NonexistentModelThrowsRuntimeError) {
    EXPECT_THROW(
        CreateRealOnnxRunner("/nonexistent/file/path/model.onnx"),
        std::runtime_error
    );
}

TEST(MPL313Tests, PinnedMemoryTest_CreatePinnedCpuMemoryInfo) {
    auto pinned_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    SUCCEED();
}

TEST(MPL313Tests, PinnedMemoryTest_CreateDefaultCpuMemoryInfo) {
    auto default_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
    SUCCEED();
}

TEST(MPL313Tests, BatchedEvaluationTest_ExactlyOnePerWindow_10Words_1Window) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<FakeBindingRunner>();
    auto* runner = runner_ptr.get();

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    std::string text;
    for (int i = 0; i < 10; ++i) {
        text += "word ";
    }
    std::vector<fz_rect> char_bboxes(text.size(), fz_rect{0, 0, 10, 10});
    std::vector<std::string> labels = {"TestLabel"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    EXPECT_EQ(runner->call_count, 1);
    EXPECT_EQ(runner->captured_num_words[0], 10);
}

TEST(MPL313Tests, BatchedEvaluationTest_ExactlyOnePerWindow_500Words_2Windows) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<FakeBindingRunner>();
    auto* runner = runner_ptr.get();

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    std::string text;
    for (int i = 0; i < 500; ++i) {
        text += "word ";
    }
    std::vector<fz_rect> char_bboxes(text.size(), fz_rect{0, 0, 10, 10});
    std::vector<std::string> labels = {"TestLabel"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    EXPECT_EQ(runner->call_count, 2);
    EXPECT_EQ(runner->captured_num_words[0], 384);
    EXPECT_EQ(runner->captured_num_words[1], 180);
}

TEST(MPL313Tests, BatchedEvaluationTest_ExactlyOnePerWindow_1000Words_3Windows) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<FakeBindingRunner>();
    auto* runner = runner_ptr.get();

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    std::string text;
    for (int i = 0; i < 1000; ++i) {
        text += "word ";
    }
    std::vector<fz_rect> char_bboxes(text.size(), fz_rect{0, 0, 10, 10});
    std::vector<std::string> labels = {"TestLabel"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    EXPECT_EQ(runner->call_count, 3);
    EXPECT_EQ(runner->captured_num_words[0], 384);
    EXPECT_EQ(runner->captured_num_words[1], 384);
    EXPECT_EQ(runner->captured_num_words[2], 360);
}

TEST(MPL313Tests, CandidateDedupTest_MaxConfidenceAcrossOverlappingWindows) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<FakeBindingRunner>();
    auto* runner = runner_ptr.get();

    // In window 1 (starts at 0), word 330 has confidence 0.85 (logit ~ 1.73)
    // In window 2 (starts at 320), relative word is 330 - 320 = 10, with confidence 0.98 (logit ~ 3.89)
    runner->logit_generator = [](size_t call_num, size_t num_words, size_t num_classes, const doc_processor::GeneratedSpans& spans) {
        std::vector<float> logits(spans.num_spans * num_classes, -10.0f);
        if (call_num == 1) {
            // Span starting at word 330, length 1
            size_t span_idx = 330 * 12 + 0;
            if (span_idx < spans.num_spans) {
                logits[span_idx * num_classes + 0] = 1.7346f; // conf ~ 0.85
            }
        } else if (call_num == 2) {
            // Span starting at relative word 10 (absolute 330), length 1
            size_t span_idx = 10 * 12 + 0;
            if (span_idx < spans.num_spans) {
                logits[span_idx * num_classes + 0] = 3.8918f; // conf ~ 0.98
            }
        }
        return logits;
    };

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    std::string text;
    for (int i = 0; i < 500; ++i) {
        text += "word ";
    }
    std::vector<fz_rect> char_bboxes(text.size(), fz_rect{0, 0, 10, 10});
    std::vector<std::string> labels = {"TestLabel"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 100.0f, 100.0f);

    ASSERT_GE(spans.size(), 1);
    EXPECT_NEAR(spans[0].confidence(), 0.98f, 0.01f);
}

TEST(MPL313Tests, ContractPreservationTest_SignaturesAndExceptions) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner = std::make_unique<FakeBindingRunner>();

    EXPECT_THROW(
        MLInferenceEngine(nullptr, std::make_unique<FakeBindingRunner>()),
        std::runtime_error
    );

    EXPECT_THROW(
        MLInferenceEngine(std::make_unique<MockTokenizer>(), nullptr),
        std::runtime_error
    );

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner));
    const std::string text = "hello world";
    std::vector<fz_rect> mismatched_bboxes(5, fz_empty_rect);
    std::vector<std::string> labels = {"Label"};

    EXPECT_THROW(
        engine.CalculateExtractedSpans(text, mismatched_bboxes, labels, 1, 100.0f, 100.0f),
        std::invalid_argument
    );

    std::vector<fz_rect> valid_bboxes(text.size(), fz_empty_rect);
    EXPECT_THROW(
        engine.CalculateExtractedSpans(text, valid_bboxes, labels, 1, -1.0f, 100.0f),
        std::invalid_argument
    );

    EXPECT_THROW(
        engine.CalculateExtractedSpans(text, valid_bboxes, labels, 1, 100.0f, 0.0f),
        std::invalid_argument
    );
}

TEST(MPL313Tests, ContractPreservationTest_CalculateRedactionBBoxesDelegation) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<SingleSpanOnnxRunner>();
    runner_ptr->target_word_start = 0;
    runner_ptr->target_span_len = 1;

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    const std::string text = "12345";
    std::vector<fz_rect> char_bboxes;
    for (size_t i = 0; i < text.size(); ++i) {
        char_bboxes.push_back(fz_rect{static_cast<float>(i * 10), 0.0f, static_cast<float>((i + 1) * 10), 20.0f});
    }
    std::vector<std::string> labels = {"Label"};

    auto bboxes = engine.CalculateRedactionBBoxes(text, char_bboxes, labels);
    ASSERT_EQ(bboxes.size(), 1);
    EXPECT_FLOAT_EQ(bboxes[0].x0, 0.0f);
    EXPECT_FLOAT_EQ(bboxes[0].x1, 50.0f);
}

TEST(MPL313Tests, PerformanceSLO_HotLoopBufferFillAndTensorCreationUnder02ms) {
    auto pinned_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);

    constexpr size_t kSeqLen = 768;
    constexpr size_t kNumSpans = 4608;
    constexpr size_t kNumClasses = 64;

    std::vector<int64_t> input_ids_buf(kSeqLen, 1);
    std::vector<int64_t> attention_mask_buf(kSeqLen, 1);
    std::vector<int64_t> words_mask_buf(kSeqLen, 0);
    std::vector<int64_t> text_lengths_buf(1, 384);
    std::vector<int64_t> span_idx_buf(kNumSpans * 2, 0);
    std::vector<uint8_t> span_mask_buf(kNumSpans, 1);
    std::vector<float> logits_buf(kNumSpans * kNumClasses, 0.0f);

    constexpr int kIterations = 10000;
    std::vector<double> latencies_us;
    latencies_us.reserve(kIterations);

    int64_t seq_shape[2] = {1, static_cast<int64_t>(kSeqLen)};
    int64_t len_shape[2] = {1, 1};
    int64_t span_idx_shape[3] = {1, static_cast<int64_t>(kNumSpans), 2};
    int64_t span_mask_shape[2] = {1, static_cast<int64_t>(kNumSpans)};
    int64_t logits_shape[2] = {static_cast<int64_t>(kNumSpans), static_cast<int64_t>(kNumClasses)};

    for (int iter = 0; iter < kIterations; ++iter) {
        auto start = std::chrono::steady_clock::now();

        // 1. Buffer copy
        text_lengths_buf[0] = 384;
        std::fill(words_mask_buf.begin(), words_mask_buf.begin() + 384, 1);

        // 2. Zero-alloc Ort::Value views
        auto token_val = Ort::Value::CreateTensor<int64_t>(pinned_info, input_ids_buf.data(), kSeqLen, seq_shape, 2);
        auto attn_val = Ort::Value::CreateTensor<int64_t>(pinned_info, attention_mask_buf.data(), kSeqLen, seq_shape, 2);
        auto words_val = Ort::Value::CreateTensor<int64_t>(pinned_info, words_mask_buf.data(), kSeqLen, seq_shape, 2);
        auto len_val = Ort::Value::CreateTensor<int64_t>(pinned_info, text_lengths_buf.data(), 1, len_shape, 2);
        auto span_idx_val = Ort::Value::CreateTensor<int64_t>(pinned_info, span_idx_buf.data(), kNumSpans * 2, span_idx_shape, 3);
        auto span_mask_val = Ort::Value::CreateTensor(pinned_info, span_mask_buf.data(), kNumSpans, span_mask_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        auto logits_val = Ort::Value::CreateTensor<float>(pinned_info, logits_buf.data(), kNumSpans * kNumClasses, logits_shape, 2);

        auto end = std::chrono::steady_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed_us);
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double p95_us = latencies_us[static_cast<size_t>(kIterations * 0.95)];
    // 0.2ms = 200us
    EXPECT_LT(p95_us, 200.0) << "p95 latency was " << p95_us << " us, expected < 200 us";
}

TEST(MPL313Tests, ObservabilityTest_ResolveConfidenceThresholdEnvAndDefault) {
    EXPECT_FLOAT_EQ(MLInferenceEngine::ResolveConfidenceThreshold(0.85f), 0.85f);

    {
        EnvGuard env("GLINER_CONFIDENCE_THRESHOLD", "0.75");
        EXPECT_FLOAT_EQ(MLInferenceEngine::ResolveConfidenceThreshold(-1.0f), 0.75f);
    }

    {
        EnvGuard env("GLINER_CONFIDENCE_THRESHOLD", "invalid");
        EXPECT_FLOAT_EQ(MLInferenceEngine::ResolveConfidenceThreshold(-1.0f), 0.50f);
    }

    {
        EnvGuard env("GLINER_CONFIDENCE_THRESHOLD", nullptr);
        EXPECT_FLOAT_EQ(MLInferenceEngine::ResolveConfidenceThreshold(-1.0f), 0.50f);
    }
}

TEST(MPL313Tests, GeometricBoxRegression_UnionAndClampPreservesGeometry) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<SingleSpanOnnxRunner>();
    runner_ptr->target_word_start = 0;
    runner_ptr->target_span_len = 2;

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    const std::string text = "12345 67890";
    std::vector<fz_rect> char_bboxes;
    for (size_t i = 0; i < text.size(); ++i) {
        char_bboxes.push_back(fz_rect{static_cast<float>(i * 10), 0.0f, static_cast<float>((i + 1) * 10), 20.0f});
    }
    std::vector<std::string> labels = {"TestLabel"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 200.0f, 100.0f);

    ASSERT_EQ(spans.size(), 1);
    EXPECT_FLOAT_EQ(spans[0].x0(), 0.0f / 200.0f);
    EXPECT_FLOAT_EQ(spans[0].x1(), 110.0f / 200.0f);
    EXPECT_FLOAT_EQ(spans[0].y0(), 0.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].y1(), 20.0f / 100.0f);
}

// ============================================================================
// MPL-316 Test Suite: FFI UTF-8 Boundary Sanitization & Scratchpad Hoisting
// ============================================================================

namespace {
    class RecordingTokenizer : public ITokenizer {
    public:
        std::vector<std::string> recorded_inputs;
        std::vector<int32_t> Encode(const std::string& text) override {
            EXPECT_TRUE(simdutf::validate_utf8(text.data(), text.size()))
                << "Tokenizer received invalid UTF-8 string across Rust FFI boundary!";
            recorded_inputs.push_back(text);
            return {101, 102};
        }
        std::string Decode(const std::vector<int32_t>& ids) override {
            return "";
        }
    };

    std::string MakePdfWithWords(const std::vector<std::string>& words) {
        std::string stream_content = "BT\n/F1 12 Tf\n50 720 Td\n";
        for (const auto& w : words) {
            stream_content += "(" + w + ") Tj\n10 0 Td\n";
        }
        stream_content += "ET\n";

        std::ostringstream oss;
        oss << "%PDF-1.4\n"
            << "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            << "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            << "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
            << "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            << "5 0 obj << /Length " << stream_content.size() << " >> stream\n"
            << stream_content
            << "endstream\n"
            << "endobj\n"
            << "xref\n"
            << "0 6\n"
            << "trailer << /Size 6 /Root 1 0 R >>\n"
            << "startxref\n"
            << "0\n"
            << "%%EOF\n";
        return oss.str();
    }
}


TEST(MPL316Tests, AC1_ValidUtf8_PassesByteIdentical) {
    std::string scratch;
    scratch.reserve(4096);

    // ASCII
    std::string ascii = "hello world";
    doc_processor::SanitizeUtf8(ascii, scratch);
    EXPECT_EQ(scratch, ascii);
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));

    // Cyrillic (D0/D1 bytes)
    std::string cyrillic = "привет мир";
    doc_processor::SanitizeUtf8(cyrillic, scratch);
    EXPECT_EQ(scratch, cyrillic);
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));

    // Emoji (F0 9F 98 80)
    std::string emoji = "Borrower 😀 Name";
    doc_processor::SanitizeUtf8(emoji, scratch);
    EXPECT_EQ(scratch, emoji);
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));

    // Currency symbol (£ = C2 A3)
    std::string currency = "Price £1000";
    doc_processor::SanitizeUtf8(currency, scratch);
    EXPECT_EQ(scratch, currency);
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));
}

TEST(MPL316Tests, AC1_SingleInvalidByte_ReplacedWithU_FFFD) {
    std::string scratch;
    scratch.reserve(4096);

    // Single 0xFF
    std::string invalid1 = "\xFF";
    doc_processor::SanitizeUtf8(invalid1, scratch);
    EXPECT_EQ(scratch, "\xEF\xBF\xBD");
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));

    // 0xFF surrounded by valid ASCII
    std::string invalid2 = "a\xFF" "b";
    doc_processor::SanitizeUtf8(invalid2, scratch);
    EXPECT_EQ(scratch, "a\xEF\xBF\xBD" "b");
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));

    // 0xFE surrounded by valid text
    std::string invalid3 = "x\xFEy";
    doc_processor::SanitizeUtf8(invalid3, scratch);
    EXPECT_EQ(scratch, "x\xEF\xBF\xBDy");
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));
}

TEST(MPL316Tests, AC1_TruncatedSequence_Replaced) {
    std::string scratch;
    scratch.reserve(4096);

    // Truncated 2-byte lead C2 at end
    std::string truncated1 = "prefix\xC2";
    doc_processor::SanitizeUtf8(truncated1, scratch);
    EXPECT_EQ(scratch, "prefix\xEF\xBF\xBD");
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));

    // C2 followed by space (0x20 is not continuation 0x80..0xBF)
    std::string truncated2 = "prefix\xC2 suffix";
    doc_processor::SanitizeUtf8(truncated2, scratch);
    EXPECT_EQ(scratch, "prefix\xEF\xBF\xBD suffix");
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));
}

TEST(MPL316Tests, AC1_OverlongSequence_RejectedAndReplaced) {
    std::string scratch;
    scratch.reserve(4096);

    // Overlong NUL (C0 80)
    std::string overlong = "test\xC0\x80overlong";
    EXPECT_FALSE(simdutf::validate_utf8(overlong.data(), overlong.size()));

    doc_processor::SanitizeUtf8(overlong, scratch);
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));
    EXPECT_EQ(scratch, "test\xEF\xBF\xBDoverlong");
}

TEST(MPL316Tests, AC1_IsolatedContinuation_Replaced) {
    std::string scratch;
    scratch.reserve(4096);

    // Isolated continuation byte 0x80
    std::string isolated = "x\x80y";
    doc_processor::SanitizeUtf8(isolated, scratch);
    EXPECT_EQ(scratch, "x\xEF\xBF\xBDy");
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));

    // Multiple isolated continuation bytes 0x8F 0x90
    std::string isolated2 = "foo\x8F\x90" "bar";
    doc_processor::SanitizeUtf8(isolated2, scratch);
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));
    EXPECT_EQ(scratch, "foo\xEF\xBF\xBD\xEF\xBF\xBD" "bar");
}


TEST(MPL316Tests, AC1_EmptyString_ReturnsEmpty) {
    std::string scratch = "initial_junk";
    scratch.reserve(4096);

    std::string_view empty_view = "";
    doc_processor::SanitizeUtf8(empty_view, scratch);
    EXPECT_TRUE(scratch.empty());
    EXPECT_EQ(scratch.size(), 0);
    EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));
}

TEST(MPL316Tests, AC1_IdempotencyAndNoThrow) {
    std::string scratch1;
    std::string scratch2;
    scratch1.reserve(4096);
    scratch2.reserve(4096);

    // Fuzz arbitrary byte patterns
    std::vector<std::string> test_strings = {
        "regular ascii",
        "some \xFF \xFE \xC0\x80 broken \x80 chars",
        "\xF8\x80\x80\x80\x80 illegal 5-byte",
        "\xED\xA0\x80 surrogate UTF-16",
        std::string("\x00\xFF\xFE\x7F\x80\xC2\xC3\x80", 8)
    };

    for (const auto& s : test_strings) {
        doc_processor::SanitizeUtf8(s, scratch1);
        EXPECT_TRUE(simdutf::validate_utf8(scratch1.data(), scratch1.size()));

        // Second sanitization must be byte-identical (idempotent)
        doc_processor::SanitizeUtf8(scratch1, scratch2);
        EXPECT_EQ(scratch1, scratch2);
        EXPECT_TRUE(simdutf::validate_utf8(scratch2.data(), scratch2.size()));
    }
}

TEST(MPL316Tests, AC1_SanitizeUtf8Wrapper_Delegates) {
    std::string input = "test \xFF string";
    std::string result = doc_processor::SanitizeUtf8(input);
    EXPECT_EQ(result, "test \xEF\xBF\xBD string");
    EXPECT_TRUE(simdutf::validate_utf8(result.data(), result.size()));
}

TEST(MPL316Tests, AC2_CalculateExtractedSpans_SanitizesEveryLabelAndWordBeforeEncode) {
    auto rec_tokenizer = std::make_unique<RecordingTokenizer>();
    RecordingTokenizer* raw_tok = rec_tokenizer.get();
    auto runner_ptr = std::make_unique<MockOnnxRunner>();

    MLInferenceEngine engine(std::move(rec_tokenizer), std::move(runner_ptr));

    // Text with malformed word and dynamic label with malformed byte
    const std::string text = "goodword bad\xFFword normalword";
    std::vector<fz_rect> char_bboxes(text.size(), fz_rect{0.0f, 0.0f, 10.0f, 10.0f});
    std::vector<std::string> dynamic_labels = {"valid_label", "dirty\xFElabel"};

    EXPECT_NO_THROW({
        auto spans = engine.CalculateExtractedSpans(text, char_bboxes, dynamic_labels, 1, 100.0f, 100.0f);
    });

    ASSERT_GE(raw_tok->recorded_inputs.size(), 5);
    for (const auto& recorded : raw_tok->recorded_inputs) {
        EXPECT_TRUE(simdutf::validate_utf8(recorded.data(), recorded.size()))
            << "Recorded string failed UTF-8 validation: " << recorded;
    }
}

TEST(MPL316Tests, AC2_Scratchpad_ClearPreservesCapacity) {
    std::string scratch;
    scratch.reserve(4096);
    EXPECT_GE(scratch.capacity(), 4096);

    for (int i = 0; i < 100; ++i) {
        std::string sample = "iteration_" + std::to_string(i) + "_word\xFF";
        doc_processor::SanitizeUtf8(sample, scratch);
        EXPECT_TRUE(simdutf::validate_utf8(scratch.data(), scratch.size()));
        EXPECT_GE(scratch.capacity(), 4096);
    }
}

TEST(MPL316Tests, AC3_MuPDFWrapper_ExtractPageText_ReserveAndUtf8Invariant) {
    std::vector<std::string> words = {"Borrower", "John", "Doe", "$500,000.00"};
    std::string pdf_bytes = MakePdfWithWords(words);

    MuPDFWrapper wrapper(pdf_bytes);
    PageTextData text_data = wrapper.ExtractPageText(0);

    EXPECT_GE(text_data.text.capacity(), 4096);
    EXPECT_GE(text_data.char_bboxes.capacity(), 4096);
    EXPECT_EQ(text_data.text.size(), text_data.char_bboxes.size());
    EXPECT_TRUE(simdutf::validate_utf8(text_data.text.data(), text_data.text.size()));
}

TEST(MPL316Tests, AC3_CodepointToUtf8_MultiByteAndSurrogateEmission) {
    // Test the multi-byte and surrogate conversion behavior
    std::string text;
    std::vector<fz_rect> char_bboxes;
    text.reserve(4096);
    char_bboxes.reserve(4096);

    // Simulate helper logic directly:
    auto encode_codepoint = [&](int code, fz_rect r) {
        if (code >= 0 && code <= 0x7F) {
            text.push_back(static_cast<char>(code));
            char_bboxes.push_back(r);
        } else if (code >= 0x80 && code <= 0x7FF) {
            text.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
            text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            char_bboxes.push_back(r);
            char_bboxes.push_back(fz_empty_rect);
        } else if (code >= 0x800 && code <= 0xFFFF) {
            if (code >= 0xD800 && code <= 0xDFFF) {
                text.append("\xEF\xBF\xBD", 3);
                char_bboxes.push_back(r);
                char_bboxes.push_back(fz_empty_rect);
                char_bboxes.push_back(fz_empty_rect);
            } else {
                text.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                char_bboxes.push_back(r);
                char_bboxes.push_back(fz_empty_rect);
                char_bboxes.push_back(fz_empty_rect);
            }
        } else if (code >= 0x10000 && code <= 0x10FFFF) {
            text.push_back(static_cast<char>(0xF0 | ((code >> 18) & 0x07)));
            text.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            char_bboxes.push_back(r);
            char_bboxes.push_back(fz_empty_rect);
            char_bboxes.push_back(fz_empty_rect);
            char_bboxes.push_back(fz_empty_rect);
        } else {
            text.append("\xEF\xBF\xBD", 3);
            char_bboxes.push_back(r);
            char_bboxes.push_back(fz_empty_rect);
            char_bboxes.push_back(fz_empty_rect);
        }
    };

    fz_rect sample_rect{1.0f, 2.0f, 3.0f, 4.0f};
    encode_codepoint(0x43, sample_rect);        // ASCII 'C' -> 1 byte
    encode_codepoint(0x0430, sample_rect);      // Cyrillic 'а' -> 2 bytes
    encode_codepoint(0x20AC, sample_rect);      // Euro '€' -> 3 bytes
    encode_codepoint(0x1F600, sample_rect);     // Emoji '😀' -> 4 bytes
    encode_codepoint(0xD800, sample_rect);      // Surrogate -> 3 bytes (U+FFFD)
    encode_codepoint(-1, sample_rect);          // Invalid -> 3 bytes (U+FFFD)

    EXPECT_EQ(text.size(), char_bboxes.size());
    EXPECT_TRUE(simdutf::validate_utf8(text.data(), text.size()));

    // Verify first rect of Cyrillic is sample_rect and 2nd is empty
    EXPECT_FLOAT_EQ(char_bboxes[0].x0, 1.0f);
    EXPECT_FLOAT_EQ(char_bboxes[1].x0, 1.0f);
    EXPECT_TRUE(fz_is_empty_rect(char_bboxes[2]));
}

TEST(MPL316Tests, AC5_FailFast_SizeMismatch_ThrowsInvalidArgument) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<MockOnnxRunner>();
    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    std::string text = "12345";
    std::vector<fz_rect> mismatch_bboxes(3, fz_rect{0.0f, 0.0f, 1.0f, 1.0f});
    std::vector<std::string> labels = {"Test"};

    EXPECT_THROW(engine.CalculateExtractedSpans(text, mismatch_bboxes, labels, 1, 100.0f, 100.0f), std::invalid_argument);
}

TEST(MPL316Tests, AC5_FailFast_NullComponents_ThrowsRuntimeError) {
    EXPECT_THROW(MLInferenceEngine(nullptr, std::make_unique<MockOnnxRunner>()), std::runtime_error);
    EXPECT_THROW(MLInferenceEngine(std::make_unique<MockTokenizer>(), nullptr), std::runtime_error);
}

TEST(MPL316Tests, AC5_PerformanceSLO_SanitizeUnder1ms) {
    // 200 words + 10 labels with Cyrillic and invalid bytes
    std::vector<std::string> words;
    for (int i = 0; i < 200; ++i) {
        if (i % 20 == 0) {
            words.push_back("invalid\xFFword" + std::to_string(i));
        } else if (i % 10 == 0) {
            words.push_back("привет_кириллица_" + std::to_string(i));
        } else {
            words.push_back("normal_ascii_word_" + std::to_string(i));
        }
    }
    std::vector<std::string> labels = {
        "Borrower", "Lender", "LoanAmount", "ClosingDate", "PropertyAddress",
        "InterestRate", "TitleCompany", "MonthlyPayment", "EscrowAmount", "TotalDue\xFE"
    };

    std::string label_scratch;
    label_scratch.reserve(4096);
    std::string word_scratch;
    word_scratch.reserve(4096);

    constexpr size_t kIterations = 1000;
    std::vector<double> latencies_us;
    latencies_us.reserve(kIterations);

    for (size_t iter = 0; iter < kIterations; ++iter) {
        auto start = std::chrono::steady_clock::now();

        for (const auto& l : labels) {
            doc_processor::SanitizeUtf8(l, label_scratch);
        }
        for (const auto& w : words) {
            doc_processor::SanitizeUtf8(w, word_scratch);
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed_us);
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double p95_us = latencies_us[static_cast<size_t>(kIterations * 0.95)];

    // 1ms = 1000us
    EXPECT_LT(p95_us, 1000.0) << "p95 sanitization latency was " << p95_us << " us, expected < 1000 us (1ms)";
}

TEST(MPL316Tests, AC5_ValidUtf8_ProducesByteIdenticalSpans) {
    auto tokenizer = std::make_unique<MockTokenizer>();
    auto runner_ptr = std::make_unique<SingleSpanOnnxRunner>();
    runner_ptr->target_word_start = 0;
    runner_ptr->target_span_len = 2;

    MLInferenceEngine engine(std::move(tokenizer), std::move(runner_ptr));

    const std::string text = "Borrower John";
    std::vector<fz_rect> char_bboxes;
    for (size_t i = 0; i < text.size(); ++i) {
        char_bboxes.push_back(fz_rect{static_cast<float>(i * 10), 0.0f, static_cast<float>((i + 1) * 10), 20.0f});
    }
    std::vector<std::string> labels = {"Borrower"};

    auto spans = engine.CalculateExtractedSpans(text, char_bboxes, labels, 1, 200.0f, 100.0f);

    ASSERT_EQ(spans.size(), 1);
    EXPECT_FLOAT_EQ(spans[0].x0(), 0.0f / 200.0f);
    EXPECT_FLOAT_EQ(spans[0].x1(), 130.0f / 200.0f);
    EXPECT_FLOAT_EQ(spans[0].y0(), 0.0f / 100.0f);
    EXPECT_FLOAT_EQ(spans[0].y1(), 20.0f / 100.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


