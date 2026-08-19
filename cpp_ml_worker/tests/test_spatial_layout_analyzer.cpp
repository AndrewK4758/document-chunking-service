#include <gtest/gtest.h>
#include <stdckdint.h>
#include "SpatialLayoutAnalyzer.h"
#include "mupdf/fitz.h"
#include "mupdf/pdf.h"

using namespace doc_processor;

TEST(SpatialLayoutAnalyzerTest, NormalizeRect_StandardAndClamped) {
    fz_rect standard_rect = {100.0f, 200.0f, 300.0f, 400.0f};
    auto [x0, y0, x1, y1] = SpatialLayoutAnalyzer::NormalizeRect(standard_rect, 1000.0f, 1000.0f);

    EXPECT_FLOAT_EQ(x0, 0.1f);
    EXPECT_FLOAT_EQ(y0, 0.2f);
    EXPECT_FLOAT_EQ(x1, 0.3f);
    EXPECT_FLOAT_EQ(y1, 0.4f);

    // Out of bounds coordinates must be clamped strictly to [0.0, 1.0]
    fz_rect out_of_bounds_rect = {-50.0f, -100.0f, 1500.0f, 2000.0f};
    NormalizedBox clamped_box = SpatialLayoutAnalyzer::NormalizeRect(out_of_bounds_rect, 1000.0f, 1000.0f);

    EXPECT_FLOAT_EQ(clamped_box.x0, 0.0f);
    EXPECT_FLOAT_EQ(clamped_box.y0, 0.0f);
    EXPECT_FLOAT_EQ(clamped_box.x1, 1.0f);
    EXPECT_FLOAT_EQ(clamped_box.y1, 1.0f);

    // Inverted coordinate values (x0 > x1 or y0 > y1)
    fz_rect inverted_rect = {300.0f, 400.0f, 100.0f, 200.0f};
    NormalizedBox inv_box = SpatialLayoutAnalyzer::NormalizeRect(inverted_rect, 1000.0f, 1000.0f);

    EXPECT_FLOAT_EQ(inv_box.x0, 0.1f);
    EXPECT_FLOAT_EQ(inv_box.y0, 0.2f);
    EXPECT_FLOAT_EQ(inv_box.x1, 0.3f);
    EXPECT_FLOAT_EQ(inv_box.y1, 0.4f);
}

TEST(SpatialLayoutAnalyzerTest, NormalizeRect_InvalidDimensionsThrows) {
    fz_rect rect = {0.0f, 0.0f, 100.0f, 100.0f};
    EXPECT_THROW((void)SpatialLayoutAnalyzer::NormalizeRect(rect, 0.0f, 500.0f), std::invalid_argument);
    EXPECT_THROW((void)SpatialLayoutAnalyzer::NormalizeRect(rect, 500.0f, -10.0f), std::invalid_argument);
}

TEST(SpatialLayoutAnalyzerTest, AnalyzePageLayout_NullInputThrows) {
    fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);
    ASSERT_NE(ctx, nullptr);

    EXPECT_THROW((void)SpatialLayoutAnalyzer::AnalyzePageLayout(nullptr, nullptr, 1, 612.0f, 792.0f), std::invalid_argument);
    EXPECT_THROW((void)SpatialLayoutAnalyzer::AnalyzePageLayout(ctx, nullptr, 1, 612.0f, 792.0f), std::invalid_argument);
    
    fz_drop_context(ctx);
}

TEST(SpatialLayoutAnalyzerTest, AnalyzePageLayout_EmptyPageHandling) {
    fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);
    ASSERT_NE(ctx, nullptr);
    fz_register_document_handlers(ctx);

    const std::string minimal_blank_pdf = 
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj\n"
        "xref\n"
        "0 4\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "trailer << /Size 4 /Root 1 0 R >>\n"
        "startxref\n"
        "190\n"
        "%%EOF\n";

    fz_stream* stream = fz_open_memory(ctx, reinterpret_cast<const unsigned char*>(minimal_blank_pdf.data()), minimal_blank_pdf.size());
    ASSERT_NE(stream, nullptr);
    fz_document* doc = fz_open_document_with_stream(ctx, "pdf", stream);
    fz_drop_stream(ctx, stream);
    ASSERT_NE(doc, nullptr);

    fz_page* page = fz_load_page(ctx, doc, 0);
    ASSERT_NE(page, nullptr);

    LayoutFeatures features = SpatialLayoutAnalyzer::AnalyzePageLayout(
        ctx,
        page,
        1,
        612.0f,
        792.0f
    );

    EXPECT_EQ(features.proto_vector.page_number(), 1);
    EXPECT_FLOAT_EQ(features.proto_vector.header_density(), 0.0f);
    EXPECT_FLOAT_EQ(features.proto_vector.body_density(), 0.0f);
    EXPECT_EQ(features.proto_vector.line_count(), 0);
    EXPECT_FALSE(features.proto_vector.has_top_header());
    EXPECT_EQ(features.header_line_count, 0);
    EXPECT_EQ(features.header_char_count, 0);
    EXPECT_EQ(features.vertical_density_histogram.size(), 10);

    for (float bin_val : features.vertical_density_histogram) {
        EXPECT_FLOAT_EQ(bin_val, 0.0f);
    }

    fz_drop_page(ctx, page);
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
}

TEST(SpatialLayoutAnalyzerTest, AnalyzePageLayout_WithTextContent) {
    fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);
    ASSERT_NE(ctx, nullptr);
    fz_register_document_handlers(ctx);

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
        "(TOP HEADER INVOICE) Tj\n"
        "/F1 12 Tf\n"
        "0 -200 Td\n"
        "(Body content line) Tj\n"
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

    fz_stream* stream = fz_open_memory(ctx, reinterpret_cast<const unsigned char*>(text_pdf.data()), text_pdf.size());
    ASSERT_NE(stream, nullptr);
    fz_document* doc = fz_open_document_with_stream(ctx, "pdf", stream);
    fz_drop_stream(ctx, stream);
    ASSERT_NE(doc, nullptr);

    fz_page* page = fz_load_page(ctx, doc, 0);
    ASSERT_NE(page, nullptr);

    LayoutFeatures features = SpatialLayoutAnalyzer::AnalyzePageLayout(
        ctx,
        page,
        1,
        612.0f,
        792.0f
    );

    EXPECT_EQ(features.proto_vector.page_number(), 1);
    EXPECT_GT(features.proto_vector.line_count(), 0);
    EXPECT_TRUE(features.proto_vector.has_top_header());
    EXPECT_GE(features.header_line_count, 1);
    EXPECT_GT(features.header_char_count, 0);
    EXPECT_FLOAT_EQ(features.max_font_size, 24.0f);
    EXPECT_GT(features.proto_vector.header_density(), 0.0f);

    float sum_histogram = 0.0f;
    for (float bin_val : features.vertical_density_histogram) {
        sum_histogram += bin_val;
    }
    EXPECT_NEAR(sum_histogram, 1.0f, 0.001f);

    fz_drop_page(ctx, page);
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
}
