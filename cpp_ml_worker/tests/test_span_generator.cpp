#include <gtest/gtest.h>
#include "SpanGenerator.h"

using namespace doc_processor;

TEST(SpanGeneratorTest, TestStandardSequence) {
    size_t num_words = 20;
    size_t max_span_len = 12;
    
    auto spans = SpanGenerator::Generate(num_words, max_span_len);
    
    // We can calculate expected number of spans.
    // For start from 0 to 19 (num_words = 20)
    // limit is min(20, start + 12)
    // For 0..8 (9 elements): limit = start + 12. Num spans per start = 12. 9 * 12 = 108
    // For 9..19 (11 elements): limit = 20. Num spans per start = 20 - start.
    // sum(11, 10, ..., 1) = 11 * 12 / 2 = 66
    // Total = 108 + 66 = 174
    EXPECT_EQ(spans.num_spans, 174);
    EXPECT_EQ(spans.span_idx_storage.size(), 174 * 2);
    EXPECT_EQ(spans.span_mask_storage.size(), 174);
    
    auto span_idx = spans.span_idx();
    // Check first span
    EXPECT_EQ((span_idx[0, 0]), 0);
    EXPECT_EQ((span_idx[0, 1]), 0);
    
    // Check second span
    EXPECT_EQ((span_idx[1, 0]), 0);
    EXPECT_EQ((span_idx[1, 1]), 1);
}

TEST(SpanGeneratorTest, TestShortSequence) {
    size_t num_words = 5;
    size_t max_span_len = 12;
    
    auto spans = SpanGenerator::Generate(num_words, max_span_len);
    
    // Total spans for 5 words is sum of spans from length 1 to 5
    // start 0: end 0..4 (5 spans)
    // start 1: end 1..4 (4 spans)
    // Total = 5+4+3+2+1 = 15
    EXPECT_EQ(spans.num_spans, 15);
    EXPECT_EQ(spans.span_idx_storage.size(), 30);
    EXPECT_EQ(spans.span_mask_storage.size(), 15);
    
    // All masks should be 1
    auto span_mask = spans.span_mask();
    for (size_t i = 0; i < spans.num_spans; ++i) {
        EXPECT_EQ(span_mask[i], 1);
    }
}

TEST(SpanGeneratorTest, TestZeroLength) {
    EXPECT_THROW((void)SpanGenerator::Generate(0, 12), std::runtime_error);
    EXPECT_THROW((void)SpanGenerator::Generate(5, 0), std::runtime_error);
}
