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
    EXPECT_EQ(spans.span_idx.size(), 174 * 2);
    EXPECT_EQ(spans.span_mask.size(), 174);
    
    // Check first span
    EXPECT_EQ(spans.span_idx[0], 0);
    EXPECT_EQ(spans.span_idx[1], 0);
    
    // Check second span
    EXPECT_EQ(spans.span_idx[2], 0);
    EXPECT_EQ(spans.span_idx[3], 1);
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
    EXPECT_EQ(spans.span_idx.size(), 30);
    EXPECT_EQ(spans.span_mask.size(), 15);
    
    // All masks should be 1
    for (int64_t mask : spans.span_mask) {
        EXPECT_EQ(mask, 1);
    }
}

TEST(SpanGeneratorTest, TestZeroLength) {
    EXPECT_THROW((void)SpanGenerator::Generate(0, 12), std::runtime_error);
    EXPECT_THROW((void)SpanGenerator::Generate(5, 0), std::runtime_error);
}
