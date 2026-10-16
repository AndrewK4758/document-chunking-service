#include <gtest/gtest.h>
#include "SpanGenerator.h"

using namespace doc_processor;

TEST(SpanGeneratorTest, TestStandardSequence) {
    size_t num_words = 20;
    size_t max_span_len = 12;
    
    auto spans = SpanGenerator::Generate(num_words, max_span_len);
    
    EXPECT_EQ(spans.num_spans, 240);
    EXPECT_EQ(spans.span_idx_storage.size(), 480);
    EXPECT_EQ(spans.span_mask_storage.size(), 240);
    
    auto span_idx = spans.span_idx();
    EXPECT_EQ((span_idx[0, 0]), 0);
    EXPECT_EQ((span_idx[0, 1]), 0);
    
    EXPECT_EQ((span_idx[1, 0]), 0);
    EXPECT_EQ((span_idx[1, 1]), 1);
}

TEST(SpanGeneratorTest, TestShortSequence) {
    size_t num_words = 5;
    size_t max_span_len = 12;
    
    auto spans = SpanGenerator::Generate(num_words, max_span_len);
    
    EXPECT_EQ(spans.num_spans, 60);
    EXPECT_EQ(spans.span_idx_storage.size(), 120);
    EXPECT_EQ(spans.span_mask_storage.size(), 60);
    
    auto span_mask = spans.span_mask();
    size_t valid_spans = 0;
    for (size_t i = 0; i < spans.num_spans; ++i) {
        if (span_mask[i] == 1) valid_spans++;
    }
    EXPECT_EQ(valid_spans, 15);
}

TEST(SpanGeneratorTest, TestZeroLength) {
    EXPECT_THROW((void)SpanGenerator::Generate(0, 12), std::runtime_error);
    EXPECT_THROW((void)SpanGenerator::Generate(5, 0), std::runtime_error);
}
