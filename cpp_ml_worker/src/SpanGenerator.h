#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace doc_processor {

struct GeneratedSpans {
    std::vector<int64_t> span_idx; // Shape: [1, num_spans, 2] -> flattened
    std::vector<int64_t> span_mask; // Shape: [1, num_spans]
    size_t num_spans{0};
};

class SpanGenerator {
public:
    // Generates sliding window spans up to max_span_len.
    // Throws std::runtime_error if num_words == 0.
    [[nodiscard]] static GeneratedSpans Generate(size_t num_words, size_t max_span_len = 12);
};

} // namespace doc_processor
