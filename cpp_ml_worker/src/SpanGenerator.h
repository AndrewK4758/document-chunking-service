#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <mdspan>
#include <span>

namespace doc_processor {

struct GeneratedSpans {
    std::vector<int64_t> span_idx_storage; 
    std::vector<int64_t> span_mask_storage; 
    size_t num_spans{0};

    [[nodiscard]] auto span_idx() const {
        return std::mdspan<const int64_t, std::extents<size_t, std::dynamic_extent, 2>>(span_idx_storage.data(), num_spans, 2);
    }
    auto span_idx_mut() {
        return std::mdspan<int64_t, std::extents<size_t, std::dynamic_extent, 2>>(span_idx_storage.data(), num_spans, 2);
    }
    [[nodiscard]] auto span_mask() const {
        return std::span<const int64_t>(span_mask_storage.data(), num_spans);
    }
    auto span_mask_mut() {
        return std::span<int64_t>(span_mask_storage.data(), num_spans);
    }
};

class SpanGenerator {
public:
    [[nodiscard]] static GeneratedSpans Generate(size_t num_words, size_t max_span_len = 12);
};

} // namespace doc_processor
