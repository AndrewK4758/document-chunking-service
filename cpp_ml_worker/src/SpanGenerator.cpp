#include "SpanGenerator.h"
#include <algorithm>
#include <stdexcept>
#include <ranges>

namespace doc_processor {

GeneratedSpans SpanGenerator::Generate(size_t num_words, size_t max_span_len) {
    if (num_words == 0) {
        throw std::runtime_error("Sequence length must be > 0");
    }
    if (max_span_len == 0) {
        throw std::runtime_error("max_span_len must be > 0");
    }

    size_t est_spans = num_words * max_span_len; 
    GeneratedSpans result;
    result.span_idx_storage.resize(est_spans * 2);
    result.span_mask_storage.resize(est_spans);
    result.num_spans = est_spans; // Temporary for mut view to work

    auto span_idx_view = result.span_idx_mut();
    auto span_mask_view = result.span_mask_mut();

    size_t idx = 0;
    
    // Using C++23 ranges
    for (size_t start : std::views::iota(0uz, num_words)) {
        size_t limit = std::min(num_words, start + max_span_len);
        for (size_t end : std::views::iota(start, limit)) {
            span_idx_view[idx, 0] = static_cast<int64_t>(start);
            span_idx_view[idx, 1] = static_cast<int64_t>(end);
            span_mask_view[idx] = 1;
            idx++;
        }
    }

    result.num_spans = idx;
    result.span_idx_storage.resize(idx * 2);
    result.span_mask_storage.resize(idx);

    return result;
}

} // namespace doc_processor
