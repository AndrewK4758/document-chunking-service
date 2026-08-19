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
    
    // Using C++26 ranges
    for (size_t start : std::views::iota(0uz, num_words)) {
        for (size_t end_offset : std::views::iota(0uz, max_span_len)) {
            size_t end = start + end_offset;
            if (end < num_words) {
                span_idx_view[idx, 0] = static_cast<int64_t>(start);
                span_idx_view[idx, 1] = static_cast<int64_t>(end);
                span_mask_view[idx] = 1;
            } else {
                // Pad the remaining spans for this start token
                span_idx_view[idx, 0] = 0;
                span_idx_view[idx, 1] = 0;
                span_mask_view[idx] = 0;
            }
            idx++;
        }
    }

    // Do NOT resize down, ONNX model expects EXACTLY num_words * max_span_len
    result.num_spans = est_spans;

    return result;
}

} // namespace doc_processor
