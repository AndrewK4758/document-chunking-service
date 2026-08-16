#include "SpanGenerator.h"
#include <algorithm>
#include <stdexcept>

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
    result.span_idx.reserve(est_spans * 2);
    result.span_mask.reserve(est_spans);
    result.num_spans = 0;

    for (size_t start = 0; start < num_words; ++start) {
        size_t limit = std::min(num_words, start + max_span_len);
        for (size_t end = start; end < limit; ++end) {
            result.span_idx.push_back(static_cast<int64_t>(start));
            result.span_idx.push_back(static_cast<int64_t>(end));
            result.span_mask.push_back(1);
            result.num_spans++;
        }
    }

    return result;
}

} // namespace doc_processor
