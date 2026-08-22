#include "SpatialLayoutAnalyzer.h"
#include <ranges>

namespace doc_processor {

NormalizedBox SpatialLayoutAnalyzer::NormalizeRect(const fz_rect& rect, float page_width, float page_height) {
    if (page_width <= 0.0f || page_height <= 0.0f) {
        throw std::invalid_argument("page dimensions must be positive for coordinate normalization");
    }

    float rx0 = std::min(rect.x0, rect.x1);
    float rx1 = std::max(rect.x0, rect.x1);
    float ry0 = std::min(rect.y0, rect.y1);
    float ry1 = std::max(rect.y0, rect.y1);

    NormalizedBox box;
    box.x0 = std::clamp(rx0 / page_width, 0.0f, 1.0f);
    box.x1 = std::clamp(rx1 / page_width, 0.0f, 1.0f);
    box.y0 = std::clamp(ry0 / page_height, 0.0f, 1.0f);
    box.y1 = std::clamp(ry1 / page_height, 0.0f, 1.0f);
    return box;
}

LayoutFeatures SpatialLayoutAnalyzer::AnalyzePageLayout(
    fz_context* ctx,
    fz_page* page,
    int page_number,
    float page_width,
    float page_height
) {
    if (ctx == nullptr || page == nullptr) {
        throw std::invalid_argument("context or page cannot be null");
    }
    if (page_width <= 0.0f || page_height <= 0.0f) {
        throw std::invalid_argument("page dimensions must be positive");
    }

    LayoutFeatures features;
    features.proto_vector.set_page_number(page_number);
    features.vertical_density_histogram.assign(10, 0.0f);

    constexpr fz_stext_options stext_options = {0};
    fz_stext_page* text_page = fz_new_stext_page_from_page(ctx, page, &stext_options);
    if (!text_page) {
        throw std::runtime_error("failed to extract structured text page from MuPDF page");
    }

    int32_t total_printable_chars = 0;
    int32_t header_chars = 0;
    int32_t body_chars = 0;
    int32_t total_lines = 0;
    int32_t header_lines = 0;

    std::array<int32_t, 10> bin_counts{};
    float header_font_sum = 0.0f;
    float max_header_font = 0.0f;

    for (fz_stext_page_block_iterator pos = fz_stext_page_block_iterator_begin(text_page);
         pos.block != nullptr;
         pos = fz_stext_page_block_iterator_next(pos)) {

        fz_stext_block* block = pos.block;
        if (block->type != FZ_STEXT_BLOCK_TEXT) {
            continue;
        }

        for (fz_stext_line* line = block->u.t.first_line; line; line = line->next) {
            total_lines++;
            bool line_in_header = false;
            std::string line_header_text;

            for (fz_stext_char* ch = line->first_char; ch; ch = ch->next) {
                // Ignore non-printable whitespace characters
                if (ch->c <= 32 || ch->c == 127) {
                    continue;
                }

                fz_rect char_rect = fz_rect_from_quad(ch->quad);
                if (fz_is_empty_rect(char_rect)) {
                    continue;
                }

                NormalizedBox norm_box = NormalizeRect(char_rect, page_width, page_height);
                float y_center = (norm_box.y0 + norm_box.y1) / 2.0f;

                int bin = std::clamp(static_cast<int>(y_center * 10.0f), 0, 9);
                bin_counts[bin]++;
                total_printable_chars++;

                if (y_center <= 0.20f) {
                    line_in_header = true;
                    header_chars++;
                    header_font_sum += ch->size;
                    if (ch->size > max_header_font) {
                        max_header_font = ch->size;
                    }
                    int code = ch->c;
                    if (code >= 0 && code <= 0x7F) {
                        line_header_text.push_back(static_cast<char>(code));
                    } else if (code >= 0x80 && code <= 0x7FF) {
                        line_header_text.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
                        line_header_text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else if (code >= 0x800 && code <= 0xFFFF) {
                        if (code >= 0xD800 && code <= 0xDFFF) {
                            line_header_text.append("\xEF\xBF\xBD", 3);
                        } else {
                            line_header_text.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                            line_header_text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            line_header_text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                    } else if (code >= 0x10000 && code <= 0x10FFFF) {
                        line_header_text.push_back(static_cast<char>(0xF0 | ((code >> 18) & 0x07)));
                        line_header_text.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
                        line_header_text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        line_header_text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        line_header_text.append("\xEF\xBF\xBD", 3);
                    }
                    features.header_bboxes.push_back(norm_box);
                } else {
                    body_chars++;
                }

            }

            if (line_in_header) {
                header_lines++;
                if (!features.header_text.empty()) {
                    features.header_text += " ";
                }
                features.header_text += line_header_text;
            }
        }
    }

    if (total_printable_chars > 0) {
        for (size_t i = 0; i < 10; ++i) {
            features.vertical_density_histogram[i] = static_cast<float>(bin_counts[i]) / static_cast<float>(total_printable_chars);
        }
        features.proto_vector.set_header_density(static_cast<float>(header_chars) / static_cast<float>(total_printable_chars));
        features.proto_vector.set_body_density(static_cast<float>(body_chars) / static_cast<float>(total_printable_chars));
    }

    features.proto_vector.set_line_count(total_lines);
    features.header_line_count = header_lines;
    features.header_char_count = header_chars;
    features.max_font_size = max_header_font;
    if (header_chars > 0) {
        features.mean_font_size = header_font_sum / static_cast<float>(header_chars);
    }

    bool has_header = (header_lines > 0 && (features.proto_vector.header_density() > 0.15f || max_header_font >= 1.2f * features.mean_font_size));
    features.proto_vector.set_has_top_header(has_header);

    fz_drop_stext_page(ctx, text_page);

    return features;
}

KeyValueMatch SpatialLayoutAnalyzer::RaycastAdjacentValue(
    std::span<const WordSpan> words,
    const fz_rect& key_box,
    RayDirection direction,
    float max_distance,
    float corridor_tolerance
) {
    if (words.empty() || fz_is_empty_rect(key_box)) {
        return {};
    }

    // Corridor raycasting filter
    std::vector<const WordSpan*> candidates;
    for (const auto& w : words) {
        if (fz_is_empty_rect(w.bbox)) continue;

        if (direction == RayDirection::HorizontalRight) {
            // Horizontal corridor: vertical overlap with key_box within corridor_tolerance
            float v_overlap_min = std::max(w.bbox.y0, key_box.y0 - corridor_tolerance);
            float v_overlap_max = std::min(w.bbox.y1, key_box.y1 + corridor_tolerance);
            if (v_overlap_min < v_overlap_max) {
                // Must be to the right of key_box
                float dx = w.bbox.x0 - key_box.x1;
                if (dx >= -5.0f && dx <= max_distance) {
                    candidates.push_back(&w);
                }
            }
        } else if (direction == RayDirection::VerticalDown) {
            // Vertical corridor: horizontal overlap with key_box within corridor_tolerance
            float h_overlap_min = std::max(w.bbox.x0, key_box.x0 - corridor_tolerance);
            float h_overlap_max = std::min(w.bbox.x1, key_box.x1 + corridor_tolerance);
            if (h_overlap_min < h_overlap_max) {
                // Must be below key_box
                float dy = w.bbox.y0 - key_box.y1;
                if (dy >= -5.0f && dy <= max_distance) {
                    candidates.push_back(&w);
                }
            }
        }
    }

    if (candidates.empty()) {
        return {};
    }

    // Sort candidates in reading/proximity order
    if (direction == RayDirection::HorizontalRight) {
        std::sort(candidates.begin(), candidates.end(), [](const WordSpan* a, const WordSpan* b) {
            return a->bbox.x0 < b->bbox.x0;
        });
    } else {
        std::sort(candidates.begin(), candidates.end(), [](const WordSpan* a, const WordSpan* b) {
            return a->bbox.y0 < b->bbox.y0;
        });
    }

    // Cluster contiguous sequence of value words
    std::vector<const WordSpan*> cluster;
    cluster.push_back(candidates[0]);

    for (size_t i = 1; i < candidates.size(); ++i) {
        const auto* prev = cluster.back();
        const auto* curr = candidates[i];

        if (direction == RayDirection::HorizontalRight) {
            float gap = curr->bbox.x0 - prev->bbox.x1;
            if (gap <= corridor_tolerance) {
                cluster.push_back(curr);
            } else {
                break;
            }
        } else {
            float gap = curr->bbox.y0 - prev->bbox.y1;
            if (gap <= corridor_tolerance) {
                cluster.push_back(curr);
            } else {
                break;
            }
        }
    }

    KeyValueMatch match;
    match.found = true;
    match.value_box = cluster[0]->bbox;
    match.value_text = cluster[0]->text;

    for (size_t i = 1; i < cluster.size(); ++i) {
        match.value_box = fz_union_rect(match.value_box, cluster[i]->bbox);
        match.value_text += " " + cluster[i]->text;
    }

    match.composite_box = fz_union_rect(key_box, match.value_box);
    return match;
}

} // namespace doc_processor
