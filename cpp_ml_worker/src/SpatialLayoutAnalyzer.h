#pragma once

#include <string>
#include <vector>
#include <span>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

#include <stdckdint.h>

#include "mupdf/fitz.h"
#include "ml_worker.pb.h"

namespace doc_processor {

struct NormalizedBox {
    float x0{0.0f};
    float y0{0.0f};
    float x1{0.0f};
    float y1{0.0f};
};

struct LayoutFeatures {
    ml::worker::v1::SpatialFeatureVector proto_vector;
    std::vector<float> vertical_density_histogram; // 10 bins summing to 1.0
    float max_font_size{0.0f};
    float mean_font_size{0.0f};
    int32_t header_line_count{0};
    int32_t header_char_count{0};
    std::string header_text;
    std::vector<NormalizedBox> header_bboxes;
};

class SpatialLayoutAnalyzer {
public:
    SpatialLayoutAnalyzer() = default;
    ~SpatialLayoutAnalyzer() = default;

    [[nodiscard]] static NormalizedBox NormalizeRect(const fz_rect& rect, float page_width, float page_height);

    [[nodiscard]] static LayoutFeatures AnalyzePageLayout(
        fz_context* ctx,
        fz_page* page,
        int page_number,
        float page_width,
        float page_height
    );
};

} // namespace doc_processor
