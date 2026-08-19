#pragma once

#include <string>
#include <vector>
#include <span>
#include <map>
#include <algorithm>
#include <stdexcept>

#include <stdckdint.h>

#include "mupdf/fitz.h"
#include "mupdf/pdf.h"
#include "ml_worker.pb.h"

struct PageTextData {
    std::string text;
    std::vector<fz_rect> char_bboxes;
};

class MuPDFWrapper {
public:
    explicit MuPDFWrapper(const std::string& pdf_data);
    ~MuPDFWrapper();

    [[nodiscard]] int GetPageCount() const;
    [[nodiscard]] PageTextData ExtractPageText(int page_index) const;
    
    // Single-page redaction with native fz_rect coordinates
    void ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes, const float color[3] = nullptr) const;

    // Multi-page batched redaction with NormalizedRedactionBox protobuf messages
    void ApplyNormalizedRedactions(
        std::span<const ml::worker::v1::NormalizedRedactionBox> custom_boxes,
        const float color[3] = nullptr
    ) const;

    [[nodiscard]] std::string GetRedactedPdf() const;

private:
    fz_context* ctx_;
    fz_document* doc_;
};
