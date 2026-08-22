#pragma once

#include <string>
#include <vector>
#include <span>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <memory>

#include <stdckdint.h>

#include "mupdf/fitz.h"
#include "mupdf/pdf.h"
#include "ml_worker.pb.h"
#include "FzContextPool.h"
#include "FzDeleters.h"

struct PageTextData {
    std::string text;
    std::vector<fz_rect> char_bboxes;
};

class MuPDFWrapper {
public:
    explicit MuPDFWrapper(const std::string& pdf_data,
                          std::shared_ptr<FzContextPool> pool);
    explicit MuPDFWrapper(const std::string& pdf_data);
    ~MuPDFWrapper();

    MuPDFWrapper(const MuPDFWrapper&) = delete;
    MuPDFWrapper& operator=(const MuPDFWrapper&) = delete;
    MuPDFWrapper(MuPDFWrapper&&) noexcept;
    MuPDFWrapper& operator=(MuPDFWrapper&&) noexcept;

    [[nodiscard]] int GetPageCount() const;
    [[nodiscard]] int GetPageRotation(int page_index) const;
    [[nodiscard]] PageTextData ExtractPageText(int page_index) const;

    [[nodiscard]] std::vector<unsigned char> RenderPagePixmap(int page_index, int dpi = 300) const;

    void ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes, const float color[3] = nullptr) const;
    void ApplyNormalizedRedactions(
        std::span<const ml::worker::v1::NormalizedRedactionBox> custom_boxes,
        const float color[3] = nullptr
    ) const;

    [[nodiscard]] std::string GetRedactedPdf() const;

private:
    fz_context* ctx() const noexcept { return pooled_ctx_.get(); }
    void ensure_valid() const;

    std::shared_ptr<FzContextPool> pool_;
    FzContextPool::PooledContext pooled_ctx_;
    FzDocumentPtr doc_;
};
