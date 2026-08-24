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

struct SurvivingGlyphsResult {
    int count = 0;
    std::vector<fz_rect> glyph_quads;
    std::vector<std::string> sample_snippets;
};

struct PostRedactionReport {
    bool is_converged = true;
    int residual_value_count = 0;
    std::vector<std::string> residual_samples;
    int pass_count = 1;
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

    // Applies redactions given page-coordinate bounding boxes (e.g. from ExtractPageText char_bboxes).
    void ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes, const float color[3] = nullptr) const;

    // Applies redactions given normalized visual [0.0, 1.0] top-left-origin boxes,
    // handling rotation, MediaBox/CropBox visible bounds, clamping, and single-pass per-page application.
    void ApplyNormalizedRedactions(
        std::span<const ml::worker::v1::NormalizedRedactionBox> custom_boxes,
        const float color[3] = nullptr
    ) const;


    [[nodiscard]] std::string GetRedactedPdf() const;

    // Post-redaction verification — dual-oracle (Layer 1 structural + Layer 2 byte scan)
    [[nodiscard]] SurvivingGlyphsResult CollectIntersectingGlyphs(const std::string& redacted_pdf_bytes, const std::vector<fz_rect>& boxes, int page_index) const;
    [[nodiscard]] int CountIntersectingGlyphs(const std::string& redacted_pdf_bytes, const std::vector<fz_rect>& boxes, int page_index) const;
    [[nodiscard]] int CountPlaintextOccurrences(const std::string& redacted_pdf_bytes, const std::vector<std::string>& plaintexts) const;

    PostRedactionReport VerifyPostRedaction(std::string& redacted_pdf_bytes,
                                           const std::map<int, std::vector<fz_rect>>& per_page_boxes,
                                           const std::vector<std::string>& target_plaintexts,
                                           const std::string& job_id,
                                           const std::string& traceparent = "",
                                           int pass_count = 1) const;

    void VerifyPostRedaction(const std::string& redacted_pdf_bytes,
                             const std::map<int, std::vector<fz_rect>>& per_page_boxes,
                             const std::vector<std::string>& target_plaintexts,
                             const std::string& job_id,
                             const std::string& traceparent = "") const;

private:
    fz_context* ctx() const noexcept { return pooled_ctx_.get(); }
    void ensure_valid() const;

    static void MergePdfDicts(fz_context* ctx, pdf_obj* dest_dict, pdf_obj* src_dict);
    static void InlineFormXObjects(fz_context* ctx, pdf_document* doc, pdf_page* page);

    std::shared_ptr<FzContextPool> pool_;
    FzContextPool::PooledContext pooled_ctx_;
    FzDocumentPtr doc_;
};
