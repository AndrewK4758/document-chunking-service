#pragma once

#include <string>
#include <vector>

extern "C" {
#include "mupdf/fitz.h"
#include "mupdf/pdf.h"
}

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
    void ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes) const;
    [[nodiscard]] std::string GetRedactedPdf() const;

private:
    fz_context* ctx_;
    fz_document* doc_;
};
