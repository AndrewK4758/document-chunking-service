#pragma once

#include <string>
#include <vector>
#include "MLInferenceEngine.h"

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
    MuPDFWrapper(const std::string& pdf_data);
    ~MuPDFWrapper();

    int GetPageCount() const;
    PageTextData ExtractPageText(int page_index) const;
    void ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes) const;
    std::string GetRedactedPdf() const;

private:
    fz_context* ctx_;
    fz_document* doc_;
};
