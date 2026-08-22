#include "MuPDFWrapper.h"
#include <stdexcept>
#include <algorithm>
#include <map>

MuPDFWrapper::MuPDFWrapper(const std::string& pdf_data) {
    ctx_ = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);
    if (!ctx_) {
        throw std::runtime_error("Failed to create MuPDF context");
    }
    fz_register_document_handlers(ctx_);

    fz_try(ctx_) {
        fz_stream* stream = fz_open_memory(ctx_, reinterpret_cast<const unsigned char*>(pdf_data.data()), pdf_data.size());
        doc_ = fz_open_document_with_stream(ctx_, "pdf", stream);
        fz_drop_stream(ctx_, stream);
    } fz_catch(ctx_) {
        std::string err_msg = fz_caught_message(ctx_);
        fz_drop_context(ctx_);
        throw std::runtime_error("MuPDF Error: " + err_msg);
    }
}

MuPDFWrapper::~MuPDFWrapper() {
    if (doc_) fz_drop_document(ctx_, doc_);
    if (ctx_) fz_drop_context(ctx_);
}

int MuPDFWrapper::GetPageCount() const {
    int count = 0;
    fz_try(ctx_) {
        count = fz_count_pages(ctx_, doc_);
    } fz_catch(ctx_) {
        throw std::runtime_error(std::string("GetPageCount error: ") + fz_caught_message(ctx_));
    }
    return count;
}

int MuPDFWrapper::GetPageRotation(int page_index) const {
    int page_count = GetPageCount();
    if (page_index < 0 || page_index >= page_count) {
        throw std::out_of_range("page_index out of range");
    }
    int rotation = 0;
    fz_try(ctx_) {
        pdf_document* pdoc = pdf_document_from_fz_document(ctx_, doc_);
        if (pdoc) {
            pdf_obj* page_obj = pdf_lookup_page_obj(ctx_, pdoc, page_index);
            if (page_obj) {
                rotation = pdf_dict_get_int(ctx_, page_obj, PDF_NAME(Rotate));
            }
        }
    } fz_catch(ctx_) {
        throw std::runtime_error(std::string("GetPageRotation error: ") + fz_caught_message(ctx_));
    }
    rotation = ((rotation % 360) + 360) % 360;
    return rotation;
}

PageTextData MuPDFWrapper::ExtractPageText(int page_index) const {
    int page_count = GetPageCount();
    if (page_index < 0 || page_index >= page_count) {
        throw std::out_of_range("page_index out of range");
    }

    PageTextData data;
    fz_try(ctx_) {
        fz_page* page = fz_load_page(ctx_, doc_, page_index);
        if (!page) {
            fz_throw(ctx_, FZ_ERROR_GENERIC, "Failed to load page");
        }

        constexpr fz_stext_options stext_options = {0};
        if (fz_stext_page* text_page = fz_new_stext_page_from_page(ctx_, page, &stext_options)) {
            for (fz_stext_page_block_iterator pos = fz_stext_page_block_iterator_begin(text_page);
                 pos.block != nullptr;
                 pos = fz_stext_page_block_iterator_next(pos)) {
                
                fz_stext_block* block = pos.block;
                if (block->type == FZ_STEXT_BLOCK_TEXT) {
                    for (fz_stext_line* line = block->u.t.first_line; line; line = line->next) {
                        for (fz_stext_char* ch = line->first_char; ch; ch = ch->next) {
                            data.text += static_cast<char>(ch->c);
                            data.char_bboxes.push_back(fz_rect_from_quad(ch->quad));
                        }
                        data.text += ' ';
                        data.char_bboxes.push_back(fz_empty_rect);
                    }
                    data.text += '\n';
                    data.char_bboxes.push_back(fz_empty_rect);
                }
            }
            fz_drop_stext_page(ctx_, text_page);
        }
        fz_drop_page(ctx_, page);
    } fz_catch(ctx_) {
        throw std::runtime_error(std::string("ExtractPageText error: ") + fz_caught_message(ctx_));
    }
    return data;
}

void MuPDFWrapper::ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes, const float color[3]) const {
    int page_count = GetPageCount();
    if (page_index < 0 || page_index >= page_count) {
        throw std::out_of_range("page_index out of range for redaction");
    }
    if (bboxes.empty()) return;

    constexpr float default_black[3] = {0.0f, 0.0f, 0.0f};
    const float* fill_color = color ? color : default_black;

    fz_try(ctx_) {
        fz_page* page = fz_load_page(ctx_, doc_, page_index);
        if (!page) {
            fz_throw(ctx_, FZ_ERROR_GENERIC, "Failed to load page for redaction");
        }

        if (pdf_page* ppage = pdf_page_from_fz_page(ctx_, page)) {
            int valid_annots = 0;
            for (const auto& rect : bboxes) {
                if (fz_is_empty_rect(rect) || rect.x0 >= rect.x1 || rect.y0 >= rect.y1) {
                    continue;
                }
                pdf_annot* annot = pdf_create_annot(ctx_, ppage, PDF_ANNOT_REDACT);
                pdf_set_annot_rect(ctx_, annot, rect);
                pdf_set_annot_color(ctx_, annot, 3, fill_color);
                valid_annots++;
            }
            if (valid_annots > 0) {
                pdf_redact_page(ctx_, pdf_document_from_fz_document(ctx_, doc_), ppage, nullptr);
            }
        }

        fz_drop_page(ctx_, page);
    } fz_catch(ctx_) {
        throw std::runtime_error(std::string("ApplyRedactions error: ") + fz_caught_message(ctx_));
    }
}

void MuPDFWrapper::ApplyNormalizedRedactions(
    std::span<const ml::worker::v1::NormalizedRedactionBox> custom_boxes,
    const float color[3]
) const {
    if (custom_boxes.empty()) return;

    int total_pages = GetPageCount();

    // Group boxes by page index (1-based page_number maps to 0-based page_index)
    std::map<int, std::vector<ml::worker::v1::NormalizedRedactionBox>> page_boxes_map;
    for (const auto& box : custom_boxes) {
        int page_idx = box.page_number() > 0 ? (box.page_number() - 1) : 0;
        if (page_idx < 0 || page_idx >= total_pages) {
            throw std::out_of_range("page_number in NormalizedRedactionBox is out of range");
        }
        page_boxes_map[page_idx].push_back(box);
    }

    constexpr float default_black[3] = {0.0f, 0.0f, 0.0f};
    const float* fill_color = color ? color : default_black;

    for (const auto& [page_idx, boxes] : page_boxes_map) {
        fz_try(ctx_) {
            fz_page* page = fz_load_page(ctx_, doc_, page_idx);
            if (!page) {
                fz_throw(ctx_, FZ_ERROR_GENERIC, "Failed to load page for normalized redaction");
            }

            if (pdf_page* ppage = pdf_page_from_fz_page(ctx_, page)) {
                fz_rect page_bounds = fz_bound_page(ctx_, page);
                float page_w = page_bounds.x1 - page_bounds.x0;
                float page_h = page_bounds.y1 - page_bounds.y0;

                int valid_annots = 0;
                for (const auto& box : boxes) {
                    float rx0 = std::min(box.x0(), box.x1());
                    float rx1 = std::max(box.x0(), box.x1());
                    float ry0 = std::min(box.y0(), box.y1());
                    float ry1 = std::max(box.y0(), box.y1());

                    // Skip degenerate or collapsed boxes
                    if (rx0 >= rx1 || ry0 >= ry1) {
                        continue;
                    }

                    fz_rect native_rect;
                    native_rect.x0 = page_bounds.x0 + std::clamp(rx0, 0.0f, 1.0f) * page_w;
                    native_rect.x1 = page_bounds.x0 + std::clamp(rx1, 0.0f, 1.0f) * page_w;
                    native_rect.y0 = page_bounds.y0 + std::clamp(ry0, 0.0f, 1.0f) * page_h;
                    native_rect.y1 = page_bounds.y0 + std::clamp(ry1, 0.0f, 1.0f) * page_h;

                    pdf_annot* annot = pdf_create_annot(ctx_, ppage, PDF_ANNOT_REDACT);
                    pdf_set_annot_rect(ctx_, annot, native_rect);
                    pdf_set_annot_color(ctx_, annot, 3, fill_color);
                    valid_annots++;
                }

                if (valid_annots > 0) {
                    pdf_redact_page(ctx_, pdf_document_from_fz_document(ctx_, doc_), ppage, nullptr);
                }
            }

            fz_drop_page(ctx_, page);
        } fz_catch(ctx_) {
            throw std::runtime_error(std::string("ApplyNormalizedRedactions error: ") + fz_caught_message(ctx_));
        }
    }
}

std::string MuPDFWrapper::GetRedactedPdf() const {
    std::string redacted_pdf_data;
    fz_try(ctx_) {
        fz_buffer* buf = fz_new_buffer(ctx_, 1024);
        fz_output* out = fz_new_output_with_buffer(ctx_, buf);
        
        pdf_write_options write_opts = {0};
        write_opts.do_clean = 1;
        write_opts.do_sanitize = 1;
        write_opts.do_garbage = 3;
        write_opts.do_compress = 1;
        pdf_write_document(ctx_, pdf_document_from_fz_document(ctx_, doc_), out, &write_opts);
        fz_close_output(ctx_, out);
        fz_drop_output(ctx_, out);

        redacted_pdf_data = std::string(reinterpret_cast<const char*>(buf->data), buf->len);
        fz_drop_buffer(ctx_, buf);
    } fz_catch(ctx_) {
        throw std::runtime_error(std::string("GetRedactedPdf error: ") + fz_caught_message(ctx_));
    }
    return redacted_pdf_data;
}
