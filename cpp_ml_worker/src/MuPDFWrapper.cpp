#include "MuPDFWrapper.h"
#include <stdexcept>

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

PageTextData MuPDFWrapper::ExtractPageText(int page_index) const {
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

void MuPDFWrapper::ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes) const {
    if (bboxes.empty()) return;

    fz_try(ctx_) {
        fz_page* page = fz_load_page(ctx_, doc_, page_index);
        if (!page) {
            fz_throw(ctx_, FZ_ERROR_GENERIC, "Failed to load page for redaction");
        }

        if (pdf_page* ppage = pdf_page_from_fz_page(ctx_, page)) {
            float red_color[3] = {1.0f, 0.0f, 0.0f};
            for (const auto& rect : bboxes) {
                pdf_annot* annot = pdf_create_annot(ctx_, ppage, PDF_ANNOT_REDACT);
                pdf_set_annot_rect(ctx_, annot, rect);
                pdf_set_annot_color(ctx_, annot, 3, red_color);
            }
            pdf_redact_page(ctx_, pdf_document_from_fz_document(ctx_, doc_), ppage, nullptr);
        }

        fz_drop_page(ctx_, page);
    } fz_catch(ctx_) {
        throw std::runtime_error(std::string("ApplyRedactions error: ") + fz_caught_message(ctx_));
    }
}

std::string MuPDFWrapper::GetRedactedPdf() const {
    std::string redacted_pdf_data;
    fz_try(ctx_) {
        fz_buffer* buf = fz_new_buffer(ctx_, 1024);
        fz_output* out = fz_new_output_with_buffer(ctx_, buf);
        
        pdf_write_options write_opts = {0};
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
