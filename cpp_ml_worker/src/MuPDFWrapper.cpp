#include "MuPDFWrapper.h"
#include <stdexcept>
#include <algorithm>
#include <map>
#include <spdlog/spdlog.h>

void MuPDFWrapper::ensure_valid() const {
    if (!pooled_ctx_ || !doc_) {
        throw std::runtime_error("MuPDFWrapper moved-from or invalid");
    }
    if (!ctx()) {
        throw std::runtime_error("MuPDFWrapper null context");
    }
}

MuPDFWrapper::MuPDFWrapper(const std::string& pdf_data, std::shared_ptr<FzContextPool> pool)
    : pool_(std::move(pool)) {
    if (!pool_) {
        throw std::runtime_error("FzContextPool is null");
    }
    pooled_ctx_ = pool_->Checkout();
    fz_context* c = ctx();
    if (!c) {
        throw std::runtime_error("Failed to checkout MuPDF context");
    }

    // RAII handles declared BEFORE fz_try
    FzStreamPtr stream(nullptr, FzStreamDeleter{c});
    fz_try(c) {
        stream.reset(fz_open_memory(c, reinterpret_cast<const unsigned char*>(pdf_data.data()), pdf_data.size()));
        // fz_open_document_with_stream does not take ownership of stream in a way that requires manual drop handled by RAII
        fz_document* raw_doc = fz_open_document_with_stream(c, "pdf", stream.get());
        doc_ = FzDocumentPtr(raw_doc, FzDocumentDeleter{c});
    } fz_catch(c) {
        std::string err_msg = fz_caught_message(c);
        spdlog::error("MuPDFWrapper constructor error: {}", err_msg);
        // stream will be dropped via RAII when scope exits; doc_ remains null
        throw std::runtime_error("MuPDF Error: " + err_msg);
    }
}

MuPDFWrapper::MuPDFWrapper(const std::string& pdf_data)
    : MuPDFWrapper(pdf_data, FzContextPool::Global()) {}

MuPDFWrapper::~MuPDFWrapper() = default;

MuPDFWrapper::MuPDFWrapper(MuPDFWrapper&& other) noexcept
    : pool_(std::move(other.pool_)),
      pooled_ctx_(std::move(other.pooled_ctx_)),
      doc_(std::move(other.doc_)) {}

MuPDFWrapper& MuPDFWrapper::operator=(MuPDFWrapper&& other) noexcept {
    if (this != &other) {
        doc_.reset();
        pooled_ctx_ = std::move(other.pooled_ctx_);
        pool_ = std::move(other.pool_);
        doc_ = std::move(other.doc_);
    }
    return *this;
}

int MuPDFWrapper::GetPageCount() const {
    ensure_valid();
    fz_context* c = ctx();
    int count = 0;
    fz_try(c) {
        count = fz_count_pages(c, doc_.get());
    } fz_catch(c) {
        throw std::runtime_error(std::string("GetPageCount error: ") + fz_caught_message(c));
    }
    return count;
}

int MuPDFWrapper::GetPageRotation(int page_index) const {
    ensure_valid();
    int page_count = GetPageCount();
    if (page_index < 0 || page_index >= page_count) {
        throw std::out_of_range("page_index out of range");
    }
    fz_context* c = ctx();
    int rotation = 0;
    fz_try(c) {
        pdf_document* pdoc = pdf_document_from_fz_document(c, doc_.get());
        if (pdoc) {
            pdf_obj* page_obj = pdf_lookup_page_obj(c, pdoc, page_index);
            if (page_obj) {
                rotation = pdf_dict_get_int(c, page_obj, PDF_NAME(Rotate));
            }
        }
    } fz_catch(c) {
        throw std::runtime_error(std::string("GetPageRotation error: ") + fz_caught_message(c));
    }
    rotation = ((rotation % 360) + 360) % 360;
    return rotation;
}

PageTextData MuPDFWrapper::ExtractPageText(int page_index) const {
    ensure_valid();
    int page_count = GetPageCount();
    if (page_index < 0 || page_index >= page_count) {
        throw std::out_of_range("page_index out of range");
    }

    PageTextData data;
    fz_context* c = ctx();
    // RAII handles declared BEFORE fz_try
    FzPagePtr page(nullptr, FzPageDeleter{c});
    FzStextPagePtr text_page(nullptr, FzStextPageDeleter{c});

    fz_try(c) {
        page.reset(fz_load_page(c, doc_.get(), page_index));
        if (!page) {
            fz_throw(c, FZ_ERROR_GENERIC, "Failed to load page");
        }

        constexpr fz_stext_options stext_options = {0};
        text_page.reset(fz_new_stext_page_from_page(c, page.get(), &stext_options));
        if (text_page) {
            data.text.reserve(4096);
            data.char_bboxes.reserve(4096);
            for (fz_stext_page_block_iterator pos = fz_stext_page_block_iterator_begin(text_page.get());
                 pos.block != nullptr;
                 pos = fz_stext_page_block_iterator_next(pos)) {

                fz_stext_block* block = pos.block;
                if (block->type == FZ_STEXT_BLOCK_TEXT) {
                    for (fz_stext_line* line = block->u.t.first_line; line; line = line->next) {
                        for (fz_stext_char* ch = line->first_char; ch; ch = ch->next) {
                            int code = ch->c;
                            fz_rect r = fz_rect_from_quad(ch->quad);
                            if (code >= 0 && code <= 0x7F) {
                                data.text.push_back(static_cast<char>(code));
                                data.char_bboxes.push_back(r);
                            } else if (code >= 0x80 && code <= 0x7FF) {
                                data.text.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
                                data.text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                                data.char_bboxes.push_back(r);
                                data.char_bboxes.push_back(fz_empty_rect);
                            } else if (code >= 0x800 && code <= 0xFFFF) {
                                if (code >= 0xD800 && code <= 0xDFFF) {
                                    data.text.append("\xEF\xBF\xBD", 3);
                                    data.char_bboxes.push_back(r);
                                    data.char_bboxes.push_back(fz_empty_rect);
                                    data.char_bboxes.push_back(fz_empty_rect);
                                } else {
                                    data.text.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                                    data.text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                                    data.text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                                    data.char_bboxes.push_back(r);
                                    data.char_bboxes.push_back(fz_empty_rect);
                                    data.char_bboxes.push_back(fz_empty_rect);
                                }
                            } else if (code >= 0x10000 && code <= 0x10FFFF) {
                                data.text.push_back(static_cast<char>(0xF0 | ((code >> 18) & 0x07)));
                                data.text.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
                                data.text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                                data.text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                                data.char_bboxes.push_back(r);
                                data.char_bboxes.push_back(fz_empty_rect);
                                data.char_bboxes.push_back(fz_empty_rect);
                                data.char_bboxes.push_back(fz_empty_rect);
                            } else {
                                data.text.append("\xEF\xBF\xBD", 3);
                                data.char_bboxes.push_back(r);
                                data.char_bboxes.push_back(fz_empty_rect);
                                data.char_bboxes.push_back(fz_empty_rect);
                            }
                        }
                        data.text += ' ';
                        data.char_bboxes.push_back(fz_empty_rect);
                    }
                    data.text += '\n';
                    data.char_bboxes.push_back(fz_empty_rect);
                }
            }
        }

    } fz_catch(c) {
        throw std::runtime_error(std::string("ExtractPageText error: ") + fz_caught_message(c));
    }
    return data;
}

std::vector<unsigned char> MuPDFWrapper::RenderPagePixmap(int page_index, int dpi) const {
    ensure_valid();
    int page_count = GetPageCount();
    if (page_index < 0 || page_index >= page_count) {
        throw std::out_of_range("page_index out of range for RenderPagePixmap");
    }
    if (dpi <= 0) {
        dpi = 300;
    }
    fz_context* c = ctx();
    std::vector<unsigned char> out;
    FzPagePtr page(nullptr, FzPageDeleter{c});
    FzPixmapPtr pix(nullptr, FzPixmapDeleter{c});

    fz_try(c) {
        page.reset(fz_load_page(c, doc_.get(), page_index));
        if (!page) {
            fz_throw(c, FZ_ERROR_GENERIC, "Failed to load page for pixmap");
        }
        float scale = static_cast<float>(dpi) / 72.0f;
        fz_matrix ctm = fz_scale(scale, scale);
        int rot = 0;
        // Get rotation without nested fz_try recursion hazard – use direct lookup
        {
            pdf_document* pdoc = pdf_document_from_fz_document(c, doc_.get());
            if (pdoc) {
                pdf_obj* page_obj = pdf_lookup_page_obj(c, pdoc, page_index);
                if (page_obj) {
                    rot = pdf_dict_get_int(c, page_obj, PDF_NAME(Rotate));
                    rot = ((rot % 360) + 360) % 360;
                }
            }
        }
        if (rot != 0) {
            fz_matrix rot_m = fz_rotate(static_cast<float>(rot));
            ctm = fz_concat(rot_m, ctm);
        }
        pix.reset(fz_new_pixmap_from_page_number(c, doc_.get(), page_index, ctm, fz_device_rgb(c), 0));
        if (!pix) {
            fz_throw(c, FZ_ERROR_GENERIC, "fz_new_pixmap_from_page_number failed");
        }
        size_t sz = static_cast<size_t>(pix->w) * static_cast<size_t>(pix->h) * static_cast<size_t>(pix->n);
        out.assign(pix->samples, pix->samples + sz);
    } fz_catch(c) {
        throw std::runtime_error(std::string("RenderPagePixmap error: ") + fz_caught_message(c));
    }
    return out;
}

void MuPDFWrapper::ApplyRedactions(int page_index, const std::vector<fz_rect>& bboxes, const float color[3]) const {
    ensure_valid();
    int page_count = GetPageCount();
    if (page_index < 0 || page_index >= page_count) {
        throw std::out_of_range("page_index out of range for redaction");
    }
    if (bboxes.empty()) return;

    constexpr float default_black[3] = {0.0f, 0.0f, 0.0f};
    const float* fill_color = color ? color : default_black;

    fz_context* c = ctx();
    FzPagePtr page(nullptr, FzPageDeleter{c});

    fz_try(c) {
        page.reset(fz_load_page(c, doc_.get(), page_index));
        if (!page) {
            fz_throw(c, FZ_ERROR_GENERIC, "Failed to load page for redaction");
        }

        if (pdf_page* ppage = pdf_page_from_fz_page(c, page.get())) {
            int valid_annots = 0;
            for (const auto& rect : bboxes) {
                if (fz_is_empty_rect(rect) || rect.x0 >= rect.x1 || rect.y0 >= rect.y1) {
                    continue;
                }
                pdf_annot* annot = pdf_create_annot(c, ppage, PDF_ANNOT_REDACT);
                pdf_set_annot_rect(c, annot, rect);
                pdf_set_annot_color(c, annot, 3, fill_color);
                valid_annots++;
            }
            if (valid_annots > 0) {
                pdf_redact_page(c, pdf_document_from_fz_document(c, doc_.get()), ppage, nullptr);
            }
        }
    } fz_catch(c) {
        throw std::runtime_error(std::string("ApplyRedactions error: ") + fz_caught_message(c));
    }
}

void MuPDFWrapper::ApplyNormalizedRedactions(
    std::span<const ml::worker::v1::NormalizedRedactionBox> custom_boxes,
    const float color[3]
) const {
    ensure_valid();
    if (custom_boxes.empty()) return;

    int total_pages = GetPageCount();

    std::map<int, std::vector<ml::worker::v1::NormalizedRedactionBox>> page_boxes_map;
    for (const auto& box : custom_boxes) {
        int page_idx = box.page_number() > 0 ? (box.page_number() - 1) : 0;
        if (page_idx < 0 || page_idx >= total_pages) {
            throw std::out_of_range("page_number in NormalizedRedactionBox is out of range: " + std::to_string(box.page_number()));
        }
        page_boxes_map[page_idx].push_back(box);
    }

    constexpr float default_black[3] = {0.0f, 0.0f, 0.0f};
    const float* fill_color = color ? color : default_black;

    fz_context* c = ctx();
    for (const auto& [page_idx, boxes] : page_boxes_map) {
        FzPagePtr page(nullptr, FzPageDeleter{c});
        fz_try(c) {
            page.reset(fz_load_page(c, doc_.get(), page_idx));
            if (!page) {
                fz_throw(c, FZ_ERROR_GENERIC, "Failed to load page for normalized redaction");
            }

            if (pdf_page* ppage = pdf_page_from_fz_page(c, page.get())) {
                // Sourced exclusively from fz_bound_page visible bounds (anchored at page_bounds.x0/y0)
                fz_rect page_bounds = fz_bound_page(c, page.get());
                float page_w = page_bounds.x1 - page_bounds.x0;
                float page_h = page_bounds.y1 - page_bounds.y0;

                int valid_annots = 0;

                for (const auto& box : boxes) {
                    float rx0 = std::min(box.x0(), box.x1());
                    float rx1 = std::max(box.x0(), box.x1());
                    float ry0 = std::min(box.y0(), box.y1());
                    float ry1 = std::max(box.y0(), box.y1());

                    if (rx0 >= rx1 || ry0 >= ry1) {
                        continue;
                    }

                    float crx0 = std::clamp(rx0, 0.0f, 1.0f);
                    float crx1 = std::clamp(rx1, 0.0f, 1.0f);
                    float cry0 = std::clamp(ry0, 0.0f, 1.0f);
                    float cry1 = std::clamp(ry1, 0.0f, 1.0f);

                    if (crx0 >= crx1 || cry0 >= cry1) {
                        continue;
                    }

                    fz_rect visual_rect;
                    visual_rect.x0 = page_bounds.x0 + crx0 * page_w;
                    visual_rect.x1 = page_bounds.x0 + crx1 * page_w;
                    visual_rect.y0 = page_bounds.y0 + cry0 * page_h;
                    visual_rect.y1 = page_bounds.y0 + cry1 * page_h;

                    if (fz_is_empty_rect(visual_rect)) {
                        continue;
                    }

                    pdf_annot* annot = pdf_create_annot(c, ppage, PDF_ANNOT_REDACT);
                    pdf_set_annot_rect(c, annot, visual_rect);
                    pdf_set_annot_color(c, annot, 3, fill_color);
                    valid_annots++;

                }

                if (valid_annots > 0) {
                    pdf_redact_page(c, pdf_document_from_fz_document(c, doc_.get()), ppage, nullptr);
                }
            }
        } fz_catch(c) {
            throw std::runtime_error(std::string("ApplyNormalizedRedactions error: ") + fz_caught_message(c));
        }
    }
}

std::string MuPDFWrapper::GetRedactedPdf() const {
    ensure_valid();
    fz_context* c = ctx();
    std::string redacted_pdf_data;
    // RAII before fz_try
    FzBufferPtr buf(nullptr, FzBufferDeleter{c});
    FzOutputPtr out(nullptr, FzOutputDeleter{c});

    fz_try(c) {
        buf.reset(fz_new_buffer(c, 1024));
        out.reset(fz_new_output_with_buffer(c, buf.get()));

        pdf_write_options write_opts = {0};
        write_opts.do_clean = 1;
        write_opts.do_sanitize = 1;
        write_opts.do_garbage = 3;
        write_opts.do_compress = 1;
        pdf_write_document(c, pdf_document_from_fz_document(c, doc_.get()), out.get(), &write_opts);
        fz_close_output(c, out.get());

        // Release output before reading buffer (output will be dropped via RAII after)
        redacted_pdf_data = std::string(reinterpret_cast<const char*>(buf->data), buf->len);
    } fz_catch(c) {
        throw std::runtime_error(std::string("GetRedactedPdf error: ") + fz_caught_message(c));
    }
    return redacted_pdf_data;
}

int MuPDFWrapper::CountIntersectingGlyphs(const std::string& redacted_pdf_bytes, const std::vector<fz_rect>& boxes, int page_index) const {
    if (boxes.empty() || redacted_pdf_bytes.empty()) return 0;
    fz_context* vctx = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);
    if (!vctx) throw std::runtime_error("[SEAM: IMPOSSIBLE_STATE] CountIntersectingGlyphs: fz_new_context failed");
    int surviving = 0;
    fz_try(vctx) {
        fz_register_document_handlers(vctx);
        fz_stream* vstream = fz_open_memory(vctx, reinterpret_cast<const unsigned char*>(redacted_pdf_bytes.data()), redacted_pdf_bytes.size());
        fz_document* vdoc = nullptr;
        fz_try(vctx) {
            vdoc = fz_open_document_with_stream(vctx, "pdf", vstream);
        } fz_always(vctx) {
            fz_drop_stream(vctx, vstream);
        } fz_catch(vctx) {
            fz_drop_context(vctx);
            throw std::runtime_error(std::string("[SEAM: IMPOSSIBLE_STATE] CountIntersectingGlyphs open: ") + fz_caught_message(vctx));
        }
        int vpage_count = 0;
        fz_try(vctx) {
            vpage_count = fz_count_pages(vctx, vdoc);
        } fz_catch(vctx) {
            fz_drop_document(vctx, vdoc);
            fz_drop_context(vctx);
            throw std::runtime_error(std::string("[SEAM: IMPOSSIBLE_STATE] CountIntersectingGlyphs count: ") + fz_caught_message(vctx));
        }
        if (page_index < 0 || page_index >= vpage_count) {
            fz_drop_document(vctx, vdoc);
            fz_drop_context(vctx);
            throw std::out_of_range("CountIntersectingGlyphs page_index out of range");
        }
        fz_page* vpage = nullptr;
        fz_stext_page* stext = nullptr;
        fz_try(vctx) {
            vpage = fz_load_page(vctx, vdoc, page_index);
            if (!vpage) fz_throw(vctx, FZ_ERROR_GENERIC, "Failed to load page for verification");
            fz_stext_options opts = {0};
            stext = fz_new_stext_page_from_page(vctx, vpage, &opts);
            if (stext) {
                for (auto block_it = fz_stext_page_block_iterator_begin(stext); block_it.block != nullptr; block_it = fz_stext_page_block_iterator_next(block_it)) {
                    fz_stext_block* block = block_it.block;
                    if (block->type != FZ_STEXT_BLOCK_TEXT) continue;
                    for (fz_stext_line* line = block->u.t.first_line; line; line = line->next) {
                        for (fz_stext_char* ch = line->first_char; ch; ch = ch->next) {
                            if (ch->c < 32 || ch->c > 126) continue;
                            fz_rect qr = fz_rect_from_quad(ch->quad);
                            if (fz_is_empty_rect(qr)) continue;
                            for (const auto& box : boxes) {
                                if (fz_is_empty_rect(box)) continue;
                                fz_rect isect = fz_intersect_rect(qr, box);
                                if (!fz_is_empty_rect(isect)) { surviving++; break; }
                            }
                        }
                    }
                }
            }
        } fz_always(vctx) {
            if (stext) fz_drop_stext_page(vctx, stext);
            if (vpage) fz_drop_page(vctx, vpage);
            fz_drop_document(vctx, vdoc);
        } fz_catch(vctx) {
            fz_drop_context(vctx);
            throw std::runtime_error(std::string("[SEAM: IMPOSSIBLE_STATE] CountIntersectingGlyphs re-extract: ") + fz_caught_message(vctx));
        }
        fz_drop_context(vctx);
    } fz_catch(vctx) {
        throw std::runtime_error(std::string("[SEAM: IMPOSSIBLE_STATE] CountIntersectingGlyphs outer: ") + fz_caught_message(vctx));
    }
    return surviving;
}

int MuPDFWrapper::CountPlaintextOccurrences(const std::string& redacted_pdf_bytes, const std::vector<std::string>& plaintexts) const {
    if (plaintexts.empty() || redacted_pdf_bytes.empty()) return 0;
    int total = 0;
    for (const auto& pt : plaintexts) {
        if (pt.empty()) continue;
        size_t pos = 0;
        while ((pos = redacted_pdf_bytes.find(pt, pos)) != std::string::npos) { total++; pos += pt.size(); }
    }
    return total;
}

void MuPDFWrapper::VerifyPostRedaction(const std::string& redacted_pdf_bytes,
                                       const std::map<int, std::vector<fz_rect>>& per_page_boxes,
                                       const std::vector<std::string>& target_plaintexts,
                                       const std::string& job_id,
                                       const std::string& traceparent) const {
    if (per_page_boxes.empty() && target_plaintexts.empty()) return;
    int total_surviving_glyphs = 0;
    int failing_page = -1;
    fz_rect failing_box = fz_empty_rect;
    for (const auto& [page_idx, boxes] : per_page_boxes) {
        if (boxes.empty()) continue;
        int cnt = CountIntersectingGlyphs(redacted_pdf_bytes, boxes, page_idx);
        if (cnt > 0) {
            total_surviving_glyphs += cnt;
            if (failing_page == -1) { failing_page = page_idx; failing_box = boxes.front(); }
        }
    }
    int total_surviving_bytes = CountPlaintextOccurrences(redacted_pdf_bytes, target_plaintexts);
    if (total_surviving_glyphs > 0 || total_surviving_bytes > 0) {
        std::string token_hash;
        if (!target_plaintexts.empty() && !target_plaintexts.front().empty()) {
            const auto& pt = target_plaintexts.front();
            token_hash = std::string(1, pt.front()) + "***" + std::string(1, pt.back()) + ":" + std::to_string(pt.size());
        } else token_hash = "n/a";
        std::string tp = traceparent.empty() ? "n/a" : traceparent;
        spdlog::critical("[SEAM: LINEAGE_GAP] job_id={} page_index={} box=[{},{},{},{}] surviving_glyph_count={} surviving_byte_count={} token_hash={} traceparent={}",
                         job_id, failing_page, failing_box.x0, failing_box.y0, failing_box.x1, failing_box.y1,
                         total_surviving_glyphs, total_surviving_bytes, token_hash, tp);
        throw std::runtime_error("[SEAM: LINEAGE_GAP] Post-redaction verification failed: surviving_glyph_count=" + std::to_string(total_surviving_glyphs) + " surviving_byte_count=" + std::to_string(total_surviving_bytes) + " job_id=" + job_id + " page_index=" + std::to_string(failing_page));
    }
}
