#pragma once

#include <stdckdint.h>

#include <memory>
#include "mupdf/fitz.h"
#include "mupdf/pdf.h"

// RAII deleters for all MuPDF handle types, each capturing the correct fz_context*.
// All deleters are noexcept and fz_catch-safe (they never call fz_try/fz_throw).

struct FzDocumentDeleter {
    fz_context* ctx = nullptr;
    void operator()(fz_document* doc) const noexcept {
        if (doc && ctx) fz_drop_document(ctx, doc);
    }
};

struct FzPageDeleter {
    fz_context* ctx = nullptr;
    void operator()(fz_page* page) const noexcept {
        if (page && ctx) fz_drop_page(ctx, page);
    }
};

struct FzPixmapDeleter {
    fz_context* ctx = nullptr;
    void operator()(fz_pixmap* pix) const noexcept {
        if (pix && ctx) fz_drop_pixmap(ctx, pix);
    }
};

struct FzStextPageDeleter {
    fz_context* ctx = nullptr;
    void operator()(fz_stext_page* tp) const noexcept {
        if (tp && ctx) fz_drop_stext_page(ctx, tp);
    }
};

struct FzBufferDeleter {
    fz_context* ctx = nullptr;
    void operator()(fz_buffer* buf) const noexcept {
        if (buf && ctx) fz_drop_buffer(ctx, buf);
    }
};

struct FzStreamDeleter {
    fz_context* ctx = nullptr;
    void operator()(fz_stream* stm) const noexcept {
        if (stm && ctx) fz_drop_stream(ctx, stm);
    }
};

struct FzOutputDeleter {
    fz_context* ctx = nullptr;
    void operator()(fz_output* out) const noexcept {
        if (out && ctx) fz_drop_output(ctx, out);
    }
};

using FzDocumentPtr  = std::unique_ptr<fz_document, FzDocumentDeleter>;
using FzPagePtr      = std::unique_ptr<fz_page, FzPageDeleter>;
using FzPixmapPtr    = std::unique_ptr<fz_pixmap, FzPixmapDeleter>;
using FzStextPagePtr = std::unique_ptr<fz_stext_page, FzStextPageDeleter>;
using FzBufferPtr    = std::unique_ptr<fz_buffer, FzBufferDeleter>;
using FzStreamPtr    = std::unique_ptr<fz_stream, FzStreamDeleter>;
using FzOutputPtr    = std::unique_ptr<fz_output, FzOutputDeleter>;

// Verify noexcept guarantees at compile time
static_assert(noexcept(std::declval<FzDocumentDeleter>()(std::declval<fz_document*>())),
              "FzDocumentDeleter must be noexcept");
static_assert(noexcept(std::declval<FzPageDeleter>()(std::declval<fz_page*>())),
              "FzPageDeleter must be noexcept");
static_assert(noexcept(std::declval<FzPixmapDeleter>()(std::declval<fz_pixmap*>())),
              "FzPixmapDeleter must be noexcept");
static_assert(noexcept(std::declval<FzStextPageDeleter>()(std::declval<fz_stext_page*>())),
              "FzStextPageDeleter must be noexcept");
static_assert(noexcept(std::declval<FzBufferDeleter>()(std::declval<fz_buffer*>())),
              "FzBufferDeleter must be noexcept");
static_assert(noexcept(std::declval<FzStreamDeleter>()(std::declval<fz_stream*>())),
              "FzStreamDeleter must be noexcept");
