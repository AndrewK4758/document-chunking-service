#include "FzContextPool.h"

#include <algorithm>
#include <spdlog/spdlog.h>

FzContextPool::PooledContext::~PooledContext() {
    if (ctx_ && pool_) {
        pool_->Checkin(ctx_);
    }
}

FzContextPool::FzContextPool(size_t capacity) : capacity_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("FzContextPool capacity must be >= 1");
    }
    spdlog::info("Initializing FzContextPool with capacity {}", capacity_);
    for (size_t i = 0; i < capacity_; ++i) {
        fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_DEFAULT);
        if (!ctx) {
            for (auto* c : free_) {
                fz_drop_context(c);
            }
            throw std::runtime_error("Failed to create MuPDF context");
        }
        fz_register_document_handlers(ctx);
        free_.push_back(ctx);
    }
    context_create_count_ = capacity_;
    spdlog::info("FzContextPool initialized: capacity={} context_create_count={}", capacity_, context_create_count_);
}

FzContextPool::~FzContextPool() {
    std::lock_guard<std::mutex> lk(mu_);
    // PooledContext holds a shared_ptr to the pool, so in_use_debug_ should normally be empty here.
    if (!in_use_debug_.empty()) {
        spdlog::warn("FzContextPool destroyed with {} contexts still checked out", in_use_debug_.size());
    }
    for (auto* ctx : free_) {
        // Scavenge store before drop to bound memory
        int phase = 0;
        (void)fz_store_scavenge(ctx, 0, &phase);
        fz_drop_context(ctx);
    }
    free_.clear();
    for (auto* ctx : in_use_debug_) {
        int phase = 0;
        (void)fz_store_scavenge(ctx, 0, &phase);
        fz_drop_context(ctx);
    }
    in_use_debug_.clear();
}

FzContextPool::PooledContext FzContextPool::Checkout() {
    auto start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu_);
    if (free_.empty()) {
        exhausted_count_.fetch_add(1, std::memory_order_relaxed);
        spdlog::warn("FzContextPool exhausted capacity={} in_use={}", capacity_, in_use_debug_.size());
        throw std::runtime_error("FzContextPool exhausted: capacity " + std::to_string(capacity_) +
                                 " in_use " + std::to_string(in_use_debug_.size()));
    }
    fz_context* ctx = free_.back();
    free_.pop_back();
    in_use_debug_.push_back(ctx);
    checkout_count_.fetch_add(1, std::memory_order_relaxed);
    auto end = std::chrono::steady_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    latencies_ms_.push_back(latency_ms);
    // Keep histogram bounded (last 10000 samples)
    if (latencies_ms_.size() > 10000) {
        latencies_ms_.erase(latencies_ms_.begin(), latencies_ms_.begin() + 1000);
    }
    spdlog::debug("FzContextPool checkout latency={:.3f}ms in_use={} available={}", latency_ms,
                  in_use_debug_.size(), free_.size());
    // Require shared_ptr ownership; stack-allocated pools must be wrapped in shared_ptr for Checkout.
    std::shared_ptr<FzContextPool> self;
    try {
        self = shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        throw std::logic_error("FzContextPool::Checkout requires pool owned by std::shared_ptr (use std::make_shared)");
    }
    return {ctx, std::move(self)};
}

void FzContextPool::Checkin(fz_context* ctx) noexcept {
    if (!ctx) {
        drop_mismatch_count_.fetch_add(1, std::memory_order_relaxed);
        spdlog::error("FzContextPool::Checkin called with null context");
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = std::ranges::find(in_use_debug_, ctx);
    if (it == in_use_debug_.end()) {
        drop_mismatch_count_.fetch_add(1, std::memory_order_relaxed);
        spdlog::error("FzContextPool::Checkin drop_mismatch: context not in in_use set");
        // Still push to free to avoid leak, but count mismatch
        free_.push_back(ctx);
        checkin_count_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    in_use_debug_.erase(it);
    // Optional scavenge to bound store memory
    int phase = 0;
    (void)fz_store_scavenge(ctx, 0, &phase);
    free_.push_back(ctx);
    checkin_count_.fetch_add(1, std::memory_order_relaxed);
    spdlog::debug("FzContextPool checkin in_use={} available={}", in_use_debug_.size(), free_.size());
}

FzContextPool::Stats FzContextPool::GetStats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s;
    s.capacity = capacity_;
    s.in_use = in_use_debug_.size();
    s.available = free_.size();
    s.checkout_count = checkout_count_.load(std::memory_order_relaxed);
    s.checkin_count = checkin_count_.load(std::memory_order_relaxed);
    s.exhausted_count = exhausted_count_.load(std::memory_order_relaxed);
    s.context_create_count = context_create_count_;
    return s;
}

double FzContextPool::GetP95CheckoutMs() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (latencies_ms_.empty()) return 0.0;
    std::vector<double> sorted = latencies_ms_;
    std::ranges::sort(sorted);
    auto idx = static_cast<size_t>(sorted.size() * 0.95);
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

double FzContextPool::GetPoolUtilization() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (capacity_ == 0) return 0.0;
    return static_cast<double>(in_use_debug_.size()) / static_cast<double>(capacity_);
}

void FzContextPool::RecordLatency(double ms) {
    std::lock_guard<std::mutex> lk(mu_);
    latencies_ms_.push_back(ms);
}

size_t FzContextPool::ResolveCapacity() {
    if (const char* env = std::getenv("MUPDF_POOL_SIZE")) {
        try {
            size_t v = static_cast<size_t>(std::stoul(env));
            if (v >= 1) return v;
            spdlog::warn("MUPDF_POOL_SIZE must be >=1, got '{}', using default", env);
        } catch (const std::exception& e) {
            spdlog::warn("Invalid MUPDF_POOL_SIZE='{}': {}, using default", env, e.what());
        } catch (...) {
            spdlog::warn("Invalid MUPDF_POOL_SIZE='{}', using default", env);
        }
    }
    unsigned hc = std::thread::hardware_concurrency();
    return std::max<size_t>(2, hc == 0 ? 4 : hc);
}

std::shared_ptr<FzContextPool> FzContextPool::Global() {
    static std::mutex g_mu;
    static std::weak_ptr<FzContextPool> g_weak;
    std::lock_guard<std::mutex> lk(g_mu);
    auto sp = g_weak.lock();
    if (!sp) {
        size_t cap = ResolveCapacity();
        sp = std::make_shared<FzContextPool>(cap);
        g_weak = sp;
    }
    return sp;
}
