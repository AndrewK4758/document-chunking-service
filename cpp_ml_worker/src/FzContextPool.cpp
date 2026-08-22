#include "FzContextPool.h"

#include <algorithm>
#include <spdlog/spdlog.h>

// PooledContext destructor returns context to pool
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
            // Clean up already created contexts
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
    // Warn if leaks outstanding
    if (!in_use_debug_.empty()) {
        spdlog::warn("FzContextPool destroyed with {} contexts still checked out", in_use_debug_.size());
        // Force drop remaining contexts (they will be reclaimed via Checkin path if PooledContext still alive)
        // But if those PooledContexts outlive pool, their Checkin will be no-op due to weak pool?
        // Since PooledContext holds shared_ptr, pool won't be destroyed until all leases released.
        // So this branch should normally be empty.
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
        spdlog::warn("FzContextPool exhausted capacity={} in_use={}", capacity_, capacity_ - free_.size());
        throw std::runtime_error("FzContextPool exhausted: capacity " + std::to_string(capacity_) +
                                 " in_use " + std::to_string(capacity_));
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
    // Construct PooledContext with shared_from_this; need to be owned by shared_ptr
    // Must handle case where this is not yet managed by shared_ptr (e.g., stack allocation in tests)
    // Try shared_from_this, fallback to raw this with no ownership if it throws
    std::shared_ptr<FzContextPool> self;
    try {
        self = shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        // Not owned by shared_ptr – create a non-owning shared_ptr with no deleter
        // But Checkin would then use self which is empty; we need to capture pool via raw pointer?
        // Instead, we require that Checkout is only called on shared_ptr-owned pools.
        // For stack-allocated pools in tests, this path may be hit; we create aliasing shared_ptr with null deleter
        // Caller must ensure pool lifetime exceeds lease – for test stack case, this is true for duration of test.
        // Use shared_ptr with custom deleter that does nothing.
        self = std::shared_ptr<FzContextPool>(this, [](FzContextPool*) {});
    }
    // Unlock before constructing return value (already unlocked via lock_guard destructor sequencing)
    // Use direct construction
    return PooledContext(ctx, std::move(self));
}

void FzContextPool::Checkin(fz_context* ctx) noexcept {
    if (!ctx) {
        drop_mismatch_count_.fetch_add(1, std::memory_order_relaxed);
        spdlog::error("FzContextPool::Checkin called with null context");
        return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    auto it = std::find(in_use_debug_.begin(), in_use_debug_.end(), ctx);
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
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(0.95 * sorted.size());
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
        } catch (...) {
            // fall through to default
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
