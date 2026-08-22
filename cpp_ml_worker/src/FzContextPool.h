#pragma once

#include <stdckdint.h>

#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <cstdlib>

#include "mupdf/fitz.h"

class FzContextPool : public std::enable_shared_from_this<FzContextPool> {
public:
    struct Stats {
        size_t capacity = 0;
        size_t in_use = 0;
        size_t available = 0;
        uint64_t checkout_count = 0;
        uint64_t checkin_count = 0;
        uint64_t exhausted_count = 0;
        uint64_t context_create_count = 0;
    };

    class PooledContext {
    public:
        PooledContext() noexcept = default;
        PooledContext(fz_context* ctx, std::shared_ptr<FzContextPool> pool) noexcept
            : ctx_(ctx), pool_(std::move(pool)), owner_thread_(std::this_thread::get_id()) {}
        ~PooledContext();

        PooledContext(const PooledContext&) = delete;
        PooledContext& operator=(const PooledContext&) = delete;

        PooledContext(PooledContext&& other) noexcept
            : ctx_(other.ctx_), pool_(std::move(other.pool_)), owner_thread_(other.owner_thread_) {
            other.ctx_ = nullptr;
        }
        PooledContext& operator=(PooledContext&& other) noexcept {
            if (this != &other) {
                // Return current context if any
                if (ctx_ && pool_) {
                    pool_->Checkin(ctx_);
                }
                ctx_ = other.ctx_;
                pool_ = std::move(other.pool_);
                owner_thread_ = other.owner_thread_;
                other.ctx_ = nullptr;
            }
            return *this;
        }

        fz_context* get() const noexcept { return ctx_; }
        explicit operator bool() const noexcept { return ctx_ != nullptr; }

        // Debug helper to verify thread affinity
        bool is_owner_thread() const noexcept {
            return owner_thread_ == std::this_thread::get_id();
        }

    private:
        fz_context* ctx_ = nullptr;
        std::shared_ptr<FzContextPool> pool_ = nullptr;
        std::thread::id owner_thread_{};
    };

    explicit FzContextPool(size_t capacity);
    ~FzContextPool();

    FzContextPool(const FzContextPool&) = delete;
    FzContextPool& operator=(const FzContextPool&) = delete;

    // Fail-fast: throws std::runtime_error if exhausted.
    PooledContext Checkout();

    Stats GetStats() const;
    double GetP95CheckoutMs() const;
    double GetPoolUtilization() const;

    static size_t ResolveCapacity();
    static std::shared_ptr<FzContextPool> Global();

private:
    void Checkin(fz_context* ctx) noexcept;
    friend class PooledContext;
    void RecordLatency(double ms);

    mutable std::mutex mu_;
    std::vector<fz_context*> free_;
    std::vector<fz_context*> in_use_debug_;
    std::atomic<uint64_t> checkout_count_{0};
    std::atomic<uint64_t> checkin_count_{0};
    std::atomic<uint64_t> exhausted_count_{0};
    std::atomic<uint64_t> drop_mismatch_count_{0};
    const size_t capacity_;
    uint64_t context_create_count_ = 0;

    // Histogram: store latencies in ms for p95 calculation (guarded by mu_)
    std::vector<double> latencies_ms_;
};
