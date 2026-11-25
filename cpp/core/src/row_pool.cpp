#include "row_pool.h"

#include <algorithm>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#endif

namespace vml {
namespace {

// A pass is usually followed by the next one within microseconds, so idle
// threads spin this many times before they sleep, which costs a system call
// to wake them from (about 2000 pauses are 10 to 50 microseconds, depending
// on the CPU).
constexpr int kSpins = 2000;

// Bands per thread: enough for threads that start late or run on slower
// cores to even out, few enough that taking a band costs nothing.
constexpr std::size_t kBandsPerThread = 4;

// claim_'s fields.
constexpr std::uint64_t kBandMask = 0xFFFF;

constexpr std::uint64_t make_claim(std::uint32_t pass, std::uint64_t bands, std::uint64_t next) {
    return static_cast<std::uint64_t>(pass) << 32 | bands << 16 | next;
}

inline void cpu_relax() {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __asm__ __volatile__("yield");
#endif
}

}  // namespace

RowPool::RowPool(std::size_t threads) {
    const std::size_t helpers = threads > 1 ? threads - 1 : 0;
    workers_.reserve(helpers);
    try {
        for (std::size_t i = 0; i < helpers; ++i) workers_.emplace_back([this] { worker(); });
    } catch (...) {
        // A thread that cannot start (std::system_error): the destructor does
        // not run for a pool that was never built, so the threads already
        // started are stopped here, as joinable threads left behind would end
        // the process.
        stop_workers();
        throw;
    }
}

RowPool::~RowPool() { stop_workers(); }

void RowPool::stop_workers() noexcept {
    stop_.store(true);
    pass_.fetch_add(1);  // seq_cst, as the sleepers' check
    pass_.notify_all();
    for (std::thread& t : workers_) t.join();
}

std::size_t RowPool::band_size(std::size_t rows, std::size_t align) const {
    const std::size_t bands = std::min(threads() * kBandsPerThread, kMaxBands);
    const std::size_t rows_per_band = std::max<std::size_t>((rows + bands - 1) / bands, (rows + kMaxBands - 1) / kMaxBands);
    return std::max<std::size_t>(1, (rows_per_band + align - 1) / align) * align;
}

void RowPool::dispatch(std::size_t rows, std::size_t band_rows, Call call, void* fn) {
    // The previous pass is over (run() waits for it), but a thread may still
    // be about to try a claim in it: that claim fails, as claim_ moves on.
    const std::uint32_t pass = pass_.load(std::memory_order_relaxed) + 1;
    const auto bands = static_cast<std::uint32_t>((rows + band_rows - 1) / band_rows);  // <= kMaxBands
    call_.store(call, std::memory_order_relaxed);
    fn_.store(fn, std::memory_order_relaxed);
    rows_.store(rows, std::memory_order_relaxed);
    band_rows_.store(band_rows, std::memory_order_relaxed);
    pending_.store(bands, std::memory_order_relaxed);
    claim_.store(make_claim(pass, bands, 0), std::memory_order_release);
    pass_.store(pass);  // seq_cst: either a thread going to sleep sees it, or this sees the sleeper
    if (sleepers_.load() != 0) pass_.notify_all();

    take_bands(pass);
    for (int spin = 0;; ++spin) {
        const std::uint32_t left = pending_.load(std::memory_order_acquire);
        if (left == 0) return;
        if (spin < kSpins) {
            cpu_relax();
        } else {
            pending_.wait(left, std::memory_order_acquire);  // the last band's thread notifies
        }
    }
}

void RowPool::take_bands(std::uint32_t pass) noexcept {
    const Call call = call_.load(std::memory_order_relaxed);
    void* const fn = fn_.load(std::memory_order_relaxed);
    const std::size_t rows = rows_.load(std::memory_order_relaxed);
    const std::size_t band_rows = band_rows_.load(std::memory_order_relaxed);
    std::uint64_t claim = claim_.load(std::memory_order_acquire);
    for (;;) {
        // A claim succeeds only while this pass has a band left, so before
        // it is done; and the next pass is set up only after every band of
        // this one is done, so the values read above are this pass's.
        const std::uint64_t next = claim & kBandMask;
        if ((claim >> 32) != pass || next >= (claim >> 16 & kBandMask)) return;
        if (!claim_.compare_exchange_weak(claim, claim + 1, std::memory_order_acquire,
                                          std::memory_order_acquire)) {
            continue;  // claim holds the current value
        }
        const std::size_t y0 = static_cast<std::size_t>(next) * band_rows;
        call(fn, y0, std::min(rows, y0 + band_rows));
        if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) pending_.notify_one();
        claim = claim_.load(std::memory_order_acquire);
    }
}

void RowPool::worker() {
    std::uint32_t seen = 0;
    for (;;) {
        std::uint32_t pass = pass_.load(std::memory_order_acquire);
        for (int spin = 0; pass == seen && spin < kSpins; ++spin) {
            cpu_relax();
            pass = pass_.load(std::memory_order_acquire);
        }
        while (pass == seen) {
            sleepers_.fetch_add(1);  // seq_cst, as dispatch's check
            if (pass_.load() == seen) pass_.wait(seen);
            sleepers_.fetch_sub(1);
            pass = pass_.load(std::memory_order_acquire);
        }
        if (stop_.load()) return;
        seen = pass;
        take_bands(pass);
    }
}

}  // namespace vml
