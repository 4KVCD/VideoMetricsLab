// A team of threads that runs one pass over a picture's rows at a time, for
// the engines that split one picture's work between threads (Butteraugli).
// The rows are cut into bands that the threads, the caller among them, take
// one by one, so that faster cores and threads the system runs late simply
// take more or fewer bands; run() returns when every band is done. A pass
// gives each row the same work whatever band it falls in, so the results do
// not depend on the number of threads.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

namespace vml {

class RowPool {
public:
    // `threads` counts the caller: 1 runs every pass on the caller alone.
    explicit RowPool(std::size_t threads);
    ~RowPool();
    RowPool(const RowPool&) = delete;
    RowPool& operator=(const RowPool&) = delete;

    std::size_t threads() const { return workers_.size() + 1; }

    // Calls fn(y0, y1) for bands [y0, y1) that cover rows [0, rows), each
    // band but the last a multiple of `align` rows (for passes that write
    // the transposed picture: whole cache lines per thread), and waits for
    // all of them.
    template <class F>
    void run(std::size_t rows, F&& fn, std::size_t align = 1) {
        if (rows == 0) return;
        const std::size_t band_rows = band_size(rows, align);
        if (workers_.empty() || band_rows >= rows) {
            fn(std::size_t{0}, rows);
            return;
        }
        using Fn = std::remove_reference_t<F>;
        dispatch(rows, band_rows, [](void* f, std::size_t y0, std::size_t y1) { (*static_cast<Fn*>(f))(y0, y1); },
                 const_cast<void*>(static_cast<const void*>(&fn)));
    }

private:
    using Call = void (*)(void* fn, std::size_t y0, std::size_t y1);

    std::size_t band_size(std::size_t rows, std::size_t align) const;
    void dispatch(std::size_t rows, std::size_t band_rows, Call call, void* fn);
    void take_bands(std::uint32_t pass) noexcept;  // a pass's fn must not throw: others may be running it
    void worker();

    std::vector<std::thread> workers_;

    // The current pass, written by the caller before it publishes the pass
    // (claim_, then pass_); atomics so that a thread reading them while the
    // next pass is being set up reads some value (its claim then fails).
    std::atomic<Call> call_{nullptr};
    std::atomic<void*> fn_{nullptr};
    std::atomic<std::size_t> rows_{0};
    std::atomic<std::size_t> band_rows_{0};

    // The pass number (top 32 bits), its number of bands (next 16) and its
    // next unclaimed band (low 16) in one word: a thread claims a band only
    // by moving this word on from its own pass's value, so one that wakes
    // late can neither take a band of a later pass nor make one skip a band.
    std::atomic<std::uint64_t> claim_{0};
    std::atomic<std::uint32_t> pending_{0};  // bands of the current pass not finished
    std::atomic<std::uint32_t> pass_{0};     // what idle threads wait on
    std::atomic<std::uint32_t> sleepers_{0};  // threads asleep on pass_ (else no notify is needed)
    std::atomic<bool> stop_{false};
};

}  // namespace vml
