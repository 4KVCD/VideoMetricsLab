// SSIMULACRA 2 computed by VideoMetricsLab itself, with the scores of libjxl
// 0.12's ssimulacra2 tool to the last bit, without its PNG files and with a
// small fraction of its memory.
//
// The tool works on whole pictures: one linear copy, XYB copies, products and
// blurred planes, about 1 GB for a 4K pair. This engine streams the pair from
// top to bottom instead, a few dozen rows at a time split between threads,
// and keeps only the rows its filters still need (at 4K, about 10 MB on one
// thread, about 30 MB on several, which take more rows at a time). Every
// value is computed with libjxl's operations in libjxl's order
// (ssimulacra2_kernels.h), and the sums run in libjxl's order, so the score
// is identical for any number of threads.
//
// Terms whose weight in the score is zero are skipped: they add nothing
// (weight 0 times a finite average), and they are about a third of the work.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vml/jxl_linear.h"

namespace vml {

class Ssimulacra2 {
public:
    // One scale's 18 averages, as libjxl's MsssimScale holds them: per
    // channel (X, Y, B) the SSIM map's mean and 4-norm, then the edge maps'
    // ringing mean and 4-norm and blurring mean and 4-norm.
    struct ScaleAverages {
        std::array<double, 6> ssim{};
        std::array<double, 12> edge{};
    };

    // `weighted` skips the terms the score gives no weight; `all` computes
    // every average (for checking them against libjxl's).
    enum class Terms { weighted, all };

    // An engine for pictures of this size, the source read as
    // `source_reading` says and the test as `test_reading` (both made with
    // fused_multiply_add()). As libjxl's tool, each picture's XYB is made
    // for its own intensity target. Each pair's work is split between
    // `threads` threads (the caller's among them).
    Ssimulacra2(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
                std::size_t height, Terms terms = Terms::weighted, std::size_t threads = 1);
    ~Ssimulacra2();
    Ssimulacra2(Ssimulacra2&&) noexcept;
    Ssimulacra2& operator=(Ssimulacra2&&) noexcept;

    std::size_t width() const;
    std::size_t height() const;

    // The memory an engine (Terms::weighted) for pictures of this size holds
    // on `threads` threads: its rings and each thread's product row; for
    // sizing how many engines run at a time, without making one.
    static std::size_t memory_bytes(std::size_t width, std::size_t height, std::size_t threads);

    // The score of `test` against `source`, both rgb48 pictures (red, green,
    // blue interleaved) whose rows are `stride` samples apart.
    double score(const std::uint16_t* source, const std::uint16_t* test, std::size_t stride);

    // The last score's averages, one entry per scale (skipped terms are 0).
    const std::vector<ScaleAverages>& averages() const;

    // Whether this CPU's code fuses multiply-adds, as libjxl's does on it.
    static bool fused_multiply_add();

private:
    class Engine;
    std::unique_ptr<Engine> engine_;
};

}  // namespace vml
