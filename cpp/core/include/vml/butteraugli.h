// Butteraugli computed by VideoMetricsLab itself, with the distortion maps
// of libjxl 0.12's butteraugli_main to the last bit (its --rawdistmap, which
// VideoMetricsLab turns into its score with
// butteraugli_norm3_top_row_first), without its PNG files or a process per
// picture.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vml/jxl_linear.h"

namespace vml {

class Butteraugli {
public:
    // An engine for pictures of this size, the source read as
    // `source_reading` says and the test as `test_reading` (both made with
    // Ssimulacra2::fused_multiply_add()), the source's linear value of 1.0
    // viewed at `intensity_target` nits (butteraugli_main's
    // --intensity_target) and the test brought to the source's scale first,
    // that splits each picture's work between `threads` threads (the
    // caller's among them; the maps are the same for any number).
    Butteraugli(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
                std::size_t height, float intensity_target, std::size_t threads = 1);
    ~Butteraugli();
    Butteraugli(Butteraugli&&) noexcept;
    Butteraugli& operator=(Butteraugli&&) noexcept;

    std::size_t width() const;
    std::size_t height() const;

    // The distortion map of `test` against `source`, both rgb48 pictures (red,
    // green, blue interleaved) whose rows are `stride` samples apart: one
    // float per pixel, top row first. Valid until the next call.
    const std::vector<float>& distortion_map(const std::uint16_t* source, const std::uint16_t* test,
                                             std::size_t stride);

private:
    class Engine;
    std::unique_ptr<Engine> engine_;
};

}  // namespace vml
