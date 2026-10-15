#include "vml/film_grain.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace vml {

namespace {

// H.274's cutoff frequencies, 2 to 14: the library takes them as indices
// from 0.
constexpr int min_cutoff = 2;
constexpr int max_cutoff = 14;

// An interval's (scale, horizontal cutoff, vertical cutoff), adapted to a
// subsampled chroma plane (_model_values). The grain is specified at full
// resolution, so a subsampled chroma plane gets cutoffs doubled in the
// directions it is subsampled in, and a smaller scale: half for 4:2:0 (as
// FFmpeg and VTM do), 1/sqrt(2) for 4:2:2 (as VTM does).
std::tuple<int, int, int> model_values(const std::vector<int>& values, bool chroma, std::pair<int, int> chroma_shift) {
    int scale = values.empty() ? 0 : values[0];
    int horizontal = values.size() > 1 ? values[1] : 8;
    int vertical = values.size() > 2 ? values[2] : horizontal;
    if (chroma && chroma_shift == std::pair(1, 1)) {
        scale >>= 1;
        horizontal *= 2;
        vertical *= 2;
    } else if (chroma && chroma_shift == std::pair(1, 0)) {
        scale = (scale * 181) >> 8;  // 181 / 256 is 1/sqrt(2)
        horizontal *= 2;
    }
    return {scale, horizontal, vertical};
}

std::uint8_t cutoff_index(int cutoff) {
    return static_cast<std::uint8_t>(std::clamp(cutoff, min_cutoff, max_cutoff) - min_cutoff);
}

}  // namespace

std::vector<std::pair<int, int>> grain_plane_sizes(int width, int height, std::pair<int, int> chroma_shift,
                                                   int planes) {
    // Python's -(-n >> s): n divided by 2^s, rounded up.
    const std::pair<int, int> chroma{-(-width >> chroma_shift.first), -(-height >> chroma_shift.second)};
    std::vector<std::pair<int, int>> sizes{{width, height}, chroma, chroma};
    sizes.resize(static_cast<std::size_t>(std::clamp(planes, 0, 3)));
    return sizes;
}

std::optional<GrainParams> grain_params(const FilmGrainMessage& message, int seed, int depth,
                                        const std::vector<std::pair<int, int>>& sizes,
                                        std::pair<int, int> chroma_shift) {
    if (message.model_id != 0) {
        return std::nullopt;
    }
    GrainParams params{};
    params.depth = depth;
    params.log2_scale_factor = message.log2_scale_factor;
    params.poc = seed;
    for (std::size_t component = 0; component < std::min<std::size_t>(sizes.size(), 3); ++component) {
        GrainPlane& plane = params.plane[component];
        std::tie(plane.width, plane.height) = sizes[component];
        const std::vector<FilmGrainInterval>& intervals = message.components[component];
        plane.intervals = static_cast<int>(std::min<std::size_t>(intervals.size(), 256));
        for (int index = 0; index < plane.intervals; ++index) {
            const FilmGrainInterval& interval = intervals[static_cast<std::size_t>(index)];
            const auto [scale, horizontal, vertical] = model_values(interval.values, component > 0, chroma_shift);
            plane.lower[index] = static_cast<std::uint8_t>(interval.lower);
            plane.upper[index] = static_cast<std::uint8_t>(interval.upper);
            plane.scale[index] = static_cast<std::int16_t>(std::clamp(scale, -32768, 32767));
            plane.h[index] = cutoff_index(horizontal);
            plane.v[index] = cutoff_index(vertical);
        }
    }
    return params;
}

}  // namespace vml
