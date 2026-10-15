// A VVC video's film grain as Video Compare draws it (the Python app's
// vmaf_app/core/d3d11_film_grain.py): a film grain characteristics message
// (H.274 section 8.5) turned into the parameters d3d11_film_grain.dll takes
// for one frame (native/h274/grain_params.h).
#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "grain_params.h"

namespace vml {

// One intensity interval of a colour component's grain: its bounds, as
// 8-bit codes, and its model values (the grain's scale, then its horizontal
// and vertical cutoff frequencies, as many of them as the message gives).
struct FilmGrainInterval {
    int lower = 0;
    int upper = 0;
    std::vector<int> values;
};

// A film grain characteristics message, as FFmpeg's decoders give it
// (AVFilmGrainH274Params): each colour component's intervals (Y, Cb, Cr),
// none for a component with no grain.
struct FilmGrainMessage {
    int model_id = 0;
    int log2_scale_factor = 0;
    std::array<std::vector<FilmGrainInterval>, 3> components;
};

// The picture's size in each of its `planes` planes (plane_sizes):
// `chroma_shift` is the chroma planes' subsampling as powers of two,
// horizontally and vertically, (1, 1) for 4:2:0; odd sizes round up.
std::vector<std::pair<int, int>> grain_plane_sizes(int width, int height, std::pair<int, int> chroma_shift,
                                                   int planes);

// The library's parameters for a frame of planes `sizes` at `depth` bits,
// its chroma subsampled by `chroma_shift`, its grain seeded by `seed` (its
// picture order count) (native_params). Nothing for a grain the library
// does not draw: H.274's autoregressive model (model 1), which FFmpeg does
// not synthesize either.
std::optional<GrainParams> grain_params(const FilmGrainMessage& message, int seed, int depth,
                                        const std::vector<std::pair<int, int>>& sizes,
                                        std::pair<int, int> chroma_shift);

}  // namespace vml
