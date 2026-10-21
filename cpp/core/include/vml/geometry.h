// The geometry of a comparison: the picture each video gives once its crop
// is applied, the size the two are compared at, their shapes on screen, and
// whether two videos can be compared at all. A port of
// vmaf_app/core/geometry.py.
#pragma once

#include <optional>
#include <string>
#include <utility>

#include "vml/options.h"
#include "vml/video.h"

namespace vml {

using Size = std::pair<int, int>;  // width, height

// The size of the picture that reaches the comparison: the crop's, or the
// whole video's.
Size content_size(const VideoInfo& info, const std::optional<CropBox>& crop);

// The size frames are compared at: one video is scaled to the other's size
// (which one, by `scale_direction`) after their crops.
Size compared_dimensions(const VideoInfo& source, const VideoInfo& distorted, ScaleDirection scale_direction,
                         const std::optional<CropBox>& source_crop = std::nullopt,
                         const std::optional<CropBox>& distorted_crop = std::nullopt);

// compared_dimensions for a comparison's options.
Size analysis_dimensions(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                         const std::optional<CropBox>& source_crop = std::nullopt,
                         const std::optional<CropBox>& distorted_crop = std::nullopt);

// The size a resolution test compares at: the source's own (cropped) size,
// as the downscale is undone before the comparison.
inline Size resample_analysis_dimensions(const VideoInfo& source, const std::optional<CropBox>& source_crop) {
    return content_size(source, source_crop);
}

// The height a resolution test scales a picture of `size` down to, `width`
// wide: even, and the picture's own shape.
int resample_down_height(int width, Size size);

// A sample aspect ratio ("4:3") as a fraction; an unknown or unset one means
// square pixels.
std::pair<int, int> sar_fraction(const std::string& sar);

// The shape of the picture as displayed, after cropping.
double display_aspect_ratio(const VideoInfo& info, const std::optional<CropBox>& crop = std::nullopt);

// Why two videos cannot be compared because their pictures are different
// shapes after cropping, or nothing: scaling one to the other would stretch
// it, and the score would look like any other while meaning nothing. Every
// metric refuses such a pair.
std::optional<std::string> shape_problem(const VideoInfo& source, const VideoInfo& distorted,
                                         const std::optional<CropBox>& source_crop,
                                         const std::optional<CropBox>& distorted_crop);

// Why two videos cannot be compared (their timelines do not match), or
// nothing.
std::optional<std::string> pair_problem(const VideoInfo& source, const VideoInfo& distorted, double duration_limit);

// Video Compare's check: a raw stream whose frame rate (FFmpeg's guess)
// disagrees with the other video's is refused, as pair_problem refuses it.
// Video Compare shows other mismatched pairs anyway, but it places a raw
// stream's frames by that guessed rate, so it would show the wrong frame.
std::optional<std::string> untimed_pair_problem(const VideoInfo& source, const VideoInfo& distorted);

// A raw stream's codec as messages name it: "H.264", "HEVC", "MPEG-2"...
std::string codec_label(const std::string& codec);

}  // namespace vml
