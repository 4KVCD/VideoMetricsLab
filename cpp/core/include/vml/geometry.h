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

// A sample aspect ratio ("4:3") as a fraction; an unknown or unset one means
// square pixels.
std::pair<int, int> sar_fraction(const std::string& sar);

// The shape of the picture as displayed, after cropping.
double display_aspect_ratio(const VideoInfo& info, const std::optional<CropBox>& crop = std::nullopt);

// Why two videos cannot be compared (their timelines do not match), or
// nothing.
std::optional<std::string> pair_problem(const VideoInfo& source, const VideoInfo& distorted, double duration_limit);

}  // namespace vml
