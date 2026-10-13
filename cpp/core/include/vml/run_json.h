// What a run is given, as text for a process of its own (isolated.h): the
// videos, the run's options and the black-bar crops. Its numbers keep every
// digit: JSON's doubles are written to the last bit.
#pragma once

#include <nlohmann/json.hpp>

#include <optional>

#include "vml/options.h"
#include "vml/video.h"

namespace vml {

nlohmann::json video_json(const VideoInfo& info);
VideoInfo video_from(const nlohmann::json& text);

nlohmann::json options_json(const VmafOptions& options);
VmafOptions options_from(const nlohmann::json& text);

// A crop, or null for none.
nlohmann::json crop_json(const std::optional<CropBox>& crop);
std::optional<CropBox> crop_from(const nlohmann::json& text);

}  // namespace vml
