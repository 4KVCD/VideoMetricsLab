// What a run is given, as text for a process of its own (isolated.h): the
// videos, the run's options and the black-bar crops. Its numbers keep every
// digit: JSON's doubles are written to the last bit.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>

#include "vml/options.h"
#include "vml/perceptual.h"
#include "vml/video.h"

namespace vml {

nlohmann::json video_json(const VideoInfo& info);
VideoInfo video_from(const nlohmann::json& text);

nlohmann::json options_json(const VmafOptions& options);
VmafOptions options_from(const nlohmann::json& text);

// A crop, or null for none.
nlohmann::json crop_json(const std::optional<CropBox>& crop);
std::optional<CropBox> crop_from(const nlohmann::json& text);

// A perceptual task's output, sent back from the process that made it: its
// metrics' numbers in a file in `folder` (too many for a message), each to
// the last bit (a NaN survives, which JSON's numbers would not), the rest
// as text.
nlohmann::json perceptual_output_json(const PerceptualTaskOutput& output, const std::filesystem::path& folder);
PerceptualTaskOutput perceptual_output_from(const nlohmann::json& result, const std::filesystem::path& folder);

}  // namespace vml
