// A result's parts as JSON, in the one form the Python app's saved runs
// (run_io) and saved scores' context.json (metric_cache) both write: a field
// added to VideoInfo is saved by both or by neither. Written in Python's key
// order (ordered_json); the saved scores sort them. Ports of models.py's
// crop_to_dict, crop_from_dict, video_info_to_dict and video_info_from_dict,
// and metric_results.py's provenance_to_dict and provenance_from_dict.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>

#include "vml/options.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml {

// A path as Python's str() writes it on this system: backslashes on Windows.
std::string python_path_text(const std::filesystem::path& path);

// A crop, or null for none.
nlohmann::ordered_json crop_to_dict(const std::optional<CropBox>& crop);
// Nothing for null. Throws std::invalid_argument for a crop of another shape.
std::optional<CropBox> crop_from_dict(const nlohmann::json& data);

// A video's details (without format_name and start_offset, which neither
// file keeps).
nlohmann::ordered_json video_info_to_dict(const VideoInfo& info);
// A field missing from an older file loads as its default. Throws
// std::invalid_argument naming a required field that is missing, as
// Python's KeyError says it ("'path'").
VideoInfo video_info_from_dict(const nlohmann::json& data);

nlohmann::ordered_json provenance_to_dict(const MetricProvenance& provenance);
// Throws std::invalid_argument for parameters that are not an object, and
// naming a missing field as Python's KeyError says it.
MetricProvenance provenance_from_dict(const nlohmann::json& data);

// A resolution test's target ({"width", "label"}), or null for none.
nlohmann::ordered_json resample_target_to_dict(const std::optional<ResampleTarget>& target);

// The scale direction as saved ("source_to_distorted"), and read back.
// Throws std::invalid_argument for another text, as Python's ValueError
// says it.
const char* scale_direction_value(ScaleDirection direction);
ScaleDirection scale_direction_from(const std::string& value);

}  // namespace vml
