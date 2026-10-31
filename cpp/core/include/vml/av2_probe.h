// An AV2 video's details (VideoInfo), which ffprobe cannot read (av2_stream.h):
// the size, depth, colour and film grain of its first picture, as dav2d
// decodes it, and its frames' count and rate, from its temporal units. The
// decoding runs in a process of its own (isolated.h, task "av2-probe"), as
// the app's other native decoding does.
#pragma once

#include <filesystem>

#include <nlohmann/json.hpp>

#include "vml/cancel.h"
#include "vml/video.h"

namespace vml {

// Whether `path` holds AV2 in a form the app reads (IVF, Matroska or WebM,
// or Section 5 OBUs: av2_stream.h).
bool is_av2_file(const std::filesystem::path& path);

// `path`'s details: codec "av2", format "ivf", "matroska" or "obu" (a
// Section 5 stream, which has no times: a raw stream, read at the frame rate
// it is given).
// Throws ProbeError (ffprobe.h) when it cannot be read or decoded, and
// Cancelled.
VideoInfo probe_av2(const std::filesystem::path& path, const CancelToken& cancel);

// The probing process's side (isolated.cpp's task "av2-probe"): {"path"} in,
// the details out (av2_info_from).
nlohmann::json av2_probe_task(const nlohmann::json& request);

// The VideoInfo av2_probe_task's result describes.
VideoInfo av2_info_from(const std::filesystem::path& path, const nlohmann::json& result);

}  // namespace vml
