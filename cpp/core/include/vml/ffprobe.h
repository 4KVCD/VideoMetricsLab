// Reading a video's details with ffprobe. A port of vmaf_app/core/ffprobe.py.
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "vml/video.h"

namespace vml {

// ffprobe could not read a file; the message says why, as the Python app's
// does.
class ProbeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The video's details, read by running `ffprobe` (the program to run) on
// `path`. Throws ProbeError when they cannot be read.
VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe);

// The video's details in ffprobe's JSON output for `path`
// (-print_format json -show_format -show_streams). Throws ProbeError when
// the output cannot be read or has no video stream.
VideoInfo video_info_from_probe(const std::filesystem::path& path, std::string_view probe_json);

}  // namespace vml
