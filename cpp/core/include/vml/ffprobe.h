// Reading a video's details with ffprobe. A port of vmaf_app/core/ffprobe.py.
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "vml/cancel.h"
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
// The same, ended by `cancel`: ffprobe on a large file over a slow or
// network disk can take many seconds, and closing the window or stopping a
// scan must not wait for it. Throws Cancelled once cancelled.
VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe, const CancelToken& cancel);

// `info`, a raw stream's details (VideoInfo::is_raw_stream), timed at `fps`
// frames a second, the rate the command line's --fps gives: a raw stream
// records none, and FFmpeg reads it as 25. Its frames are counted with
// `ffprobe` (without decoding them), and its length and bitrate follow from
// them. Throws ProbeError when they cannot be counted, Cancelled once
// cancelled.
VideoInfo timed_raw_stream(VideoInfo info, double fps, const std::string& ffprobe,
                           const CancelToken& cancel = CancelToken());

// The video's details in ffprobe's JSON output for `path`
// (-print_format json -show_format -show_streams). Throws ProbeError when
// the output cannot be read or has no video stream.
VideoInfo video_info_from_probe(const std::filesystem::path& path, std::string_view probe_json);

}  // namespace vml
