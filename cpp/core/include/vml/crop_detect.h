// Detecting black bars (letterbox and pillarbox) with FFmpeg's cropdetect
// filter. A port of vmaf_app/core/crop_detect.py.
//
// cropdetect with reset=0 reports the tightest box that holds every
// non-black pixel it has seen, so one bright pixel of noise near an edge can
// keep any crop from being found. So five short windows are sampled across
// the middle of the video, and the most common box wins. A video's windows
// run one after another; the two videos of a comparison are detected at the
// same time. The answer for a file is kept, in this process and between
// sessions (black_bars.json beside the saved scores, the Python app's file).
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "vml/cancel.h"
#include "vml/video.h"

namespace vml {

// Black bars could not be detected; the message says why, as the Python
// app's does.
class CropDetectError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Keeps the answers between sessions in `store` (a folder can move, so it is
// asked for each time); nothing keeps them only in this process.
void set_crop_store(std::function<std::filesystem::path()> store);

// The box that cuts the video's black bars. `hwaccel` is the decoder the
// run will use for this video (only how fast the answer comes depends on
// it). Throws CropDetectError when it cannot be found, and Cancelled.
CropBox detect_crop(const std::string& ffmpeg, const VideoInfo& info, const CancelToken& cancel,
                    const std::optional<std::string>& hwaccel);

// Both videos' boxes, detected at the same time.
std::pair<CropBox, CropBox> detect_pair(const std::string& ffmpeg, const VideoInfo& source,
                                        const VideoInfo& distorted, const CancelToken& cancel,
                                        const std::optional<std::string>& source_hwaccel,
                                        const std::optional<std::string>& distorted_hwaccel);

// The two videos' boxes, made to cover the same picture where the two show
// the same picture at one size or two (an encode's soft bar edge can put its
// box a row from the source's): the boxes' overlap, in each video's own
// pixels, on even samples. Boxes that agree, or that are not the same
// picture, are kept.
std::pair<std::optional<CropBox>, std::optional<CropBox>> common_picture(const VideoInfo& source,
                                                                         const VideoInfo& distorted,
                                                                         std::optional<CropBox> source_crop,
                                                                         std::optional<CropBox> distorted_crop);

}  // namespace vml
