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
// A video of a codec the app's own decoder decodes (app_decoders.h: VVC, by
// VVdeC) is sampled from that decoder's pictures (samples_through_the_app).
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

// The store's file, kept with the saved scores.
inline constexpr std::string_view black_bars_file = "black_bars.json";

// Forgets the answers this process holds, and reads the store again when
// next asked: clearing the saved results, which deletes the store too, then
// starts afresh.
void forget_crop_answers();

// The box that cuts the video's black bars. `hwaccel` is the decoder the
// run will use for this video (only how fast the answer comes depends on
// it). A video whose size changes (VideoInfo::scaled_to) is sampled scaled
// with `scale_algorithm`, as the run scales it, and its box is not kept.
// Throws CropDetectError when it cannot be found, Cancelled, and
// VideoSizeChanged (size_changes.h) when FFmpeg rebuilt its filters partway
// through a sample.
CropBox detect_crop(const std::string& ffmpeg, const VideoInfo& info, const CancelToken& cancel,
                    const std::optional<std::string>& hwaccel, const std::string& scale_algorithm);

// Whether a video's black bars are sampled from the pictures the app's own
// decoder makes, fed to FFmpeg's cropdetect (grain_feed.h), rather than from
// FFmpeg decoding the file: a codec app_decoders.h lists (FFmpeg's vvc fails
// on VVenC's --fga 3 encodes at 10 bits, so their bars were not found). Not a
// raw stream (sampled from bytes, not seeked to). A window the app cannot
// feed is sampled by FFmpeg where FFmpeg decodes the codec, and fails where
// it does not.
bool samples_through_the_app(const VideoInfo& info);

// Both videos' boxes, detected at the same time.
std::pair<CropBox, CropBox> detect_pair(const std::string& ffmpeg, const VideoInfo& source,
                                        const VideoInfo& distorted, const CancelToken& cancel,
                                        const std::optional<std::string>& source_hwaccel,
                                        const std::optional<std::string>& distorted_hwaccel,
                                        const std::string& scale_algorithm);

// The two videos' boxes, made to cover the same picture where the two show
// the same picture at one size or two (an encode's soft bar edge can put its
// box a row from the source's): the boxes' overlap, in each video's own
// pixels (as scaled_to, where set), on even samples. Boxes that agree, or that are not the same
// picture, are kept.
std::pair<std::optional<CropBox>, std::optional<CropBox>> common_picture(const VideoInfo& source,
                                                                         const VideoInfo& distorted,
                                                                         std::optional<CropBox> source_crop,
                                                                         std::optional<CropBox> distorted_crop);

}  // namespace vml
