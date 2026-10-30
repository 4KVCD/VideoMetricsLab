// One frame of a video as a PNG, for Video Compare's still frames (the Python
// app's core/frame_extract.py): cropped and scaled as a metric run would
// compare it, and converted for the display (PreviewColorSettings).
#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/geometry.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml {

// The two sides of a comparison.
enum class FrameSide { source, distorted };

// How HDR video is converted for the window, which shows 8-bit SDR
// pictures: by the display's own HDR state, always to SDR at 100 nits
// (assuming PQ when the file has no tags), or not at all.
enum class PreviewColorMode { display_aware, hdr_to_sdr, unmanaged };

// The settings' text for a mode ("display_aware", "hdr_to_sdr",
// "unmanaged"), and the mode for a text (Display-aware for any other).
std::string_view preview_color_mode_value(PreviewColorMode mode);
PreviewColorMode preview_color_mode_from(std::string_view value);

// How Video Compare converts HDR frames for the display: the mode, and the
// display's HDR state where Windows says it (nothing where not).
struct PreviewColorSettings {
    PreviewColorMode mode = PreviewColorMode::display_aware;
    std::optional<bool> display_hdr_enabled;
    std::optional<double> display_sdr_white_nits;

    // The SDR white level, in nits, that HDR is mapped to: Windows' SDR
    // content brightness while its HDR is on (Display-aware), within 40 to
    // 1000 nits; otherwise 100.
    double target_nits() const;
    // What the converted picture depends on, for keeping it: the mode,
    // Windows' HDR state and the target, as the Python app's cache_token.
    std::string cache_token() const;
    bool operator==(const PreviewColorSettings&) const = default;
};

// A frame could not be extracted; the message says why.
class FrameExtractError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Everything needed to show one pair of comparable frames: the files, how
// each is cropped, and which is scaled to the other. Not the scores, on
// purpose: frames are worth comparing before anything has been measured.
struct FrameComparison {
    VideoInfo source_info;
    VideoInfo distorted_info;
    std::optional<CropBox> source_crop;
    std::optional<CropBox> distorted_crop;
    ScaleDirection scale_direction = ScaleDirection::source_to_distorted;
    std::string scale_algorithm = "bicubic";
    std::optional<ResampleTarget> resample_target;
    double fps = 0.0;
    long long frame_count = 0;
    // Auto black bars are chosen but not yet detected: the frames are shown
    // uncropped, and the window says so.
    bool auto_crop_pending = false;

    // The geometry a finished run used.
    static FrameComparison from_result(const ComparisonResult& result);
    bool operator==(const FrameComparison&) const = default;
};

// The size of the pictures the metrics would compare.
Size comparison_dimensions(const FrameComparison& comparison);
// The file decoded for `side`: the source for both sides of a resolution
// test.
const std::filesystem::path& frame_input_path(const FrameComparison& comparison, FrameSide side);
// The details of the video shown on `side`.
const VideoInfo& frame_video_info(const FrameComparison& comparison, FrameSide side);

// FFmpeg's input -ss for `frame`, counted from the video's first frame as
// the metrics count it (the file's start can come before it), an eighth of
// a frame early so that seeking lands on the frame and not the next.
double seek_seconds(const VideoInfo& info, long long frame, double fps);

// How much earlier than they are shown the video's keyframes are decoded, at
// most (s): the largest gap between a packet's presentation and decoding
// times among the video's first 200 packets, read (not decoded) by `ffmpeg`.
// An MP4's or MOV's seek finds keyframes by their decoding time, so -ss can
// land this much late (seek_point). Kept for each file once found; 0
// where FFmpeg cannot read the packets. Throws Cancelled once `cancel` ends
// it.
double keyframe_lead(const std::string& ffmpeg, const std::filesystem::path& path,
                     const CancelToken& cancel = CancelToken());

// Where a command starts, to give the frames from `seconds` (seek_seconds)
// on of a video whose keyframes are decoded up to `lead` seconds before they
// are shown (keyframe_lead): the time for -ss, and how long after it the
// frames to give start (a trim filter's start, first in the filters, by the
// frames' times: before a filter that numbers them anew) where -ss must
// seek earlier.
struct SeekPoint {
    double seek = 0.0;
    double skip = 0.0;
};
SeekPoint seek_point(double seconds, double lead);

// The trim filter that drops the frames before `start` (SeekPoint::skip), or
// nothing for none.
std::optional<std::string> trim_filter(const SeekPoint& start);

// The HDR family a video's transfer tag declares ("HDR10 / PQ", "HLG"), or
// nothing: never guessed from the bit depth.
std::optional<std::string> hdr_kind(const VideoInfo& info);

// Whether scaling `info` (cropped by `crop`) to the output size would change
// its shape: only while its black bars are still to be detected.
bool would_distort(const FrameComparison& comparison, const VideoInfo& info, const std::optional<CropBox>& crop,
                   int output_width, int output_height);

// The filter chain for one side: crop, the metrics' pixel format, scaling,
// square pixels, and the conversion for the display.
std::string frame_filter(const FrameComparison& comparison, FrameSide side, const PreviewColorSettings& colors = {},
                         std::optional<Size> output_size = std::nullopt);

// The FFmpeg command that writes frame `frame` of `side` as a PNG to its
// standard output, its keyframes decoded `lead` seconds before they are
// shown (keyframe_lead). Throws FrameExtractError when the video has no
// usable frame rate, std::invalid_argument for a negative frame.
std::vector<std::string> build_frame_command(const std::string& ffmpeg, const FrameComparison& comparison,
                                             FrameSide side, long long frame,
                                             const PreviewColorSettings& colors = {}, double lead = 0.0);

// Whether a still of `info` is made from the app's decoder's pictures
// (grain_feed.h's windows_from_the_app, where the app can decode it here),
// with its grain as it is scored, rather than by FFmpeg from the file.
bool still_from_the_app(const VideoInfo& info);

// What a still of `info`'s film grain depends on, for keeping the still:
// the decoder that makes it and how it is told the grain.
std::string still_grain(const VideoInfo& info);

// Frame `frame` of `side` as PNG bytes: from the app's decoder's pictures
// (still_from_the_app), else from FFmpeg's decoding of the file
// (build_frame_command; logged for a codec the app decodes elsewhere).
// Throws FrameExtractError (a missing file, FFmpeg's error, no frame there,
// the app's decoder failing on it, a codec only the app decodes that it
// cannot here), and Cancelled once `cancel` ends it.
std::string extract_frame_png(const std::string& ffmpeg, const FrameComparison& comparison, FrameSide side,
                              long long frame, const CancelToken& cancel, const PreviewColorSettings& colors = {});

}  // namespace vml
