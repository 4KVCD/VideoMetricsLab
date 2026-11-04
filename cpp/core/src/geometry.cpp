#include "vml/geometry.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "vml/frame_range.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// The even number nearest, a half to even, as Python's round() rounds a half.
int nearest_even(double value) { return static_cast<int>(std::nearbyint(value / 2)) * 2; }

// Whether two videos last as long, to within a tenth of a second or two
// frames: compared to their end, neither has frames the other lacks.
bool durations_match(const VideoInfo& source, const VideoInfo& distorted) {
    const double frame_duration = 1.0 / std::max({source.fps, distorted.fps, 1.0});
    return std::abs(source.duration - distorted.duration) <= std::max(0.1, 2 * frame_duration);
}

// Why the two videos' frame rates or durations do not allow a comparison of
// seconds [range_start, range_end) (range_end 0: to the end).
std::optional<std::string> timeline_problem(const VideoInfo& source, const VideoInfo& distorted,
                                            long long range_start, long long range_end) {
    if (range_end > 0 && range_end <= range_start) {
        return "The range ends before it starts.";
    }
    if ((range_start > 0 || range_end > 0) && (source.fps <= 0 || distorted.fps <= 0)) {
        return "A range needs the videos' frame rates, which are not known.";  // its seconds count frames
    }
    if (range_end > 0 &&
        first_frame_of_second(range_start, source.fps) >= first_frame_of_second(range_end, source.fps)) {
        return "The range contains no frame of the videos.";  // below 1 fps, a second may have none
    }
    if (source.is_variable_frame_rate() || distorted.is_variable_frame_rate()) {
        return "Variable-frame-rate video is not supported safely yet. Convert both videos "
               "to the same constant frame rate before comparing them.";
    }
    // Half the gap between a whole frame rate and its NTSC counterpart (24
    // and 23.976 are a thousandth apart), so those two are told apart, while
    // the rates two containers give one video (24000/1001 against 2997/125)
    // still match.
    const double fps_tolerance = std::max(0.005, std::max(source.fps, distorted.fps) * 0.0005);
    if (std::abs(source.fps - distorted.fps) > fps_tolerance) {
        return "Frame rates do not match (" + format_fixed(source.fps, 3) + " vs " + format_fixed(distorted.fps, 3) +
               " fps).";
    }
    if (source.duration > 0 && distorted.duration > 0) {
        // Counted in frames, as the run counts them (frame_range.h): a
        // duration ends where the last frame's display does, so 288 frames
        // at 24000/1001 fps last 12.012 s and second 12 has none of them. An
        // end within the shorter video's last, partial second compares to
        // its end (0-13 for a 300-frame encode, 12.5 s), where every route
        // stops: the two videos' frames are paired until either ends.
        const long long shorter_frames =
            std::min(source.estimated_frame_count(), distorted.estimated_frame_count());
        const long long last_end = end_second_of(shorter_frames, source.fps);
        if (range_start > 0 && range_start >= last_end) {
            return "The range starts after the end of one of the videos.";
        }
        if (range_end <= 0) {
            if (!durations_match(source, distorted)) {
                return "Durations do not match (" + format_fixed(source.duration, 3) + " vs " +
                       format_fixed(distorted.duration, 3) +
                       " seconds). Set a range within both files if comparing only a part they share.";
            }
        } else if (range_end > last_end) {
            return "The range extends beyond the end of one of the videos.";
        }
    }
    return std::nullopt;
}

}  // namespace

std::string codec_label(const std::string& codec) {
    if (codec == "h264") {
        return "H.264";
    }
    if (codec == "mpeg2video") {
        return "MPEG-2";
    }
    if (codec == "mpeg4") {
        return "MPEG-4";
    }
    if (codec == "mjpeg") {
        return "MJPEG";
    }
    return ascii_upper(codec);
}

Size content_size(const VideoInfo& info, const std::optional<CropBox>& crop) {
    return crop ? Size{crop->w, crop->h} : info.frame_size();
}

std::optional<std::string> size_change_scale(const VideoInfo& info, const std::string& scale_algorithm) {
    if (!info.scaled_to) {
        return std::nullopt;
    }
    // In the video's own pixel format: the conversion to the analysis format
    // stays where it is for every video, after the crop.
    return "scale=" + std::to_string(info.scaled_to->first) + ":" + std::to_string(info.scaled_to->second) +
           ":flags=" + scale_algorithm + ",format=" + info.pix_fmt;
}

std::pair<VideoInfo, std::optional<CropBox>> at_own_size(const VideoInfo& info, const std::optional<CropBox>& crop) {
    VideoInfo own = info;
    own.scaled_to.reset();
    if (!info.scaled_to || !crop || info.scaled_to->first <= 0 || info.scaled_to->second <= 0) {
        return {own, crop};
    }
    // Each edge at its place in the video's own pixels, the nearest even one.
    const auto edge = [](int at, int own_length, int scaled_length) {
        return 2 * static_cast<int>(std::lround(static_cast<double>(at) * own_length / scaled_length / 2));
    };
    const auto [scaled_width, scaled_height] = *info.scaled_to;
    const int left = edge(crop->x, info.width, scaled_width);
    const int top = edge(crop->y, info.height, scaled_height);
    const int right = std::min(info.width, edge(crop->x + crop->w, info.width, scaled_width));
    const int bottom = std::min(info.height, edge(crop->y + crop->h, info.height, scaled_height));
    return {own, CropBox{std::max(2, right - left), std::max(2, bottom - top), left, top}};
}

Size compared_dimensions(const VideoInfo& source, const VideoInfo& distorted, ScaleDirection scale_direction,
                         const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop) {
    const Size distorted_size = content_size(distorted, distorted_crop);
    const Size source_size = content_size(source, source_crop);
    if (source_size == distorted_size) {
        return distorted_size;
    }
    return scale_direction == ScaleDirection::distorted_to_source ? source_size : distorted_size;
}

Size analysis_dimensions(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                         const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop) {
    return compared_dimensions(source, distorted, options.scale_direction, source_crop, distorted_crop);
}

int resample_down_height(int width, Size size) {
    return std::max(2, nearest_even(static_cast<double>(width) * size.second / size.first));
}

int resample_smallest_width(Size size) {
    constexpr int smallest_side = 16;
    return std::max(smallest_side, nearest_even(static_cast<double>(smallest_side) * size.first / size.second));
}

int resample_largest_width(Size size) { return (size.first - 1) / 2 * 2; }

Size resample_size_for_width(int width, Size size) {
    // min after max, not std::clamp: the two limits can cross (resample_smallest_width).
    const int at_least_smallest = std::max(nearest_even(width), resample_smallest_width(size));
    const int kept = std::min(at_least_smallest, resample_largest_width(size));
    return {kept, resample_down_height(kept, size)};
}

Size resample_size_for_height(int height, Size size) {
    return resample_size_for_width(nearest_even(static_cast<double>(height) * size.first / size.second), size);
}

std::pair<int, int> sar_fraction(const std::string& sar) {
    if (sar.empty() || sar == "N/A" || sar == "0:1") {
        return {1, 1};
    }
    const std::size_t colon = sar.find(':');
    if (colon == std::string::npos) {
        return {1, 1};
    }
    const std::optional<long long> numerator = python_int(std::string_view(sar).substr(0, colon));
    const std::optional<long long> denominator = python_int(std::string_view(sar).substr(colon + 1));
    if (!numerator || !denominator || *numerator <= 0 || *denominator <= 0) {
        return {1, 1};
    }
    return {static_cast<int>(*numerator), static_cast<int>(*denominator)};
}

double display_aspect_ratio(const VideoInfo& info, const std::optional<CropBox>& crop) {
    const auto [width, height] = content_size(info, crop);
    if (height <= 0) {
        return 0.0;
    }
    const auto [numerator, denominator] = sar_fraction(info.sar);
    // As Python computes it: whole products, then one division.
    return static_cast<double>(static_cast<long long>(width) * numerator) /
           static_cast<double>(static_cast<long long>(height) * denominator);
}

std::optional<std::string> shape_problem(const VideoInfo& source, const VideoInfo& distorted,
                                         const std::optional<CropBox>& source_crop,
                                         const std::optional<CropBox>& distorted_crop) {
    const double source_shape = display_aspect_ratio(source, source_crop);
    const double distorted_shape = display_aspect_ratio(distorted, distorted_crop);
    if (source_shape <= 0 || distorted_shape <= 0) {
        return std::nullopt;  // nothing meaningful to compare
    }
    // Wide enough for the rounding even sizes force (1918x1080 against
    // 1920x1080), far tighter than a real mismatch.
    if (std::abs(source_shape - distorted_shape) <= 0.01 * std::max(source_shape, distorted_shape)) {
        return std::nullopt;
    }
    const auto describe = [](const VideoInfo& info, const std::optional<CropBox>& crop, double shape) {
        const std::string size = crop ? std::to_string(crop->w) + "x" + std::to_string(crop->h) + " (cropped)"
                                      : std::to_string(info.frame_size().first) + "x" +
                                            std::to_string(info.frame_size().second);
        return size + ", " + format_fixed(shape, 3) + ":1";
    };
    return "The two videos are different shapes after cropping: source " +
           describe(source, source_crop, source_shape) + " versus distorted " +
           describe(distorted, distorted_crop, distorted_shape) +
           ". Scoring them would stretch one to fit the other and the result would be meaningless. If one is "
           "letterboxed, set black-bar handling to 'Auto-detect' so the bars are removed before comparison.";
}

std::optional<std::string> untimed_pair_problem(const VideoInfo& source, const VideoInfo& distorted) {
    if (!source.is_raw_stream() && !distorted.is_raw_stream()) {
        return std::nullopt;
    }
    return pair_problem(source, distorted, 0, 0);
}

std::optional<std::string> pair_problem(const VideoInfo& source, const VideoInfo& distorted, long long range_start,
                                        long long range_end) {
    std::optional<std::string> problem = timeline_problem(source, distorted, range_start, range_end);
    const auto untimed = [](const VideoInfo& info) { return info.is_raw_stream() && !info.timed_raw; };
    const VideoInfo* raw = untimed(source) ? &source : untimed(distorted) ? &distorted : nullptr;
    if (!problem || raw == nullptr) {
        return problem;
    }
    // An untimed raw stream's frame rate and length are FFmpeg's guesses, so
    // the message says how to give it a real frame rate. It names mkvmerge,
    // not FFmpeg: FFmpeg cannot copy a raw stream with B-frames into Matroska.
    // (The command line's --fps times one: its problems are real ones.)
    return platform::path_text(raw->path.filename()) + " is a raw " + codec_label(raw->codec_name) +
           " stream with no timestamps, so its frame rate is unknown. Put it in a container with its frame rate "
           "first, e.g. mkvmerge -o video.mkv --default-duration 0:120fps <file> (MKVToolNix), with the video's "
           "own frame rate.";
}

}  // namespace vml
