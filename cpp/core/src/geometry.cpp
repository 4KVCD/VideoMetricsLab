#include "vml/geometry.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// Why the two videos' frame rates or durations do not allow a comparison.
std::optional<std::string> timeline_problem(const VideoInfo& source, const VideoInfo& distorted,
                                            double duration_limit) {
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
        if (duration_limit <= 0) {
            const double frame_duration = 1.0 / std::max({source.fps, distorted.fps, 1.0});
            if (std::abs(source.duration - distorted.duration) > std::max(0.1, 2 * frame_duration)) {
                return "Durations do not match (" + format_fixed(source.duration, 3) + " vs " +
                       format_fixed(distorted.duration, 3) +
                       " seconds). Set a duration limit within both files if comparing only their common opening "
                       "segment.";
            }
        } else if (std::min(source.duration, distorted.duration) + 0.1 < duration_limit) {
            return "The duration limit extends beyond the end of one of the videos.";
        }
    }
    return std::nullopt;
}

// How the Python app names a raw stream's codec in its message.
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
    std::string upper = codec;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return upper;
}

}  // namespace

Size content_size(const VideoInfo& info, const std::optional<CropBox>& crop) {
    return crop ? Size{crop->w, crop->h} : Size{info.width, info.height};
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
    // The half height rounded as Python's round() rounds it, a half to even.
    const double half = static_cast<double>(width) * size.second / size.first / 2;
    return std::max(2, static_cast<int>(std::nearbyint(half)) * 2);
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

std::optional<std::string> untimed_pair_problem(const VideoInfo& source, const VideoInfo& distorted) {
    if (!source.is_raw_stream() && !distorted.is_raw_stream()) {
        return std::nullopt;
    }
    return pair_problem(source, distorted, 0.0);
}

std::optional<std::string> pair_problem(const VideoInfo& source, const VideoInfo& distorted, double duration_limit) {
    std::optional<std::string> problem = timeline_problem(source, distorted, duration_limit);
    const VideoInfo* raw = source.is_raw_stream() ? &source : distorted.is_raw_stream() ? &distorted : nullptr;
    if (!problem || raw == nullptr) {
        return problem;
    }
    // A raw stream's frame rate and length are FFmpeg's guesses, so the
    // message says how to give it a real frame rate. It names mkvmerge, not
    // FFmpeg: FFmpeg cannot copy a raw stream with B-frames into Matroska.
    return platform::path_text(raw->path.filename()) + " is a raw " + codec_label(raw->codec_name) +
           " stream with no timestamps, so its frame rate is unknown. Put it in a container with its frame rate "
           "first, e.g. mkvmerge -o video.mkv --default-duration 0:120fps <file> (MKVToolNix), with the video's "
           "own frame rate.";
}

}  // namespace vml
