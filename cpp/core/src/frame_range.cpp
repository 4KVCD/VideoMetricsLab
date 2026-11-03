#include "vml/frame_range.h"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

#include "vml/number_format.h"

namespace vml {

namespace {

// VMAF's motion looks this many frames back at most: its five-frame window
// (two back) and its moving average (the frame before's motion).
constexpr long long most_frames_back = 4;

}  // namespace

long long second_of_frame(long long frame, double fps) {
    // As CvvdpLane computes it, in doubles: the range's seconds are its.
    return static_cast<long long>(static_cast<double>(frame) / fps);
}

long long first_frame_of_second(long long second, double fps) {
    if (second <= 0 || fps <= 0) {
        return 0;
    }
    // second * fps, then a step either way to where second_of_frame changes:
    // the division's rounding can put a frame at a second's very edge in the
    // second before (24000/1001 fps).
    auto frame = static_cast<long long>(std::ceil(static_cast<double>(second) * fps));
    while (frame > 0 && second_of_frame(frame - 1, fps) >= second) {
        --frame;
    }
    while (second_of_frame(frame, fps) < second) {
        ++frame;
    }
    return frame;
}

long long end_second_of(long long frames, double fps) {
    return frames > 0 && fps > 0 ? second_of_frame(frames - 1, fps) + 1 : 0;
}

FrameRange frame_range(long long first_second, long long end_second, double fps, int step, bool looks_around) {
    FrameRange range;
    range.start = first_frame_of_second(first_second, fps);
    if (end_second > 0) {
        range.end = first_frame_of_second(end_second, fps);
        range.decode_end = *range.end + (looks_around ? 1 : 0);  // VMAF's motion looks a frame ahead
    }
    long long decode_start = range.start;
    if (looks_around && range.start > 0) {
        decode_start = std::max(0LL, std::min(first_frame_of_second(first_second - 1, fps),
                                              range.start - most_frames_back));
    }
    const long long every = std::max(1, step);
    range.decode_start = decode_start / every * every;
    return range;
}

std::optional<std::string> range_trim_filter(const FrameRange& range, const std::optional<RangeSeek>& seek) {
    if (range.decode_start == 0 && !range.decode_end) {
        return std::nullopt;
    }
    const auto at_picture = [&seek](const char* name, long long number) {
        return seek ? std::string(name) + "=" + format_fixed(seconds_before_picture(*seek, number), 6)
                    : std::string(name) + "_frame=" + std::to_string(number);
    };
    std::string trim = "trim=" + at_picture("start", range.decode_start);
    if (range.decode_end) {
        trim += ":" + at_picture("end", *range.decode_end);
    }
    return trim;
}

void mark_range(MetricProvenance& provenance, const FrameRange& range, std::optional<long long> reached) {
    if (range.whole()) {
        return;
    }
    const std::optional<long long> end = reached ? reached : range.end;
    provenance.parameters["range_frames"] =
        nlohmann::json::array({range.start, end ? nlohmann::json(*end) : nlohmann::json(nullptr)}).dump();
}

}  // namespace vml
