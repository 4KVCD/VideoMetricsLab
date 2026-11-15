#include "vml/frame_range.h"

#include <algorithm>
#include <cmath>
#include <tuple>

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

std::pair<long long, std::optional<long long>> compared_frames(const ComparedRange& range, double fps) {
    if (range.in_frames) {
        return {std::max(0LL, range.start), range.end > 0 ? std::optional(range.end) : std::nullopt};
    }
    return {first_frame_of_second(range.start, fps),
            range.end > 0 ? std::optional(first_frame_of_second(range.end, fps)) : std::nullopt};
}

ComparedRange cvvdp_seconds(const ComparedRange& range, double fps, long long video_frames) {
    if (!range.in_frames || range.whole() || fps <= 0) {
        return range.in_frames ? ComparedRange{} : range;
    }
    const auto nearest = [fps](long long frame) { return std::llround(static_cast<double>(frame) / fps); };
    long long start = nearest(std::max(0LL, range.start));
    long long end = 0;
    if (range.end > 0) {
        end = nearest(range.end);
        if (end <= start) {
            // Under a second, rounded to no second: the one with its middle
            // frame, which holds at least half of its frames (frames 12 to
            // 23 at 24 fps: second 0, not second 1, which has none of them).
            start = second_of_frame((std::max(0LL, range.start) + range.end - 1) / 2, fps);
            end = start + 1;
        }
    }
    const long long last_end = end_second_of(video_frames, fps);
    if (last_end > 0) {
        if (end > last_end) {
            end = last_end;
        }
        start = std::min(start, last_end - 1);
    }
    return {start, end, false};
}

FrameRange frame_range(const ComparedRange& compared, double fps, int step, bool looks_around) {
    FrameRange range;
    std::tie(range.start, range.end) = compared_frames(compared, fps);
    if (range.end) {
        range.decode_end = *range.end + (looks_around ? 1 : 0);  // VMAF's motion looks a frame ahead
    }
    long long decode_start = range.start;
    if (looks_around && range.start > 0) {
        // A second before (as much as CVVDP's look back needs), and at
        // least VMAF's frames.
        const long long second_before =
            compared.in_frames ? range.start - static_cast<long long>(std::ceil(fps))
                               : first_frame_of_second(compared.start - 1, fps);
        decode_start = std::max(0LL, std::min(second_before, range.start - most_frames_back));
    }
    const long long every = std::max(1, step);
    range.decode_start = decode_start / every * every;
    return range;
}

FrameRange frame_range(long long first_second, long long end_second, double fps, int step, bool looks_around) {
    return frame_range(ComparedRange{first_second, end_second, false}, fps, step, looks_around);
}

FrameRange union_of(const FrameRange& one, const FrameRange& other) {
    FrameRange both = one;
    both.decode_start = std::min(one.decode_start, other.decode_start);
    both.decode_end = one.decode_end && other.decode_end
                          ? std::optional(std::max(*one.decode_end, *other.decode_end))
                          : std::nullopt;
    return both;
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

namespace {

// The first second that starts at `frame` or after it.
long long second_reaching(long long frame, double fps) {
    long long second = static_cast<long long>(static_cast<double>(frame) / fps);
    while (first_frame_of_second(second, fps) < frame) {
        ++second;
    }
    while (second > 0 && first_frame_of_second(second - 1, fps) >= frame) {
        --second;
    }
    return second;
}

}  // namespace

std::optional<std::pair<long long, long long>> scored_seconds(const MetricProvenance& provenance, double fps) {
    const auto marked = provenance.parameters.find("range_frames");
    if (marked == provenance.parameters.end() || fps <= 0.0) {
        return std::nullopt;
    }
    const nlohmann::json frames = nlohmann::json::parse(marked->second, nullptr, false);
    if (!frames.is_array() || frames.size() != 2 || !frames[0].is_number_integer()) {
        return std::nullopt;
    }
    const long long start = second_reaching(frames[0].get<long long>(), fps);
    return std::pair(start, frames[1].is_number_integer() ? second_reaching(frames[1].get<long long>(), fps) : 0);
}

std::optional<ComparedRange> scored_frames(const MetricProvenance& provenance, double fps) {
    const std::optional<std::pair<long long, long long>> seconds = scored_seconds(provenance, fps);
    if (!seconds) {
        return std::nullopt;
    }
    const nlohmann::json frames = nlohmann::json::parse(provenance.parameters.at("range_frames"));
    const long long start = frames[0].get<long long>();
    const long long end = frames[1].is_number_integer() ? frames[1].get<long long>() : 0;
    if (first_frame_of_second(seconds->first, fps) == start &&
        (end == 0 || first_frame_of_second(seconds->second, fps) == end)) {
        return ComparedRange{seconds->first, seconds->second, false};
    }
    return ComparedRange{start, end, true};
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
