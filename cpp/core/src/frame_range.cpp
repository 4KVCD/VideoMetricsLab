#include "vml/frame_range.h"

#include <algorithm>
#include <cmath>
#include <tuple>

#include <nlohmann/json.hpp>

#include "vml/frame_sync.h"
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

namespace {

// A frame's time to the millisecond, as Metric Graphs shows it: frame i's
// i / fps rounded to the nearest (half a millisecond up); and the frames a
// time typed to the millisecond names, those within half a millisecond of
// it, as Metric Graphs reads it. At under 1000 fps a millisecond names one
// frame at most, so a frame's time typed back is the frame. Counted in
// whole numbers at the exact rate (24000/1001, which `fps` is the nearest
// double to): in doubles, a frame exactly at a millisecond (frame 120 at
// 5.005 s) can come out a hair before it.
Rational exact_rate(double fps) {
    return fps > 0 ? limit_denominator(fps) : Rational{0, 1};
}

// The first frame shown at `milliseconds` or after: the first whose time
// is not before it, to the half millisecond (as first_frame_of_second for a
// second's, which is CVVDP's and exact).
long long first_frame_at(long long milliseconds, double fps) {
    const Rational rate = exact_rate(fps);
    if (milliseconds <= 0 || rate.numerator <= 0 || rate.denominator <= 0) {
        return 0;
    }
    // The least i with i / fps >= milliseconds - 0.5 ms:
    // i * 2000 * denominator >= (2 * milliseconds - 1) * numerator.
    const long long per_frame = 2000 * rate.denominator;
    return ((2 * milliseconds - 1) * rate.numerator + per_frame - 1) / per_frame;
}

}  // namespace

ComparedRange time_range_of(long long start_ms, long long end_ms) {
    if (start_ms % 1000 != 0 || end_ms % 1000 != 0) {
        return {start_ms, end_ms, false, true};
    }
    return {start_ms / 1000, end_ms / 1000};
}

ComparedRange in_frames_at(const ComparedRange& range, double fps) {
    if (!range.in_milliseconds || range.in_frames) {
        return range;
    }
    // An end above 0 is a frame above 0 (frame 0 is at 0 ms), never "to the end".
    return {first_frame_at(range.start, fps), range.end > 0 ? first_frame_at(range.end, fps) : 0, true};
}

std::string range_time_text(long long milliseconds) {
    const long long ms = std::max(0LL, milliseconds);
    const auto two = [](long long value) { return (value < 10 ? "0" : "") + std::to_string(value); };
    const long long fraction = ms % 1000;
    return std::to_string(ms / 3600000) + ":" + two(ms / 60000 % 60) + ":" + two(ms / 1000 % 60) + "." +
           (fraction < 100 ? "0" : "") + two(fraction);
}

std::pair<long long, std::optional<long long>> compared_frames(const ComparedRange& given, double fps) {
    const ComparedRange range = in_frames_at(given, fps);
    if (range.in_frames) {
        return {std::max(0LL, range.start), range.end > 0 ? std::optional(range.end) : std::nullopt};
    }
    return {first_frame_of_second(range.start, fps),
            range.end > 0 ? std::optional(first_frame_of_second(range.end, fps)) : std::nullopt};
}

ComparedRange cvvdp_seconds(const ComparedRange& given, double fps, long long video_frames) {
    const ComparedRange range = fps > 0 ? in_frames_at(given, fps) : given;
    if (!range.in_frames || range.whole() || fps <= 0) {
        return range.in_seconds() ? range : ComparedRange{};
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

FrameRange frame_range(const ComparedRange& given, double fps, int step, bool looks_around) {
    const ComparedRange compared = in_frames_at(given, fps);
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

namespace {

// The time of `frame` to the millisecond, as Metric Graphs shows it (30.030
// s for frame 720 at 24000/1001, 4.167 s for frame 100 at 24 fps), which
// first_frame_at reads back as the frame itself.
long long millisecond_of_frame(long long frame, double fps) {
    const Rational rate = exact_rate(fps);
    if (frame <= 0 || rate.numerator <= 0) {
        return 0;
    }
    // frame * 1000 * denominator / numerator, rounded half up.
    return (frame * 2000 * rate.denominator + rate.numerator) / (2 * rate.numerator);
}

// A range of time in milliseconds, either way it is held.
std::pair<long long, long long> milliseconds_of(const ComparedRange& range) {
    const long long scale = range.in_milliseconds ? 1 : 1000;
    return {range.start * scale, range.end * scale};
}

// Whether two ranges in the same unit are the same part of the videos.
bool same_range(const ComparedRange& one, const ComparedRange& other) {
    if (one.whole() || other.whole()) {
        return one.whole() && other.whole();
    }
    if (one.in_frames || other.in_frames) {
        return one.in_frames && other.in_frames && one.start == other.start && one.end == other.end;
    }
    return milliseconds_of(one) == milliseconds_of(other);
}

}  // namespace

std::optional<ComparedRange> range_in_unit(const ComparedRange& range, bool frames, double fps) {
    if (range.whole()) {
        return ComparedRange{0, 0, frames};
    }
    if (range.in_frames == frames) {
        return range;  // time is time, in seconds or milliseconds
    }
    if (!(fps > 0)) {
        return std::nullopt;
    }
    if (frames) {
        const auto [first, end] = compared_frames(range, fps);
        return ComparedRange{first, end.value_or(0), true};
    }
    const long long start = millisecond_of_frame(range.start, fps);
    const long long end = range.end > 0 ? millisecond_of_frame(range.end, fps) : 0;
    if (start % 1000 == 0 && end % 1000 == 0) {
        return ComparedRange{start / 1000, end / 1000, false};
    }
    return ComparedRange{start, end, false, true};
}

ComparedRange switch_range_unit(const ComparedRange& current, const std::optional<ComparedRange>& before, bool frames,
                                double fps) {
    if (before && before->in_frames == frames) {
        const std::optional<ComparedRange> before_now = range_in_unit(*before, current.in_frames, fps);
        if (before_now ? same_range(*before_now, current) : current.whole()) {
            return *before;
        }
    }
    return range_in_unit(current, frames, fps).value_or(ComparedRange{0, 0, frames});
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
