// How every metric pairs the two videos' frames. A port of
// vmaf_app/core/frame_sync.py.
//
// The test video's frame N is compared with the source's frame N, counting
// from each video's first frame, until either video ends. FFmpeg's
// two-input filters pair frames by their timestamps, so each video's frames
// are first restamped with their numbers (numbering_filters): frame N gets
// the time of N frames of the test video, in both videos. A duration limit
// (an output's -t, duration_args) then keeps the same frames of both.
//
// With the environment variable VML_FRAME_PAIRING=timestamp, frames are
// paired by timestamp, as the app did before.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/video.h"

namespace vml {

// A frame rate as an exact fraction (24000/1001, not 23.976...).
struct Rational {
    long long numerator = 0;
    long long denominator = 1;
    bool operator==(const Rational&) const = default;
};

// The environment variable that brings back the old pairing by timestamp.
inline constexpr std::string_view pairing_variable = "VML_FRAME_PAIRING";

// Whether frames are paired the old way, by timestamp.
bool by_timestamp();

// FFmpeg's frame sync options for its two-input filters: the comparison ends
// with the shorter video, and no video's last frame is repeated past its
// end (when pairing by timestamp, the nearest timestamp is also taken).
std::vector<std::string> framesync_opts();

// The fraction nearest `value` with a denominator up to `max_denominator`,
// exactly as Python's Fraction(value).limit_denominator(max_denominator).
Rational limit_denominator(double value, long long max_denominator = 1'000'000);

// The video's frame rate as an exact fraction, which numbered frames are
// timed by; 25 for a video that gives none.
Rational frame_rate(const VideoInfo& info);

// FFmpeg filters that restamp each frame by its number: frame N gets the
// time N / `rate` seconds, in whole microseconds.
std::vector<std::string> numbering_filters(Rational rate);

// The filters at the end of each video's chain before a filter pairs the
// two: numbering_filters, or when pairing by timestamp each video's times
// started at 0.
std::vector<std::string> pairing_filters(Rational rate);

// An FFmpeg output's options for a duration limit, given as -t takes it
// ("30.042"); none without a limit.
std::vector<std::string> duration_args(const std::optional<std::string>& duration_limit);

// How many frames, numbered at `rate`, a duration limit keeps: those whose
// time is below it. Nothing when there is no limit.
std::optional<long long> frames_within(const std::optional<std::string>& duration_limit, Rational rate);

}  // namespace vml
