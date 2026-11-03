// How every metric pairs the two videos' frames. A port of
// vmaf_app/core/frame_sync.py.
//
// The test video's frame N is compared with the source's frame N, counting
// from each video's first frame, until either video ends. FFmpeg's
// two-input filters pair frames by their timestamps, so each video's frames
// are first restamped with their numbers (numbering_filters): frame N gets
// the time of N frames of the test video, in both videos. A range keeps the
// same frames of both (frame_range.h: a trim before the numbering).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "vml/video.h"

namespace vml {

// A frame rate as an exact fraction (24000/1001, not 23.976...).
struct Rational {
    long long numerator = 0;
    long long denominator = 1;
    bool operator==(const Rational&) const = default;
};

// FFmpeg's frame sync options for its two-input filters: the comparison ends
// with the shorter video, and no video's last frame is repeated past its
// end.
std::vector<std::string> framesync_opts();

// The fraction nearest `value` with a denominator up to `max_denominator`,
// exactly as Python's Fraction(value).limit_denominator(max_denominator).
Rational limit_denominator(double value, long long max_denominator = 1'000'000);

// The video's frame rate as an exact fraction, which numbered frames are
// timed by; 25 for a video that gives none.
Rational frame_rate(const VideoInfo& info);

// FFmpeg filters that restamp each frame by its number: frame N gets the
// time N / `rate` seconds, in whole microseconds. They end each video's chain
// before a filter pairs the two.
std::vector<std::string> numbering_filters(Rational rate);

// How many frames, numbered at `rate`, a duration limit keeps: those whose
// time is below it. Nothing when there is no limit. For scores saved before
// ranges, whose runs cut the videos so (metric_cache.cpp's fit_seconds).
std::optional<long long> frames_within(const std::optional<std::string>& duration_limit, Rational rate);

// A duration as -t takes it ("30.042") in whole microseconds, read as FFmpeg
// reads it: exactly, with any digits past the sixth decimal dropped. Throws
// std::invalid_argument for text that is not one.
long long duration_microseconds(const std::string& duration);

// (a * b + add) / c, rounded down, exactly: the product of two 64-bit
// numbers can need 128 bits (a long limit at a frame rate with a big
// numerator), so it is formed in two halves and divided bit by bit. Throws
// std::overflow_error for a result beyond 64 bits.
unsigned long long multiply_add_divide(unsigned long long a, unsigned long long b, unsigned long long add,
                                       unsigned long long c);

}  // namespace vml
