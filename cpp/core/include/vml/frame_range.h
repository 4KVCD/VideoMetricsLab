// The frames a run compares when it is given a range of whole seconds, and
// those it decodes for them.
//
// A range's frames keep their numbers in the whole video (frame N pairs with
// N, counted from each video's first frame: frame_sync.h), so that its
// scores are the same frames' scores of a run over the whole video. Its
// seconds are counted as CVVDP counts them, the reference's frame i being in
// second (long long)(i / fps) (vship_run.cpp's CvvdpLane): a range is then
// whole CVVDP seconds, each scored as in a run over the whole video.
//
// Some metrics look at frames around the one they score: VMAF's motion up to
// 4 frames back (its five-frame window and moving average) and 1 ahead,
// XPSNR 2 back, CVVDP a quarter of a second back (Vship's temporal filter).
// A run with them decodes a second before the range (at least 4 frames) and
// a frame past it, and keeps only the range's scores: its first frames, with
// nothing before them, are scored as no run over the whole video scores
// them. Subsampled runs score the whole video's every n-th frame, so the
// first frame decoded is a multiple of n.
#pragma once

#include <optional>
#include <string>

#include "vml/range_seek.h"
#include "vml/results.h"

namespace vml {

// The second a frame is in, as CVVDP counts them at the reference's `fps`.
long long second_of_frame(long long frame, double fps);

// The first frame of second `second` (or of a later one, where a second has
// no frame of its own: below 1 fps).
long long first_frame_of_second(long long second, double fps);

// The end, in whole seconds, of the ranges that reach the last of `frames`
// frames: the second after the last frame's (0 for no frames). A range's
// second s has a frame of them exactly when s < end_second_of(frames).
long long end_second_of(long long frames, double fps);

struct FrameRange {
    long long start = 0;                  // the first frame compared
    std::optional<long long> end;         // the first frame past them; none: to the end
    long long decode_start = 0;           // the first frame decoded
    std::optional<long long> decode_end;  // the first frame not decoded; none: to the end

    // Whether it is the whole video, as a run without a range compares it.
    bool whole() const { return start == 0 && !end; }
    bool operator==(const FrameRange&) const = default;
};

// The frames of seconds [first_second, end_second) (end_second 0: to the
// end) at the reference's `fps`, for a run scoring every `step`-th frame.
// `looks_around`: the run has a metric that looks at the frames around the
// one it scores (VMAF, XPSNR, CVVDP).
FrameRange frame_range(long long first_second, long long end_second, double fps, int step, bool looks_around);

// FFmpeg's filter at the head of a video's chain that keeps the frames a run
// decodes, counting them as the numbering after it does (frame_sync.h), or
// nothing for a run over the whole video. A video read from a `seek` has no
// pictures to count from: they are kept by their timestamps, which the seek's
// proof ties to their numbers.
std::optional<std::string> range_trim_filter(const FrameRange& range,
                                             const std::optional<RangeSeek>& seek = std::nullopt);

// A ranged run's scores say which frames they are of: their provenance's
// range_frames, [start, end] (`reached`: the first frame past those it
// scored, which CVVDP's last second needs; else the range's end, null: to
// the end). Scores saved before ranges, of a duration limit (the range
// from 0), say nothing and had a frame more for VMAF's look ahead: the
// saved scores tell them apart by it (metric_cache.cpp). Nothing for the
// whole videos.
void mark_range(MetricProvenance& provenance, const FrameRange& range,
                std::optional<long long> reached = std::nullopt);

}  // namespace vml
