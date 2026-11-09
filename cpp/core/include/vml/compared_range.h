// The part of the videos a comparison compares: whole seconds of the
// reference, or frames (VmafOptions' range, a saved comparison's recipe).
#pragma once

#include <optional>
#include <utility>

namespace vml {

struct ComparedRange {
    // Seconds as CVVDP counts them (frame_range.h), or frames numbered from
    // each video's first (frame_sync.h): from `start` to `end`, the first
    // second or frame past the range; an end of 0 compares to the videos'
    // end. Both 0: the whole videos.
    long long start = 0;
    long long end = 0;
    bool in_frames = false;

    bool whole() const { return start <= 0 && end <= 0; }
    bool operator==(const ComparedRange&) const = default;
};

// The frames of `range` at the reference's `fps`: [first, end), no end: to
// the videos' end.
std::pair<long long, std::optional<long long>> compared_frames(const ComparedRange& range, double fps);

// The whole seconds CVVDP scores of `range`, which scores a second at a time:
// a range of frames rounded to the nearest seconds, at least one, ending no
// later than the second after the last of `video_frames` (when known, above
// 0). A range of seconds is its own.
ComparedRange cvvdp_seconds(const ComparedRange& range, double fps, long long video_frames);

}  // namespace vml
