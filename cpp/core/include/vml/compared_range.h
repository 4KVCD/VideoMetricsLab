// The part of the videos a comparison compares: whole seconds of the
// reference, milliseconds of it, or frames (VmafOptions' range, a saved
// comparison's recipe).
#pragma once

#include <optional>
#include <string>
#include <utility>

namespace vml {

struct ComparedRange {
    // Seconds as CVVDP counts them (frame_range.h), frames numbered from
    // each video's first (frame_sync.h), or milliseconds (a time that is not
    // whole seconds): from `start` to `end`, the first second, frame or
    // millisecond past the range; an end of 0 compares to the videos' end.
    // Both 0: the whole videos.
    long long start = 0;
    long long end = 0;
    bool in_frames = false;
    // Milliseconds: the frames whose times (frame i's: i / fps) are in
    // [start, end), as a range of whole seconds counts them. Never set for
    // whole seconds, which keep the identity they have always had.
    bool in_milliseconds = false;

    bool whole() const { return start <= 0 && end <= 0; }
    // Whole seconds, each scored as in a run over the whole video (CVVDP's
    // too); the other units are frames, CVVDP's rounded to whole seconds.
    bool in_seconds() const { return !in_frames && !in_milliseconds; }
    bool operator==(const ComparedRange&) const = default;
};

// A range of milliseconds as the range of frames it covers at the
// reference's `fps`; any other range as it is.
ComparedRange in_frames_at(const ComparedRange& range, double fps);

// A time of a range in milliseconds as H:MM:SS.mmm (0:01:02.500).
std::string range_time_text(long long milliseconds);

// The frames of `range` at the reference's `fps`: [first, end), no end: to
// the videos' end.
std::pair<long long, std::optional<long long>> compared_frames(const ComparedRange& range, double fps);

// The whole seconds CVVDP scores of `range`, which scores a second at a time:
// a range of frames rounded to the nearest seconds (one rounded to none: the
// second with its middle frame), ending no later than the second after the
// last of `video_frames` (when known, above 0). A range of seconds is its
// own.
ComparedRange cvvdp_seconds(const ComparedRange& range, double fps, long long video_frames);

}  // namespace vml
