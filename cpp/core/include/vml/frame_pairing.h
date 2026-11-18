// Whether a run's frames may have been paired out of step. Frames are paired
// by number: frame N of the test video with frame N of the reference. When
// one of the two has fewer frames than the other in the same length, the
// pairs part from there on, and every later pair is of two different
// pictures. A remux of a Blu-ray's joined clips can leave small gaps in its
// timestamps, say, which an encode at a constant frame rate fills with a
// repeated frame: the two then play in step, but the encode has more frames.
// The scores fall from the first such gap on, with nothing else saying why.
//
// A run that scored fewer pairs than both videos' lengths hold is the sign,
// which costs nothing to see (frames_short_of_lengths). It is not proof: two
// videos with the same gaps (an encode at a variable frame rate keeps them)
// have the same frames, paired in step. Counting each video's frames tells
// (count_frames_short): at once where both containers state their counts,
// which a run always reads when it sees the sign; otherwise by reading the
// whole of a file, which a run does only when asked (Settings::
// check_frame_counts, the command line's --frame-counts check).
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "vml/cancel.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml {

// The frame pairs `result` scored: of the frame metric whose scores reach
// furthest, the frame after its last scored one counted in its own steps (a
// metric scored every nth frame scores the frame n after its last one
// next). 0 without frame scores.
long long scored_frame_pairs(const ComparisonResult& result);

// For a run of the two whole videos (`to_the_end`: from the first frame to
// the end), the shortfall when it scored at least 2 frame pairs fewer than
// both videos' lengths hold (their durations times the frame rate; one frame
// is the rounding of a duration). Nothing otherwise: a range, a resolution
// test (whose test video is made from the source, frame for frame), or a
// video whose length is not known.
std::optional<FramesShort> frames_short_of_lengths(const ComparisonResult& result, bool to_the_end);

// Counts a video's frames: nothing where it cannot.
using FrameCounter = std::function<std::optional<long long>(const VideoInfo& video)>;

// `short_of` (frames_short_of_lengths) with the two videos counted by
// `count`, both at once (FramesShort::in_step when the counts are the same);
// uncounted when either count is missing, or far from what its video's
// length holds (more than 1% and 10 frames off: a count of something else,
// such as an interlaced video's fields).
FramesShort count_frames_short(const FramesShort& short_of, const VideoInfo& source, const VideoInfo& test,
                               const FrameCounter& count);

// What `short_of` means, in plain sentences: that the frames may be out of
// step and how to tell (uncounted), that they are (counted, different), or
// that they line up (counted, the same).
std::string frames_short_note(const FramesShort& short_of);

// frames_short_note as a command-line note and a log line (lowercase first,
// no full stop); empty when the frames line up.
std::string frames_short_text(const FramesShort& short_of);

// A video's "frame_pairing" in a results JSON (the window's export and the
// command line's --json), for every result: "status" "may_be_out_of_step"
// (uncounted), "out_of_step" or "in_step" (counted), "no_mismatch_seen" (as
// many pairs as the lengths hold) or "not_checked" (a range, a resolution
// test, a length not known, no frame scores); a "note" in plain words that
// says what it means and, where it may be out of step, how to tell; the pairs
// compared, the frames the shorter length holds, and each video's count
// (null uncounted).
nlohmann::ordered_json frame_pairing_json(const ComparisonResult& result);

// A FrameCounter with `ffprobe`: the count a file's container states where it
// can be trusted (trusted_frame_count), at once; with `read_whole_files`,
// for a file that states none, its shown video packets, counted by reading
// it through (count_shown_video_packets: minutes for a film on a network
// drive). Nothing where it has none; Cancelled once `cancel` is.
FrameCounter container_frame_counter(std::string ffprobe, CancelToken cancel, bool read_whole_files);

}  // namespace vml
