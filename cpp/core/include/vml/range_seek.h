// Starting to decode near a range late in a video, instead of decoding and
// passing over every picture before it.
//
// A range's frames are counted from each video's first shown picture
// (frame_range.h), so a seek must give exactly those pictures with exactly those
// numbers. That is proved of a file before it is used: its pictures' timestamps
// must lie on a grid of its frame rate from the first to the last with none
// missing (plan_range_seek, gpu_frames.h), so a picture's number follows from its
// timestamp. A stream that then shows otherwise refuses (SeekRefused), before
// anything is scored, and the pass is made again without seeking.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vml/frame_sync.h"

namespace vml {

// Where a stream of a ranged run starts reading, and how its pictures are numbered.
// The timestamps are the file's own (FFmpeg's -copyts): a picture's stays what it
// is in a run from the start.
struct RangeSeek {
    double start = 0;         // the seek, in seconds of the video
    long long first_pts = 0;  // the first shown picture's timestamp: picture number 0
    Rational time_base;
    Rational frame_rate;

    bool operator==(const RangeSeek&) const = default;
};

// Thrown by a seeked stream whose pictures are not what the seek promised, or
// which failed before reaching the range. Nothing has been scored yet: the pass
// is made again (turn_off_range_seek), never by the GPU decoding fallbacks, as
// the decoder did nothing wrong.
class SeekRefused : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Whether a range starting at `decode_start` is far enough into a video of this
// rate to be worth the checks a seek takes.
bool seek_is_worthwhile(long long decode_start, Rational frame_rate);

// Where to seek for that range: two seconds of pictures early, which also keeps
// the leading pictures of an open GOP's keyframe out of the range.
double seek_seconds(long long decode_start, Rational frame_rate);

// Whether a file of `file_bytes` is cheap enough to read through to count its
// packets, against decoding the `decode_start` pictures before the range.
bool counting_packets_pays(long long file_bytes, long long decode_start);

// A picture's number, from its timestamp; -1 when it is off the grid.
long long picture_number_at(long long pts, long long first_pts, Rational time_base, Rational frame_rate);

// Whether `pictures` shown pictures fill the grid from the first timestamp to the last.
bool pictures_fill_grid(long long first_pts, long long last_pts, long long pictures, Rational time_base,
                        Rational frame_rate);

// FFmpeg's options before a video's -i, with its reading from the seek added: every
// picture after the place it lands on comes through, on the file's own timestamps
// (range_trim_filter then keeps the range's).
std::vector<std::string> with_seek(std::vector<std::string> input_options, const std::optional<RangeSeek>& seek);

// Where a filter cuts, in the file's seconds, to keep picture `number` and not the
// one before it: half a picture early, which a container's rounded timestamps
// cannot cross.
double seconds_before_picture(const RangeSeek& seek, long long number);

}  // namespace vml
