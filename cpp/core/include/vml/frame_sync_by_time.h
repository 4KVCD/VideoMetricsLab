// Pairs the two videos' frames by timestamp: the old way of pairing, kept
// beside the pairing by frame number (frame_sync.h) and used only when the
// environment variable VML_FRAME_PAIRING is "timestamp" (by_timestamp). A
// port of vmaf_app/core/frame_sync_by_time.py.
//
// It works out, from the two videos' timestamps, the frame pairs FFmpeg's
// libvmaf filter would compare: FFmpeg's frame sync (libavfilter's
// framesync.c) with the options framesync_opts gives when pairing by
// timestamp (shortest=1, repeatlast=0, ts_sync_mode=nearest):
//
// - Each input's timestamps start at 0 (setpts=PTS-STARTPTS). The source's
//   are rescaled to the test video's time base, rounded to nearest, the
//   frame sync's time base being its only synchronising input's.
// - Each frame of the test video (the "main" input) gives one pair, when it
//   is the earliest frame either input has next.
// - The source moves on to its next frame when that frame is the earliest
//   next frame, or ("nearest") when it is strictly nearer to the time being
//   decided than the source's current frame.
// - When the test video ends, so does the comparison. When the source ends,
//   its end is stamped one tick after its last frame, and the comparison
//   ends at the first test frame at or after that time.
#pragma once

#include <functional>
#include <optional>
#include <string>

#include "vml/cancel.h"
#include "vml/frame_sync.h"

namespace vml {

// `value` from time base `from` to time base `to`, as FFmpeg's av_rescale_q
// gives it: exactly, rounded to nearest with halves away from zero.
long long rescale(long long value, Rational from, Rational to);

// FFmpeg's output option -t `text` ("30.042") in `time_base`, as FFmpeg's
// trim filter holds it: a picture is kept while its time from the first
// picture is below the result.
long long duration_in(const std::string& text, Rational time_base);

// The two videos' frames in pairs, by timestamp (the Python app's
// frame_sync_by_time.frame_pairs), one pair at a time.
class TimedPairs {
public:
    // A frame of an input: whatever number the caller knows it by (a ring
    // slot), and its timestamp in the input's time base.
    struct Frame {
        int frame = 0;
        long long pts = 0;
    };

    // A pair: the test frame, the source frame (nothing only where FFmpeg's
    // filter would have none, the source having no frame by the time of the
    // test video's first; callers refuse that), and the pair's time in the
    // test video's time base, counted from its first frame.
    struct Pair {
        int test = 0;
        std::optional<int> source;
        long long time = 0;
    };

    // Each input's next frame in display order, or nothing at its end.
    using Pull = std::function<std::optional<Frame>()>;
    // Called with each frame once no later pair can use it.
    using Release = std::function<void(int frame)>;

    TimedPairs(Pull test, Pull source, Rational test_time_base, Rational source_time_base, Release release_test,
               Release release_source);
    ~TimedPairs();
    TimedPairs(const TimedPairs&) = delete;
    TimedPairs& operator=(const TimedPairs&) = delete;

    // The next pair, or nothing once the comparison has ended. A pair's
    // frames stay valid until the next call (or close), which releases
    // those no later pair uses.
    std::optional<Pair> next();

    // Releases the frames still held; the pairing is then over. Safe to
    // call twice.
    void close();

private:
    // One input of the frame sync: its current frame and its next one, with
    // their times in the frame sync's time base.
    struct Input {
        Pull pull;
        Release release;
        std::function<long long(long long)> to_sync_time;
        bool nearest = false;
        std::optional<long long> first;  // the first timestamp (STARTPTS)
        bool ended = false;              // all of its frames have been pulled
        bool done = false;               // its end has been taken (framesync's STATE_EOF)
        std::optional<int> frame;
        std::optional<long long> pts;  // nothing until a frame is taken
        std::optional<int> next;
        long long next_pts = 0;
        bool has_next = false;

        void fill();
        void take();
        void drop();
    };

    Input test_;
    Input source_;
    bool over_ = false;
};

class GpuFrameStream;

// Scores two videos decoded in the app pair by pair, paired by timestamp as
// FFmpeg's overlay pairs them (the Python app's
// frame_sync_by_time.decoded_pairs): while the test frame's time from its
// first frame is below `duration_limit` (-t's text) in the test video's
// time base, where FFmpeg's trim filter cuts. Both streams must already be
// started. `score` is called with each pair's slots (the test video's, then
// the source's); a slot is given back to its decoder once no later pair
// uses it, so a source frame may be scored with several test frames.
// Progress goes to `on_progress` every quarter of a second. Throws
// GpuDecodeFailedError, and Cancelled.
void score_timed_pairs(GpuFrameStream& test, GpuFrameStream& reference,
                       const std::optional<std::string>& duration_limit, long long total_frames,
                       const std::function<void(int, int)>& score,
                       const std::function<void(long long, long long, double)>& on_progress,
                       const CancelToken& cancel);

}  // namespace vml
