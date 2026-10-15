// The clock native playback times its frames by (the Python app's
// SingleSoundtrack.poll): the soundtrack's position, read from the sound
// device's clock, which moves in steps of its period (10 ms), carried on
// between steps so that 120 fps video skips no frame.
#pragma once

#include <optional>
#include <utility>

namespace vml {

class SoundtrackClock {
public:
    // The position (ms from the video's first frame) for the device clock's
    // `reading` (ms on the file's timeline, or nothing while it has none) at
    // `now` (seconds, a steady clock). While playing, it is carried on from
    // the last one given by the time that has passed, kept between the
    // reading and one step past it, and never goes back.
    std::optional<double> poll(std::optional<double> reading, double now);
    // The clock's own last reading (ms from the video's first frame), not
    // carried on: what falling behind is judged by.
    std::optional<double> reading_ms() const { return reading_ms_; }
    // Playing or paused; a change starts the carrying on again from now.
    void set_playing(bool playing);
    bool playing() const { return playing_; }
    // After a seek: nothing carried on from before it.
    void restart();
    // Where the video's first frame is on the file's timeline (ms).
    void set_offset(int offset_ms) { offset_ms_ = offset_ms; }
    int offset_ms() const { return offset_ms_; }

private:
    int offset_ms_ = 0;
    bool playing_ = false;
    std::optional<double> reading_ms_;
    std::optional<double> last_reading_;
    double step_ms_ = 10.0;
    std::optional<std::pair<double, double>> given_;  // (ms given, when: s)
};

}  // namespace vml
