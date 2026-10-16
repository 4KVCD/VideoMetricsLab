#include "vml/soundtrack_clock.h"

#include <algorithm>

namespace vml {

std::optional<double> SoundtrackClock::poll(std::optional<double> reading, double now) {
    if (!reading) {
        reading_ms_.reset();
        return std::nullopt;
    }
    // Don't carry on from the moment a poll saw the clock step: playback then
    // falls up to a frame behind, and catches up by showing two frames at
    // once.
    const double position = *reading;
    reading_ms_ = position - offset_ms_;
    if (playing_ && last_reading_ && position - *last_reading_ >= 5 && position - *last_reading_ < step_ms_) {
        step_ms_ = position - *last_reading_;
    }
    last_reading_ = position;
    if (!playing_) {
        given_.reset();
        return position - offset_ms_;
    }
    double given = position;
    if (given_) {
        const auto [given_ms, given_at] = *given_;
        const double carried = given_ms + (now - given_at) * 1000;
        given = std::max(given_ms, std::min(std::max(carried, position), position + step_ms_));
    }
    given_ = std::pair(given, now);
    return given - offset_ms_;
}

void SoundtrackClock::set_playing(bool playing) {
    if (playing != playing_) {
        // Carried on from the position given before a pause, it would jump
        // a whole step.
        restart();
    }
    playing_ = playing;
}

void SoundtrackClock::restart() {
    last_reading_.reset();
    given_.reset();
}

}  // namespace vml
