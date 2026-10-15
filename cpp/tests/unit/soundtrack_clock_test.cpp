// The soundtrack's clock (soundtrack_clock.h): the Python app's test of the
// same behaviour (tests/test_locked_presentation.py, its soundtrack part).
// The sound device's clock moves in steps of 10 ms while a frame of 120 fps
// video lasts 8.3 ms: one frame in five was skipped until it was carried on
// between its steps.
#include "vml/soundtrack_clock.h"

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <vector>

namespace vml {
namespace {

long long frame_120(double ms) { return std::llround(ms * 120 / 1000); }

TEST(SoundtrackClock, CarriesOnBetweenItsStepsSoThatNoFrameIsSkipped) {
    double now = 100.0;  // s
    double device_ms = 1000;
    SoundtrackClock clock;
    clock.set_playing(true);

    // Carried on between its 10 ms steps: 120 fps video, no frame skipped
    // in a second of the view's 4 ms ticks.
    std::set<long long> frames;
    for (int tick = 0; tick < 250; ++tick) {
        now = 100 + tick * 0.004;
        device_ms = 1000 + tick * 4 / 10 * 10;
        frames.insert(frame_120(*clock.poll(device_ms, now)));
        ASSERT_EQ(clock.reading_ms(), device_ms) << "what falling behind is judged by is carried on";
    }
    std::set<long long> every;
    for (long long frame = 120; frame <= 240; ++frame) {
        every.insert(frame);  // to 1996 ms: frame 240 is due at 1995.8
    }
    EXPECT_EQ(frames, every);

    // Never further than a step, and never back while playing.
    now += 1.0;  // the device's clock stopped a second
    const double stopped = *clock.poll(device_ms, now);
    EXPECT_EQ(stopped, device_ms + 10);
    device_ms -= 5;
    EXPECT_EQ(*clock.poll(device_ms, now), stopped);

    // Paused, then playing again a second later: no step at once.
    clock.set_playing(false);
    EXPECT_EQ(*clock.poll(device_ms, now), device_ms);
    now += 1.0;
    clock.set_playing(true);
    EXPECT_EQ(*clock.poll(device_ms, now), device_ms);
    now += 0.004;
    EXPECT_NEAR(*clock.poll(device_ms, now), device_ms + 4, 1e-6);

    // Polled once a frame, as the view's ticks come when a frame is due:
    // each frame once. Carried on from when a poll saw it step, it fell up
    // to a poll behind, then caught up at once: a frame twice, then two at
    // once.
    clock.set_playing(false);
    clock.set_playing(true);
    const double start_s = now;
    const double start_ms = device_ms;
    std::vector<long long> shown;
    for (int tick = 0; tick < 120; ++tick) {
        now = start_s + (tick + 0.1) / 120;
        device_ms = start_ms + static_cast<long long>((now - start_s) * 1000) / 10 * 10;
        shown.push_back(frame_120(*clock.poll(device_ms, now)));
    }
    for (std::size_t index = 1; index < shown.size(); ++index) {
        EXPECT_EQ(shown[index], shown[0] + static_cast<long long>(index)) << "tick " << index;
    }

    // Counted from the video's first frame; nothing without a reading.
    clock.set_offset(32);
    clock.set_playing(false);
    EXPECT_EQ(*clock.poll(device_ms, now), device_ms - 32);
    EXPECT_FALSE(clock.poll(std::nullopt, now));
    EXPECT_FALSE(clock.reading_ms());
}

}  // namespace
}  // namespace vml
