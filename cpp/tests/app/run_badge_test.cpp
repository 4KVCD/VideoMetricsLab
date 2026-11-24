// The progress button's state (run_badge.h, #226): Ready before a run, its
// progress during it, Done or Cancelled after it until the ticks change.

#include <gtest/gtest.h>

#include "run_badge.h"

namespace vml::app {
namespace {

using Phase = RunBadge::Phase;
using Tone = BadgeLook::Tone;

const RunTicks two_videos{{"a", {"vmaf", "psnr"}}, {"b", {"vmaf", "psnr"}}};

// Done stays while the ticks are those the run ended with, and is Ready
// again when a video's tick or a metric's changes, or a video is added; a
// new run is its progress, whatever came before.
TEST(RunBadge, DoneStaysUntilTheTicksChange) {
    RunBadge badge;
    EXPECT_EQ(badge.phase(), Phase::ready);
    EXPECT_FALSE(badge.ticks_now(two_videos));  // Ready stays Ready

    badge.start();
    EXPECT_EQ(badge.phase(), Phase::running);
    EXPECT_FALSE(badge.ticks_now({}));  // a run is not interrupted by ticks

    badge.end(false, {2, 2, 0, 1.0}, "Done in 0:00:39.", two_videos);
    EXPECT_EQ(badge.phase(), Phase::done);
    EXPECT_FALSE(badge.ticks_now(two_videos));
    EXPECT_EQ(badge.phase(), Phase::done);
    EXPECT_EQ(badge.ended_tooltip(), "Done in 0:00:39.");

    RunTicks fewer_metrics = two_videos;
    fewer_metrics[1].second.pop_back();
    EXPECT_TRUE(badge.ticks_now(fewer_metrics));
    EXPECT_EQ(badge.phase(), Phase::ready);

    for (const RunTicks& changed : {RunTicks{two_videos[0]}, RunTicks{two_videos[0], two_videos[1], {"c", {"vmaf"}}},
                                    RunTicks{two_videos[1], two_videos[0]}}) {
        badge.start();
        badge.end(true, {2, 1, 0, 0.5}, {}, two_videos);
        EXPECT_EQ(badge.phase(), Phase::cancelled);
        EXPECT_TRUE(badge.ticks_now(changed));
        EXPECT_EQ(badge.phase(), Phase::ready);
    }
}

// Each phase's text, accessible state, bar and tone, as the proposal's
// table has them.
TEST(RunBadge, LookOfEachPhase) {
    const BadgeLook ready = badge_look(Phase::ready, false, {});
    EXPECT_EQ(ready.text, "Ready");
    EXPECT_EQ(ready.tone, Tone::normal);

    const RunCounts half{2, 1, 0, 0.5};
    const BadgeLook running = badge_look(Phase::running, false, half);
    EXPECT_EQ(running.text, "1 of 2 · 50 %");
    EXPECT_EQ(running.state, "1 of 2, 50 %");
    EXPECT_DOUBLE_EQ(running.fraction, 0.5);
    EXPECT_EQ(badge_look(Phase::running, true, half).text, "Paused · 1 of 2 · 50 %");

    const BadgeLook done = badge_look(Phase::done, false, {2, 2, 0, 0.98});
    EXPECT_EQ(done.text, "Done · 2 of 2");
    EXPECT_DOUBLE_EQ(done.fraction, 1.0);  // full, whatever the last tick said
    EXPECT_EQ(done.tone, Tone::normal);

    const BadgeLook failed = badge_look(Phase::done, false, {3, 3, 1, 1.0});
    EXPECT_EQ(failed.text, "Done · 1 failed");
    EXPECT_EQ(failed.state, "Done, 1 failed");
    EXPECT_EQ(failed.tone, Tone::error);

    const BadgeLook cancelled = badge_look(Phase::cancelled, false, {2, 1, 0, 0.62});
    EXPECT_EQ(cancelled.text, "Cancelled · 1 of 2");
    EXPECT_DOUBLE_EQ(cancelled.fraction, 0.62);  // where it stopped
    EXPECT_EQ(cancelled.tone, Tone::muted);
}

TEST(RunBadge, ReadyTooltipCountsTheTicks) {
    EXPECT_EQ(ready_tooltip({}), "Ready: no test videos ticked.");
    EXPECT_EQ(ready_tooltip({two_videos[0]}), "Ready: 1 test video ticked, 2 metrics.");
    EXPECT_EQ(ready_tooltip(two_videos), "Ready: 2 test videos ticked, 2 metrics each.");
    EXPECT_EQ(ready_tooltip({two_videos[0], {"c", {"vmaf"}}}), "Ready: 2 test videos ticked, 1 to 2 metrics.");
}

}  // namespace
}  // namespace vml::app
