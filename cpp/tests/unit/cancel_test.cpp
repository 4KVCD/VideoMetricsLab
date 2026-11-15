// A run's Cancel reaches every program it runs (cancel.h), as the Python
// app's process handles do (tests/test_process_control.py), here with real
// programs: the test child, sleeping for a minute.
#include "vml/cancel.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <mutex>
#include <thread>

#include "vml/platform/process.h"

namespace vml {
namespace {

platform::ChildProcess sleeping() { return platform::ChildProcess::start({VML_TEST_CHILD, "sleep"}); }

// A paused program writes nothing, so the code reading it cannot look at the
// flag: Cancel ends the run's programs, which ends the read (the Python app
// waited for good there before release/v2.1 c21eebf). A program attached
// after the Cancel ends at once: a fallback started then would otherwise run
// on, and the run wait for it.
TEST(CancelToken, CancelEndsEveryProgramOfTheRunEvenPausedAndAnyAttachedAfter) {
    CancelToken cancel;
    platform::ChildProcess first = sleeping();
    platform::ChildProcess second = sleeping();
    const CancelToken::Attached first_attached = cancel.attach(first);
    const CancelToken::Attached second_attached = cancel.attach(second);
    cancel.pause();
    std::thread reader([&first] {  // as a run reads its FFmpeg
        while (first.read_line()) {
        }
    });
    cancel.cancel();
    platform::ChildProcess late = sleeping();
    const CancelToken::Attached late_attached = cancel.attach(late);
    for (platform::ChildProcess* program : {&first, &second, &late}) {
        // Ended now, not after its minute: not ended within 30 s is a
        // failure, and the program is then ended here so that the test ends.
        const bool ended = program->wait_for(std::chrono::seconds(30));
        EXPECT_TRUE(ended);
        if (!ended) {
            program->terminate();
        }
    }
    reader.join();
}

// The time a run spent paused, for the results file: every pause in all, the
// one still on counted to now, a second Pause or Resume changing nothing.
TEST(CancelToken, CountsTheTimeSpentPaused) {
    double clock = 100.0;
    CancelToken token([&clock] { return clock; });
    EXPECT_EQ(token.now(), 100.0);
    EXPECT_EQ(token.paused_seconds(), 0.0);
    token.pause();
    clock = 103.0;
    token.pause();
    EXPECT_EQ(token.paused_seconds(), 3.0);
    clock = 104.0;
    token.resume();
    clock = 110.0;
    token.resume();
    EXPECT_EQ(token.paused_seconds(), 4.0);
    token.pause();
    clock = 112.5;
    EXPECT_EQ(token.paused_seconds(), 6.5);
}

// A wait for a lock another caller holds (a GPU probe, which can take many
// seconds) answers Cancel: it throws, leaving the lock to its holder.
TEST(CancelToken, AWaitForALockEndsWhenTheRunIsCancelled) {
    std::timed_mutex mutex;
    std::promise<void> locked;
    std::promise<void> release;
    std::thread holder([&] {
        const std::lock_guard lock(mutex);
        locked.set_value();
        release.get_future().wait();
    });
    locked.get_future().wait();
    CancelToken token;
    token.cancel();
    EXPECT_THROW(lock_unless_cancelled(mutex, token), Cancelled);
    release.set_value();
    holder.join();
    // Free, it is taken whatever the run's state.
    EXPECT_TRUE(lock_unless_cancelled(mutex, token).owns_lock());
}

}  // namespace
}  // namespace vml
