// A run's Cancel reaches every program it runs (cancel.h), as the Python
// app's process handles do (tests/test_process_control.py), here with real
// programs: the test child, sleeping for a minute.
#include "vml/cancel.h"

#include <gtest/gtest.h>

#include <chrono>
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

}  // namespace
}  // namespace vml
