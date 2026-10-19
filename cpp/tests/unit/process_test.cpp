#include "vml/platform/process.h"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

namespace vml::platform {
namespace {

using namespace std::chrono_literals;

// A program's arguments arrive as they were given, whatever they hold
// (spaces, quotes, backslashes, nothing), and its output, error output and
// exit code come back.
TEST(Process, RunsAProgramAndReturnsWhatItWrote) {
    const std::vector<std::string> arguments{
        "plain", "with space", "say \"hi\"", R"(C:\folder\)", R"(back\\"slash)", "", "ünïcödé ∞"};
    std::vector<std::string> command{VML_TEST_CHILD, "echo"};
    command.insert(command.end(), arguments.begin(), arguments.end());

    const ProcessOutput output = run_program(command, 30s);

    std::string expected;
    for (const std::string& argument : arguments) {
        expected += argument + "\n";  // read back as lines, the CR LF dropped
    }
    EXPECT_EQ(output.standard_output, expected);
    EXPECT_EQ(output.standard_error.substr(0, 17), "to standard error");
    EXPECT_EQ(output.exit_code, 3);
    EXPECT_FALSE(output.timed_out);
}

// Of a program that writes a warning for every frame of a long video, only
// the end of its error output is kept, from the start of a line, however
// much it writes.
TEST(Process, KeepsTheEndOfALongErrorOutput) {
    const ProcessOutput output = run_program({VML_TEST_CHILD, "warn", "50000"}, 60s);  // about 1 MB
    EXPECT_LE(output.standard_error.size(), error_tail_bytes);
    EXPECT_GT(output.standard_error.size(), error_tail_bytes / 2);
    EXPECT_TRUE(output.standard_error.starts_with("warning "));
    EXPECT_TRUE(output.standard_error.ends_with("warning 49999 of 50000\nthe last line\n"));
}

TEST(Process, AProgramThatIsNotThereCannotBeStarted) {
    EXPECT_THROW(ChildProcess::start({"vml-no-such-program-anywhere"}), ProcessError);
}

// A program gets its own three handles and none of the app's others. Given
// whatever the app could hand down, a program started while another one's
// pipe was being made took a copy of that pipe, and the other program's
// output then lasted as long as the copy: of two videos compared at once,
// one waited for the other's FFmpeg to end. Here the pipe is the test's,
// made as one that could be handed down.
TEST(Process, AProgramGetsNoneOfTheAppsHandlesButItsOwn) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_end = nullptr;
    HANDLE write_end = nullptr;
    ASSERT_TRUE(CreatePipe(&read_end, &write_end, &inheritable, 0));
    ChildProcess child = ChildProcess::start({VML_TEST_CHILD, "sleep"});
    CloseHandle(write_end);
    // With no other copy of its write end, the pipe has ended.
    DWORD waiting = 0;
    const BOOL peeked = PeekNamedPipe(read_end, nullptr, 0, nullptr, &waiting, nullptr);
    const DWORD error = GetLastError();
    CloseHandle(read_end);
    EXPECT_FALSE(peeked);
    EXPECT_EQ(error, static_cast<DWORD>(ERROR_BROKEN_PIPE));
#else
    int ends[2];
    ASSERT_EQ(pipe(ends), 0);  // without FD_CLOEXEC
    ChildProcess child = ChildProcess::start({VML_TEST_CHILD, "sleep"});
    close(ends[1]);
    // With no other copy of its write end, the pipe has ended: reading it
    // gives nothing at once.
    pollfd read_end{ends[0], POLLIN, 0};
    const int ready = poll(&read_end, 1, 0);
    char byte = 0;
    const ssize_t read_bytes = ready == 1 ? read(ends[0], &byte, 1) : -1;
    close(ends[0]);
    EXPECT_EQ(ready, 1);
    EXPECT_EQ(read_bytes, 0);
#endif
    child.terminate();
}

}  // namespace
}  // namespace vml::platform
