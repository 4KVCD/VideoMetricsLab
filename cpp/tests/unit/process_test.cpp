#include "vml/platform/process.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "test_videos.h"
#include "vml/platform/pipe.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
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

// A frame much bigger than the pipe it comes through is read whole, byte for
// byte, as FFmpeg writes it, and the end of the stream after it.
TEST(Process, ReadsAFrameBiggerThanItsPipeWhole) {
    constexpr std::size_t frame = 3'000'000;
    ProcessOptions options;
    options.output_pipe_bytes = 1024 * 1024;
    ChildProcess process = ChildProcess::start({VML_TEST_CHILD, "data", std::to_string(frame)}, options);
    const std::atomic<bool> stop{false};
    std::vector<std::uint8_t> data(frame + 1);  // one more than comes: read_full stops at the end
    ASSERT_EQ(process.read_full(data.data(), data.size(), stop), frame);
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < frame; ++i) {
        wrong += data[i] != i % 251 ? 1 : 0;
    }
    EXPECT_EQ(wrong, 0u);
    EXPECT_EQ(process.wait(), 0);
}

// A program gets its own three handles and none of the app's others. Given
// whatever the app could hand down, a program started while another one's
// pipe was being made took a copy of that pipe, and the other program's
// output then lasted as long as the copy: of two videos compared at once,
// one waited for the other's FFmpeg to end. Here the pipe is the test's,
// made as one that could be handed down.
#ifndef _WIN32
// A descriptor a program inherits is its own: what it writes arrives, and the
// reader sees the end when it ends. This process keeps no copy, whether the
// program started or not.
TEST(Process, AProgramInheritsADescriptorThatEndsWithIt) {
    InboundPipe pipe("inherited", 64 * 1024);
    const int writer = pipe.take_writer();
    ASSERT_GE(writer, 0);
    EXPECT_EQ(pipe.path(), "pipe:" + std::to_string(writer));
    EXPECT_EQ(pipe.take_writer(), -1);  // once only
    ProcessOptions options;
    options.inherited_descriptors = {writer};
    ChildProcess process =
        ChildProcess::start({"/bin/sh", "-c", "printf x >&" + std::to_string(writer)}, options);
    char bytes[2] = {};
    EXPECT_EQ(pipe.read(bytes, 1), 1u);
    EXPECT_EQ(bytes[0], 'x');
    EXPECT_EQ(pipe.read(bytes, 1), 0u);
    EXPECT_EQ(fcntl(writer, F_GETFD), -1);
    EXPECT_EQ(process.wait(), 0);

    InboundPipe never_started("never", 64 * 1024);
    ProcessOptions failing;
    failing.inherited_descriptors = {never_started.take_writer()};
    const int number = failing.inherited_descriptors[0];
    EXPECT_THROW(ChildProcess::start({"/nonexistent/program"}, failing), ProcessError);
    EXPECT_EQ(fcntl(number, F_GETFD), -1);
    EXPECT_EQ(never_started.read(bytes, 1), 0u);
}

// A reader whose writer will not come finds the end, as often as it is said.
TEST(Process, APipeWithNoWriterEnds) {
    InboundPipe pipe("no writer", 64 * 1024);
    pipe.no_writer_will_come();
    pipe.no_writer_will_come();
    EXPECT_EQ(pipe.take_writer(), -1);
    char byte = 0;
    EXPECT_EQ(pipe.read(&byte, 1), 0u);
}
#endif

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

#ifndef _WIN32
// The program a program started is ended with it, however that ends -- a
// crash or a Force Quit too, which nothing in it sees -- as Windows' jobs end
// theirs: the test's child starts a program (its group its own) that holds a
// FIFO open, and is killed; the FIFO's end says that program ended too.
TEST(Process, WhatAProgramStartedEndsWithItHoweverItEnds) {
    const test::Folder folder;
    const std::string fifo = (folder / "held").string();
    ASSERT_EQ(mkfifo(fifo.c_str(), 0600), 0);
    const int reader = open(fifo.c_str(), O_RDONLY | O_NONBLOCK);
    ASSERT_GE(reader, 0);
    ChildProcess child = ChildProcess::start({VML_TEST_CHILD, "spawn-hold", fifo});
    // Read returns nothing until the grandchild has written its byte (no
    // writer yet: 0; one with nothing written: EAGAIN), then the byte.
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    char byte = 0;
    while (read(reader, &byte, 1) != 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(10ms);
    }
    ASSERT_EQ(byte, 'h');
    child.terminate();  // SIGKILL: the child cannot end what it started
    child.wait();
    // The grandchild held the FIFO open: EAGAIN while it runs, 0 once it is gone.
    ssize_t got = -1;
    while ((got = read(reader, &byte, 1)) < 0 && errno == EAGAIN && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(10ms);
    }
    close(reader);
    EXPECT_EQ(got, 0);
}

// A program suspended (Pause) is not taken for ended: a wait that does not
// wait leaves it be, and one that waits lets it be resumed meanwhile (when
// the waiting thread is in its wait by then, as it mostly is).
TEST(Process, ASuspendedProgramIsNotEndedByWaitingForIt) {
    ChildProcess child = ChildProcess::start({VML_TEST_CHILD, "sleep"});
    child.suspend();
    siginfo_t info{};
    ASSERT_EQ(waitid(P_PID, static_cast<id_t>(child.process_id()), &info, WSTOPPED | WNOWAIT), 0);  // stopped now
    EXPECT_FALSE(child.wait_for(0ms));
    EXPECT_EQ(kill(child.process_id(), 0), 0);  // still there
    std::atomic<int> code{0};
    std::thread waiting([&] { code = child.wait(); });
    child.resume();     // not held up by the wait
    child.terminate();  // which then ends
    waiting.join();
    EXPECT_EQ(code.load(), -SIGKILL);
}
#endif

}  // namespace
}  // namespace vml::platform
