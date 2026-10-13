#include "vml/platform/process.h"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

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

TEST(Process, AProgramRunningTooLongIsEnded) {
    const auto started = std::chrono::steady_clock::now();
    const ProcessOutput output = run_program({VML_TEST_CHILD, "sleep"}, 300ms);
    EXPECT_TRUE(output.timed_out);
    EXPECT_LT(std::chrono::steady_clock::now() - started, 20s);  // ended, not waited for
}

TEST(Process, TerminateEndsARunningProgram) {
    ChildProcess child = ChildProcess::start({VML_TEST_CHILD, "sleep"});
    EXPECT_FALSE(child.wait_for(100ms));
    child.terminate();
    EXPECT_TRUE(child.wait_for(20s));
}

TEST(Process, AProgramThatIsNotThereCannotBeStarted) {
    EXPECT_THROW(ChildProcess::start({"vml-no-such-program-anywhere"}), ProcessError);
}

}  // namespace
}  // namespace vml::platform
