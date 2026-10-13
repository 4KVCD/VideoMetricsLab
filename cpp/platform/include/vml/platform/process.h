// Starting programs (FFmpeg, ffprobe, the metric tools) and reading what they
// write. A port of vmaf_app/core/proc.py and the parts of
// process_control.py the command line needs.
//
// Every program is started without a console window. terminate() ends it
// and every program it started. On Windows they also end with the app,
// however the app ends: each program is in a job object that is closed with
// the app. (On Linux and macOS that is still to do.)
#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vml::platform {

// A program could not be started.
class ProcessError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ProcessOptions {
    // The folder the program runs in; empty is ours.
    std::filesystem::path working_directory;
};

// A program run to its end: its exit code and everything it wrote, as the
// bytes it wrote (UTF-8, for FFmpeg's tools).
struct ProcessOutput {
    int exit_code = -1;
    std::string standard_output;
    std::string standard_error;
    bool timed_out = false;  // it was ended for taking too long
};

// A running program, its standard output read line by line here and its
// standard error collected in the background. Ended (with everything it
// started) when this is destroyed.
class ChildProcess {
public:
    // Starts `arguments[0]` (a path, or a name found on the PATH) with the
    // rest as its arguments. Throws ProcessError if it cannot be started.
    static ChildProcess start(const std::vector<std::string>& arguments, const ProcessOptions& options = {});

    ChildProcess(ChildProcess&&) noexcept;
    ChildProcess& operator=(ChildProcess&&) noexcept;
    ~ChildProcess();

    // The next line of standard output without its line ending, or nothing
    // once the program has closed it.
    std::optional<std::string> read_line();

    // Waits for the program to end, and returns its exit code.
    int wait();

    // Waits up to `timeout`; true if the program has ended.
    bool wait_for(std::chrono::milliseconds timeout);

    // Ends the program, and every program it started.
    void terminate();

    // Everything the program wrote to standard error; complete once it has
    // ended (after wait()).
    std::string standard_error();

    int process_id() const;

private:
    struct State;
    explicit ChildProcess(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

// Runs a program to its end and returns what it wrote. A program still
// running after `timeout` is ended, and the result says so. Throws
// ProcessError if it cannot be started.
ProcessOutput run_program(const std::vector<std::string>& arguments, std::chrono::milliseconds timeout,
                          const ProcessOptions& options = {});

}  // namespace vml::platform
