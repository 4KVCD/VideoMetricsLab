// Starting programs (FFmpeg, ffprobe, the metric tools) and reading what they
// write. A port of vmaf_app/core/proc.py and the parts of
// process_control.py the command line needs.
//
// Every program is started without a console window. terminate() ends it
// and every program it started. On Windows they also end with the app,
// however the app ends: each program is in a job object that is closed with
// the app. (On Linux and macOS that is still to do.)
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vml::platform {

// A program could not be started.
class ProcessError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// How much of a program's standard error is kept: its end. FFmpeg can write
// a warning for every frame of a long video, and what is used of it is its
// last lines (an error's last 25, cropdetect's last one), so keeping it all
// would cost memory for nothing.
inline constexpr std::size_t error_tail_bytes = 64 * 1024;

// Adds what a program wrote to standard error to `kept`, which holds its
// end: once `kept` reaches twice error_tail_bytes it is cut to the last
// ones, so the cutting is rare.
void keep_error_tail(std::string& kept, std::string_view written);

// The end of `kept`: at most its last error_tail_bytes, from the start of a
// line (cut mid-line only when one line is longer than that).
std::string error_tail(const std::string& kept);

// `text` with every line ending made "\n" ("\r\n", and a lone "\r"), as
// Python reads a program's output in text mode: Windows programs end their
// lines with "\r\n", which would otherwise stay inside the app's messages.
std::string universal_newlines(std::string_view text);

struct ProcessOptions {
    // The folder the program runs in; empty is ours.
    std::filesystem::path working_directory;
    // How much the pipe of the program's standard output holds before the
    // program waits for this process to read it; 0 is the system's size.
    // On Linux it is asked for, and a size past the system's limit
    // (/proc/sys/fs/pipe-max-size, 1 MiB unless raised) is refused, which
    // leaves the system's size; macOS keeps its own size.
    std::size_t output_pipe_bytes = 0;
    // Called with each line of the program's standard error as it comes,
    // without its line ending, on the thread that reads it. Its end is kept
    // for standard_error() all the same.
    std::function<void(std::string_view)> on_error_line = nullptr;
};

// A program run to its end: its exit code, everything it wrote to standard
// output and the end of its standard error (its lines ended with "\n"), as
// the bytes it wrote (UTF-8, for FFmpeg's tools).
struct ProcessOutput {
    int exit_code = -1;
    std::string standard_output;
    std::string standard_error;
    bool timed_out = false;  // it was ended for taking too long
    bool stopped = false;    // it was ended because the caller's `stop` said so
};

// An exit code as text, as Python's returncode shows it: on Windows the
// unsigned code the process ended with (3221225477 for an access violation,
// the number people search for, not -1073741819); elsewhere as wait() gives
// it, minus the signal for a program a signal ended.
std::string exit_code_text(int exit_code);

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

    // Reads up to `bytes` of standard output into `into`, for a program
    // that writes data rather than lines: how many were read, 0 once the
    // program has closed it.
    std::size_t read(void* into, std::size_t bytes);

    // Why standard output ended, once a read found its end: 0 when the
    // program closed it, otherwise the system's error code for the read
    // that failed while it was open (Windows' GetLastError, elsewhere errno).
    int output_error() const;

    // Waits until `bytes` of standard output are waiting to be read, the
    // program has ended, or `stop` is set, so that one read takes them at
    // once: read as it comes, a 4K frame arrives in some 300 pieces, each
    // waking the reading thread (6 ms of CPU per frame, against 4). It never
    // waits for more than half of what the pipe holds, or a program waiting
    // for room in a full pipe would be waited for in turn. Only on Windows,
    // as the Python app gathers; elsewhere it returns at once.
    void gather(std::size_t bytes, const std::atomic<bool>& stop);

    // Fills `bytes` at `into` from standard output, gathering before each
    // read: a frame comes in a few large reads, however much bigger than the
    // pipe it is, and straight into `into`. How many were read: fewer only
    // once the program has closed it.
    std::size_t read_full(void* into, std::size_t bytes, const std::atomic<bool>& stop);

    // Waits for the program to end, and returns its exit code.
    int wait();

    // Waits up to `timeout`; true if the program has ended.
    bool wait_for(std::chrono::milliseconds timeout);

    // Ends the program, and every program it started.
    void terminate();

    // Stops the program and every program it started from running, as the
    // Python app suspends a process tree (proc.py), until a resume() of this
    // call's own. Each call is a hold of its own, and the program runs again
    // only once every hold is let go: the CPU metrics' flow control and the
    // user's Pause each hold it, from their own threads, and neither lets go
    // of the other's. A call while held stops what was started since: a
    // launcher's program found after the first call would otherwise run on.
    // On Windows each process of its job is suspended once, less the console
    // host Windows gives a console program; Windows counts suspends, so these
    // nest with any made elsewhere. Elsewhere its process group is sent
    // SIGSTOP.
    void suspend();

    // Lets go of one hold of suspend(). The last lets every process
    // suspend() stopped run again: on Windows the last suspended first, so
    // that a program never runs under a stopped launcher; elsewhere its
    // process group is sent SIGCONT.
    void resume();

    // The end of what the program wrote to standard error (error_tail), its
    // lines ended with "\n" (universal_newlines), complete once it has ended
    // (after wait()).
    std::string standard_error();

    int process_id() const;

private:
    struct State;
    explicit ChildProcess(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

// From now on, the programs this process starts are ended, stopped and
// resumed with it by whoever started it, as the programs a Windows process
// starts are in its job: called by a process of its own (run_isolated's
// child), whose parent ends it with what it started. Elsewhere than Windows
// they then join this process's group instead of making their own, and
// terminate(), suspend() and resume() reach the program alone. On Windows
// it does nothing.
void start_programs_in_this_group();

// Ends at once every program this process started that is still running,
// with what they started (end_process_now calls it before the process
// ends: a Windows job does this by itself).
void end_programs_now();

// Runs a program to its end and returns what it wrote. A program still
// running after `timeout`, or once `stop` returns true (asked every 100 ms,
// before the time is checked), is ended, and the result says which. Throws
// ProcessError if it cannot be started.
ProcessOutput run_program(const std::vector<std::string>& arguments, std::chrono::milliseconds timeout,
                          const ProcessOptions& options = {}, const std::function<bool()>& stop = nullptr);

}  // namespace vml::platform
