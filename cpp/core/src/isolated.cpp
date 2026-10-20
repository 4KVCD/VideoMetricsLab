#include "vml/isolated.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <utility>

#include "vml/app_log.h"
#include "vml/ffmpeg_run.h"
#include "vml/gpu_frames.h"
#include "vml/platform/console.h"
#include "vml/platform/pipe.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/temp_files.h"
#include "vml/vmaf_cuda.h"
#include "vml/vship.h"
#include "vml/vship_run.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vmaf_vulkan.h"

namespace vml {

namespace {

// Starts each message's line, so that whatever else the child's libraries
// write to its standard output is passed over.
constexpr std::string_view message_mark = "\x1E" "vml:";

std::optional<std::filesystem::path>& program() {
    static std::optional<std::filesystem::path> path;
    return path;
}

// How long a child's folder is tried while a program the child started
// (FFmpeg) is still ending with a file in it open.
constexpr std::chrono::seconds folder_wait{5};

// How long a child that has sent its answer is given to end, before it is
// ended: one that hangs as it ends (a GPU driver unloading) would hold the
// run for good, and its answer is all that is wanted (the Python app's
// join(timeout=5)).
constexpr std::chrono::seconds answered_wait{5};

// The file in a child's folder where it reports a crash.
constexpr std::string_view crash_report_name = "crash.txt";

// A folder of its own for a child's request and files, deleted with it.
class ChildFolder {
public:
    ChildFolder() {
        std::optional<std::filesystem::path> made = make_temp_folder("vml-isolated-");
        if (!made) {
            throw std::runtime_error("A folder for a process of its own could not be made.");
        }
        path_ = std::move(*made);
    }
    // A program the child started can still have a file here open for a
    // moment after the child has ended: FFmpeg, ended with it by a Cancel.
    // Ending a program does not wait for it to end, so the folder is tried
    // again until that program has ended too.
    ~ChildFolder() {
        const auto deadline = std::chrono::steady_clock::now() + folder_wait;
        while (true) {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
            if (!std::filesystem::exists(path_, ignored) || std::chrono::steady_clock::now() >= deadline) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    ChildFolder(const ChildFolder&) = delete;
    ChildFolder& operator=(const ChildFolder&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace

IsolatedCrashError::IsolatedCrashError(const std::string& what, int exit_code)
    : std::runtime_error(what + " crashed (" + describe_exit_code(exit_code) + ")"), exit_code_(exit_code) {}

std::string describe_exit_code(int exit_code) {
    const auto code = static_cast<std::uint32_t>(exit_code);
    if (code <= 0xFFFF) {
        return "exit code " + std::to_string(exit_code);
    }
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string hex = "exit code 0x";
    for (int shift = 28; shift >= 0; shift -= 4) {
        hex += digits[(code >> shift) & 0xF];
    }
    return hex;
}

namespace {

// Writes `request`, for the child running `what`, into `folder`; returns
// its file.
std::filesystem::path write_request(const std::filesystem::path& folder, const nlohmann::json& request,
                                    std::string_view what) {
    const std::filesystem::path request_file = folder / "request.json";
    std::ofstream file(request_file, std::ios::binary);
    file << request.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    if (!file) {
        throw std::runtime_error("The request of " + std::string(what) + " could not be written.");
    }
    return request_file;
}

// The parent's side of a child's run: a thread reads the child's messages as
// they come, and this one hands them on and watches for Cancel, which the
// child cannot see.
class ChildConversation {
public:
    ChildConversation(platform::ChildProcess& child, const std::function<void(const nlohmann::json&)>& on_message,
                      const CancelToken& cancel)
        : child_(child), on_message_(on_message), cancel_(cancel), reader_([this] { read(); }) {}
    ~ChildConversation() {
        if (reader_.joinable()) {
            reader_.join();
        }
    }
    ChildConversation(const ChildConversation&) = delete;
    ChildConversation& operator=(const ChildConversation&) = delete;

    // Hands on the child's messages until its output ends. Throws what
    // on_message throws, the child ended first.
    void run();

    // Its result, or its error, once it said either.
    std::optional<nlohmann::json> result;
    std::optional<std::string> failure;
    // Whether Cancel ended it, and whether it was ended for not ending by
    // itself once it had answered.
    bool stopped = false;
    bool ended_after_answer = false;

private:
    void read();
    void take(const std::string& text);
    void end_if_answered_long_ago();

    platform::ChildProcess& child_;
    const std::function<void(const nlohmann::json&)>& on_message_;
    const CancelToken& cancel_;
    std::mutex mutex_;
    std::condition_variable arrived_;
    std::deque<std::string> messages_;
    bool ended_ = false;
    std::optional<std::chrono::steady_clock::time_point> answered_;  // when its result or error came
    std::thread reader_;  // last: it starts once the rest is set up
};

void ChildConversation::run() {
    try {
        std::unique_lock lock(mutex_);
        while (true) {
            arrived_.wait_for(lock, std::chrono::milliseconds(100), [this] { return !messages_.empty() || ended_; });
            if (cancel_.cancelled() && !stopped) {
                child_.terminate();  // the child, and every program it started (they are in its job)
                stopped = true;
            }
            while (!messages_.empty()) {
                const std::string text = std::move(messages_.front());
                messages_.pop_front();
                lock.unlock();
                take(text);
                lock.lock();
            }
            if (ended_) {
                break;
            }
            end_if_answered_long_ago();
        }
    } catch (...) {
        child_.terminate();  // on_message threw: the child must not outlive the run
        reader_.join();
        throw;
    }
    reader_.join();
}

// The reader's thread: each message line the child writes, until its output
// ends.
void ChildConversation::read() {
    while (const std::optional<std::string> line = child_.read_line()) {
        // Looked for anywhere in the line: a library may have written text
        // without ending its line.
        const std::size_t mark = line->find(message_mark);
        if (mark != std::string::npos) {
            const std::lock_guard lock(mutex_);
            messages_.push_back(line->substr(mark + message_mark.size()));
            arrived_.notify_one();
        }
    }
    const std::lock_guard lock(mutex_);
    ended_ = true;
    arrived_.notify_one();
}

// One message of the child's: its result or its error kept, a line of its
// log written to this app's under its own name, anything else handed to
// on_message (not once Cancel stopped it).
void ChildConversation::take(const std::string& text) {
    const nlohmann::json message = nlohmann::json::parse(text, nullptr, false);
    if (message.is_object() && message.contains("result")) {
        result = message["result"];
    } else if (message.is_object() && message.contains("error")) {
        failure = message["error"].is_string() ? message["error"].get<std::string>() : message["error"].dump();
    } else if (message.is_object() && message.contains("log") && message["log"].is_object()) {
        const nlohmann::json& line = message["log"];
        app_log::write(line.value("level", static_cast<int>(app_log::Level::info)),
                       line.value("name", std::string("vml")), line.value("text", std::string()));
    } else if (!message.is_discarded() && !stopped && on_message_) {
        on_message_(message);
    }
}

// Ends a child that has answered and not ended within answered_wait.
void ChildConversation::end_if_answered_long_ago() {
    if (!(result || failure) || ended_after_answer) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!answered_) {
        answered_ = now;
    } else if (now - *answered_ >= answered_wait) {
        child_.terminate();  // its answer is here; it and what it started go
        ended_after_answer = true;
    }
}

}  // namespace

void set_isolation_program(std::optional<std::filesystem::path> path) { program() = std::move(path); }

std::optional<std::filesystem::path> isolation_program() { return program(); }

nlohmann::json run_isolated(std::string_view task, std::string_view what, const nlohmann::json& request,
                            const std::function<void(const nlohmann::json&)>& on_message, const CancelToken& cancel,
                            const IsolatedFiles& on_files) {
    cancel.throw_if_cancelled();
    if (!program()) {
        throw std::logic_error("no program to run " + std::string(what) + " in");
    }
    const ChildFolder folder;
    const std::filesystem::path request_file = write_request(folder.path(), request, what);
    // In this process's working folder, as the Python app's child is: a
    // video named by a relative path is found there (the child finds its own
    // files by the request's path).
    platform::ChildProcess child = platform::ChildProcess::start(
        {platform::path_text(*program()), std::string(isolated_argument), std::string(task),
         platform::path_text(request_file)});
    // Pause holds the child and every program it started (they are in its
    // job), so the work in it waits whatever it is doing.
    const CancelToken::Attached attached = cancel.attach(child);
    ChildConversation conversation(child, on_message, cancel);
    conversation.run();
    const int exit_code = child.wait();
    platform::InboundPipe::remove_left_by(child.process_id());  // a Cancel's kill left them
    if (conversation.ended_after_answer) {
        app_log::write(app_log::Level::warning, "vml.isolated",
                       std::string(what) + " did not end within " + std::to_string(answered_wait.count()) +
                           " s of its answer, and was ended");
    }
    if (conversation.stopped || cancel.cancelled()) {
        throw Cancelled();
    }
    if (conversation.failure) {
        throw std::runtime_error(*conversation.failure);
    }
    if (!conversation.result) {
        IsolatedCrashError crash(std::string(what), exit_code);
        if (const std::optional<std::string> report = platform::read_file(folder.path() / crash_report_name);
            report && !report->empty()) {
            app_log::write(app_log::Level::error, "vml.isolated",
                           std::string(crash.what()) + "; its stack at the crash:\n" + *report);
        }
        throw crash;
    }
    if (on_files) {
        on_files(*conversation.result, folder.path());
    }
    return *conversation.result;
}

void send_isolated_message(const nlohmann::json& message) {
    platform::write_text(platform::Stream::output,
                         std::string(message_mark) +
                             message.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) + "\n");
}

int run_isolated_task(std::string_view task, const std::filesystem::path& request_file) {
    platform::crash_quietly();  // its parent reports a crash; no dialog waits for the user
    // What it starts (FFmpeg) is ended and paused with it, by its parent's
    // Cancel and Pause.
    platform::start_programs_in_this_group();
    // Where a crash happened, for the parent to put in its log: a crash ends
    // this process before it could send anything.
    platform::report_crashes_to(request_file.parent_path() / crash_report_name);
    // Its log's lines go to the app that started it, which writes them.
    app_log::forward_lines([](int level, std::string_view name, std::string_view text) {
        send_isolated_message({{"log", {{"name", std::string(name)}, {"level", level}, {"text", std::string(text)}}}});
    });
    int exit_code = 0;
    try {
        std::ifstream file(request_file, std::ios::binary);
        const nlohmann::json request = nlohmann::json::parse(file);
        nlohmann::json result;
        if (task == "vulkan-probe") {
            const VulkanProbe probe = probe_vulkan();
            result = {{"available", probe.available},
                      {"device", probe.device ? nlohmann::json(*probe.device) : nlohmann::json(nullptr)},
                      {"text", probe.text}};
        } else if (task == "vmaf-v1-probe") {
            const V1Probe probe = probe_v1();
            result = {{"available", probe.available}, {"text", probe.text}};
        } else if (task == "cuda-probe") {
            const CudaProbe probe = probe_cuda();
            result = {{"available", probe.available}, {"text", probe.text}};
        } else if (task == "vship-pass") {
            result = vship_pass_task(request, request_file.parent_path());
        } else if (task == "vship-probe") {
            result = vship_probe_result(probe_vship_device());
        } else if (task == "gpu-run" || task == "cpu-metrics") {
            result = scoring_task(request, request_file.parent_path());
        } else if (task == "process-id") {
            // For the window's self-test: that a process of its own starts.
            result = platform::process_id();
        } else if (task == "test-echo") {
            // For the tests of run_isolated: two messages, a file (the
            // request's text, then the folder the child works in), and the
            // request back as the result.
            send_isolated_message({{"progress", {1, 2, 3.5}}});
            send_isolated_message({{"status", "on its way"}});
            std::ofstream(request_file.parent_path() / "echo.txt", std::ios::binary)
                << request.value("file", "") << "\n"
                << platform::path_text(std::filesystem::current_path());
            result = request;
        } else if (task == "test-crash") {
            // For the tests: ends as an access violation ends a process.
            std::_Exit(static_cast<int>(0xC0000005u));
        } else if (task == "test-fault") {
            // For the tests: a real crash, which the process reports before
            // it ends: an access violation on Windows, an abort elsewhere
            // (AddressSanitizer keeps SIGSEGV to itself, not SIGABRT).
#ifdef _WIN32
            volatile int* nowhere = nullptr;
            *nowhere = 1;
#else
            std::abort();
#endif
        } else if (task == "test-wait") {
            // For the tests: waits to be cancelled.
            send_isolated_message({{"waiting", true}});
            std::this_thread::sleep_for(std::chrono::minutes(1));
        } else if (task == "test-hold") {
            // For the tests: a program this one starts holds a file in this
            // one's folder open, as FFmpeg holds its timestamps, and this
            // waits to be cancelled.
            const std::filesystem::path folder = request_file.parent_path();
            const std::filesystem::path hold_request = folder / "hold.json";
            std::ofstream(hold_request, std::ios::binary)
                << nlohmann::json{{"file", platform::path_text(folder / "held.txt")}}.dump();
            platform::ChildProcess holder = platform::ChildProcess::start(
                {platform::path_text(platform::executable_path()), std::string(isolated_argument), "test-hold-file",
                 platform::path_text(hold_request)});
            holder.read_line();  // the file is open
            send_isolated_message({{"waiting", true}, {"folder", platform::path_text(folder)}});
            std::this_thread::sleep_for(std::chrono::minutes(1));
        } else if (task == "test-hold-file") {
            // For the tests: holds the request's file open.
            const std::ofstream held(platform::path_from_text(request.at("file").get<std::string>()),
                                     std::ios::binary);
            send_isolated_message({{"holding", true}});
            std::this_thread::sleep_for(std::chrono::minutes(1));
        } else {
            throw std::invalid_argument("there is no task " + std::string(task));
        }
        send_isolated_message({{"result", result}});
    } catch (const std::exception& error) {
        send_isolated_message({{"error", error.what()}});
        exit_code = 1;
    }
    if (stuck_decoders() > 0) {
        // A decoder never finished closing, and its thread holds the
        // driver's locks, which a normal exit, unloading every library in
        // turn, could wait on forever. What the parent needs has been sent.
        platform::end_process_now(exit_code);
    }
    return exit_code;
}

}  // namespace vml
