// ChildProcess on Linux and macOS: posix_spawnp with pipes, in a process
// group of its own, so that terminate() ends the program and whatever it
// started together.
//
// Still to do (for the Linux and macOS releases): a program is not yet ended
// when the app crashes, as Windows' job objects do.
#include "vml/platform/process.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

extern char** environ;

namespace vml::platform {

namespace {

[[noreturn]] void fail(const std::string& what, int error = errno) {
    throw ProcessError(what + " (" + std::strerror(error) + ")");
}

// A file descriptor, closed when it goes.
class Descriptor {
public:
    Descriptor() = default;
    explicit Descriptor(int fd) : fd_(fd) {}
    Descriptor(Descriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Descriptor& operator=(Descriptor&& other) noexcept {
        if (this != &other) {
            reset(std::exchange(other.fd_, -1));
        }
        return *this;
    }
    ~Descriptor() { reset(); }

    int get() const { return fd_; }
    void reset(int fd = -1) {
        if (fd_ >= 0) {
            close(fd_);
        }
        fd_ = fd;
    }

private:
    int fd_ = -1;
};

struct Pipe {
    Descriptor read;
    Descriptor write;
};

// A pipe neither of whose ends survives into another program (FD_CLOEXEC):
// the program gets its own copy of the write end, made in the spawn.
Pipe make_pipe() {
    int ends[2];
    if (pipe(ends) != 0) {
        fail("a pipe could not be made");
    }
    for (const int end : ends) {
        fcntl(end, F_SETFD, FD_CLOEXEC);
    }
    return Pipe{Descriptor(ends[0]), Descriptor(ends[1])};
}

// An exit status as Python's subprocess gives it: the exit code, or minus
// the signal that ended the program.
int exit_code_of(int status) {
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return -WTERMSIG(status);
    }
    return -1;
}

}  // namespace

struct ChildProcess::State {
    pid_t id = -1;
    std::mutex wait_mutex;
    bool ended = false;
    int exit_code = -1;
    Descriptor output;
    std::string unread;
    bool output_ended = false;

    std::thread error_reader;
    std::mutex error_mutex;
    std::condition_variable error_ended_signal;
    std::string error_text;
    bool error_ended = false;

    // Reaps the program if it has ended (or, with `block`, once it has).
    bool reap(bool block) {
        const std::lock_guard lock(wait_mutex);
        if (ended) {
            return true;
        }
        int status = 0;
        pid_t done = waitpid(id, &status, block ? 0 : WNOHANG);
        while (done < 0 && errno == EINTR) {  // a signal (Ctrl+C) came while it waited
            done = waitpid(id, &status, block ? 0 : WNOHANG);
        }
        if (done == id) {
            ended = true;
            exit_code = exit_code_of(status);
        }
        return ended;
    }

    ~State() {
        if (!reap(false)) {
            kill(-id, SIGKILL);
            reap(true);
        }
        if (error_reader.joinable()) {
            error_reader.join();
        }
    }
};

ChildProcess ChildProcess::start(const std::vector<std::string>& arguments, const ProcessOptions& options) {
    if (arguments.empty()) {
        throw ProcessError("no program to start");
    }
    Pipe output = make_pipe();
    Pipe error = make_pipe();

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, output.write.get(), STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, error.write.get(), STDERR_FILENO);
    const std::string folder = options.working_directory.string();
    if (!folder.empty()) {
        posix_spawn_file_actions_addchdir_np(&actions, folder.c_str());
    }
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    // A process group of its own, which terminate() ends as one.
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attributes, 0);

    std::vector<char*> argv;
    for (const std::string& argument : arguments) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    pid_t id = -1;
    const int spawned = posix_spawnp(&id, argv.front(), &actions, &attributes, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    if (spawned != 0) {
        fail("could not start " + arguments.front(), spawned);
    }

    auto state = std::make_unique<State>();
    state->id = id;
    // The program has its copies; ours must close, or the pipes never end.
    output.write.reset();
    error.write.reset();
    state->output = std::move(output.read);
    state->error_reader = std::thread([raw = state.get(), pipe = std::move(error.read)]() {
        char buffer[4096];
        ssize_t count = 0;
        while ((count = read(pipe.get(), buffer, sizeof buffer)) > 0 || (count < 0 && errno == EINTR)) {
            if (count > 0) {
                const std::lock_guard lock(raw->error_mutex);
                keep_error_tail(raw->error_text, std::string_view(buffer, static_cast<std::size_t>(count)));
            }
        }
        const std::lock_guard lock(raw->error_mutex);
        raw->error_ended = true;
        raw->error_ended_signal.notify_all();
    });
    return ChildProcess(std::move(state));
}

ChildProcess::ChildProcess(std::unique_ptr<State> state) : state_(std::move(state)) {}
ChildProcess::ChildProcess(ChildProcess&&) noexcept = default;
ChildProcess& ChildProcess::operator=(ChildProcess&&) noexcept = default;
ChildProcess::~ChildProcess() = default;

std::optional<std::string> ChildProcess::read_line() {
    while (true) {
        const std::size_t end = state_->unread.find('\n');
        if (end != std::string::npos) {
            std::string line = state_->unread.substr(0, end);
            state_->unread.erase(0, end + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return line;
        }
        if (state_->output_ended) {
            if (state_->unread.empty()) {
                return std::nullopt;
            }
            return std::exchange(state_->unread, std::string());
        }
        char buffer[4096];
        const ssize_t count = read(state_->output.get(), buffer, sizeof buffer);
        if (count > 0) {
            state_->unread.append(buffer, static_cast<std::size_t>(count));
        } else if (count == 0 || errno != EINTR) {
            state_->output_ended = true;
        }
    }
}

int ChildProcess::wait() {
    state_->reap(true);
    return state_->exit_code;
}

bool ChildProcess::wait_for(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!state_->reap(false)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

void ChildProcess::terminate() {
    if (!state_->reap(false)) {
        kill(-state_->id, SIGKILL);
    }
}

std::string ChildProcess::standard_error() {
    std::unique_lock lock(state_->error_mutex);
    state_->error_ended_signal.wait_for(lock, std::chrono::seconds(5), [this] { return state_->error_ended; });
    return error_tail(state_->error_text);
}

int ChildProcess::process_id() const { return static_cast<int>(state_->id); }

}  // namespace vml::platform
