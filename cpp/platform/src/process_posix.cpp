// ChildProcess on Linux and macOS: posix_spawnp with pipes, in a process
// group of its own, so that terminate() ends the program and whatever it
// started together (or, in a process that start_programs_in_this_group()
// was called in, in that process's group, which its parent ends).
//
// However this process ends -- a crash or a Force Quit too -- the groups of
// the programs it leaves running end with it, as a Windows job's programs
// do: a reaper (a shell started with the first of them) is told of each
// group as it starts and ends, and ends those still listed when this
// process's end of their pipe closes, which the system does as it ends.
#include "vml/platform/process.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "error_lines.h"
#include "read_ahead.h"

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

// A pipe neither of whose ends survives into another program (close on
// exec): the program gets its own copy of the write end, made in the spawn.
// On Linux the pipe is made so; elsewhere it is so a moment later, and the
// spawn hands a program nothing else meanwhile (start).
Pipe make_pipe() {
    int ends[2];
#ifdef __linux__
    if (pipe2(ends, O_CLOEXEC) != 0) {
        fail("a pipe could not be made");
    }
#else
    if (pipe(ends) != 0) {
        fail("a pipe could not be made");
    }
    for (const int end : ends) {
        fcntl(end, F_SETFD, FD_CLOEXEC);
    }
#endif
    return Pipe{Descriptor(ends[0]), Descriptor(ends[1])};
}

// Whether programs join this process's group (start_programs_in_this_group).
std::atomic<bool> programs_in_this_group{false};

// The reaper's script: it keeps the groups it is told of ("+ID" a line, "-ID"
// gone), and when its input ends ends those left. The signals a terminal or a
// logout sends are ignored: only the end of its input ends it. POSIX sh, as
// macOS's /bin/sh (bash) and Debian's (dash) both run it.
constexpr const char* reaper_script =
    "trap '' HUP INT TERM PIPE; groups=; "
    "while IFS= read -r line; do case $line in "
    "+*) groups=\"$groups ${line#+}\";; "
    "-*) rest=; for g in $groups; do [ \"$g\" = \"${line#-}\" ] || rest=\"$rest $g\"; done; groups=$rest;; "
    "esac; done; "
    "for g in $groups; do kill -s KILL -- \"-$g\" 2>/dev/null; done";

// This process's reaper: the write end of its input, never closed (its
// closing must mean this process has ended), and its id. Not destroyed at
// exit, when a program's destructor may still tell it of its group.
struct Reaper {
    int input = -1;
    pid_t id = -1;
};
Reaper& reaper() {
    static Reaper* const made = new Reaper;
    return *made;
}

// Starts the reaper, once; without it (no /bin/sh) programs start as before.
void start_reaper() {
    // Neither end survives into another program (close on exec), as
    // make_pipe's: a program holding the write end would keep the reaper
    // from ever ending its groups.
    int ends[2];
#ifdef __linux__
    if (pipe2(ends, O_CLOEXEC) != 0) {
        return;
    }
#else
    if (pipe(ends) != 0) {
        return;
    }
    for (const int end : ends) {
        fcntl(end, F_SETFD, FD_CLOEXEC);
    }
#endif
    // A line that would wait fails at once (tell_reaper then ends the
    // reaper), and one to a reaper gone fails instead of ending this process.
    fcntl(ends[1], F_SETFL, fcntl(ends[1], F_GETFL) | O_NONBLOCK);
#ifdef F_SETNOSIGPIPE
    fcntl(ends[1], F_SETNOSIGPIPE, 1);
#else
    std::signal(SIGPIPE, SIG_IGN);  // programs get it back (posix_spawnattr_setsigdefault)
#endif
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, ends[0], STDIN_FILENO);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    // None of this process's other descriptors (another program's pipe would
    // last as long as the reaper), in a group of its own, which a terminal's
    // Ctrl+C or a program's group does not reach.
    short flags = POSIX_SPAWN_SETPGROUP;
#ifdef __APPLE__
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#elif defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    posix_spawn_file_actions_addclosefrom_np(&actions, STDERR_FILENO + 1);
#endif
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes, flags);
    posix_spawnattr_setpgroup(&attributes, 0);
    char shell[] = "/bin/sh";
    char option[] = "-c";
    char* argv[] = {shell, option, const_cast<char*>(reaper_script), nullptr};  // not written to
    pid_t id = -1;
    const int spawned = posix_spawn(&id, shell, &actions, &attributes, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    close(ends[0]);
    if (spawned != 0) {
        close(ends[1]);
        return;
    }
    reaper().input = ends[1];
    reaper().id = id;
}

// Tells the reaper that the group `id` started ('+') or is gone ('-'); with
// running_mutex held, so that it hears them in order. A line it cannot take
// whole ends it: one lost could leave it a group that is gone, whose id
// another process may then have.
void tell_reaper(char what, pid_t id) {
    Reaper& known = reaper();
    if (known.input < 0) {
        return;
    }
    const std::string line = what + std::to_string(id) + "\n";  // shorter than PIPE_BUF: written whole or not
    ssize_t written = -1;
    do {
        written = ::write(known.input, line.data(), line.size());
    } while (written < 0 && errno == EINTR);
    if (written != static_cast<ssize_t>(line.size())) {
        kill(known.id, SIGKILL);  // never reaped here: its id stays ours
        known.input = -1;         // left open: its end would end every group listed
    }
}

// How a program waitid found ended, rather than stopped (as macOS's waitid
// says a program SIGSTOP stopped is, until SIGCONT).
bool exited(const siginfo_t& info) {
    return info.si_code == CLD_EXITED || info.si_code == CLD_KILLED || info.si_code == CLD_DUMPED;
}

// The programs started and not yet reaped, and whether each leads a group
// of its own, for end_programs_now.
std::mutex running_mutex;
std::map<pid_t, bool>& running_programs() {
    static std::map<pid_t, bool> running;
    return running;
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
    bool own_group = true;  // the leader of a group of its own, which signals reach as one
    std::mutex wait_mutex;
    bool ended = false;
    int exit_code = -1;
    // suspend() has stopped its process group, with `holds` calls not yet
    // let go; `hold_mutex` guards both.
    std::mutex hold_mutex;
    int holds = 0;
    bool stopped = false;
    Descriptor output;
    std::string unread;
    bool output_ended = false;
    int output_error = 0;  // output_error()

    std::thread error_reader;
    std::mutex error_mutex;
    std::condition_variable error_ended_signal;
    std::string error_text;
    bool error_ended = false;

    // Reaps the program if it has ended (or, with `block`, once it has).
    // The leader of a group of its own takes the rest of its group with it:
    // what it started and left running (FFmpeg, after a crash) is ended
    // first, while its id, the group's, is still its own (not yet reaped).
    // A program stopped (suspend) is not ended, though macOS's waitid says
    // so too. Waiting holds no lock: suspend(), resume() and terminate() may
    // be what lets it end.
    bool reap(bool block) {
        std::unique_lock lock(wait_mutex);
        while (true) {
            if (ended) {
                return true;
            }
            siginfo_t info{};
            int waited = 0;
            do {
                waited = waitid(P_PID, static_cast<id_t>(id), &info, WEXITED | WNOWAIT | WNOHANG);
            } while (waited < 0 && errno == EINTR);
            if (waited == 0 && info.si_pid == id && exited(info)) {
                finish(true, block);
                return ended;
            }
            if (waited < 0) {
                finish(false, block);  // waitid cannot wait for it: waitpid says, as before
                return ended;
            }
            if (!block) {
                return false;  // running, or stopped
            }
            lock.unlock();
            info = siginfo_t{};
            do {
                waited = waitid(P_PID, static_cast<id_t>(id), &info, WEXITED | WNOWAIT);
            } while (waited < 0 && errno == EINTR);
            if (waited == 0 && info.si_pid == id && !exited(info)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));  // stopped: again once it may run
            }
            lock.lock();  // another thread may have reaped it meanwhile
        }
    }

    // Reaps the program, with wait_mutex held, once waitid saw it end
    // (`seen`), or to learn what waitpid says (waiting, with `block`) when
    // waitid could not wait. Its group is ended and the reaper told before
    // the leader is reaped: until then its id cannot be another process's.
    void finish(bool seen, bool block) {
        if (seen && own_group) {
            kill(-id, SIGKILL);
            const std::lock_guard running_lock(running_mutex);
            tell_reaper('-', id);
        }
        int status = 0;
        const int options = seen || block ? 0 : WNOHANG;
        pid_t done = waitpid(id, &status, options);
        while (done < 0 && errno == EINTR) {  // a signal (Ctrl+C) came while it waited
            done = waitpid(id, &status, options);
        }
        if (done == id || (done < 0 && errno == ECHILD)) {
            // ECHILD: reaped by something else, and so no longer ours to signal.
            ended = true;
            exit_code = done == id ? exit_code_of(status) : -1;
            const std::lock_guard running_lock(running_mutex);
            running_programs().erase(id);
            if (!seen && own_group) {
                tell_reaper('-', id);  // only now known to be gone
            }
        }
    }

    // Sends `signal` to the program's process group: the program and what it
    // started, which may run on after the program itself has ended. Only
    // while the program is not reaped (running, or ended and waiting to be
    // reaped), when its id -- the group's -- cannot belong to another
    // process. Whether it was sent.
    bool signal_group(int signal) {
        const std::lock_guard lock(wait_mutex);
        if (ended) {
            return false;
        }
        kill(own_group ? -id : id, signal);
        return true;
    }

    ~State() {
        signal_group(SIGKILL);
        reap(true);
        if (error_reader.joinable()) {
            error_reader.join();
        }
    }
};

ChildProcess ChildProcess::start(const std::vector<std::string>& arguments, const ProcessOptions& options) {
    // The program's, or nobody's if it does not start: closed here at the
    // end, whichever way this ends.
    const struct InheritedDescriptors {
        const std::vector<int>& descriptors;
        ~InheritedDescriptors() {
            for (const int descriptor : descriptors) {
                ::close(descriptor);
            }
        }
    } closed_at_the_end{options.inherited_descriptors};
    if (arguments.empty()) {
        throw ProcessError("no program to start");
    }
    Pipe output = make_pipe();
    Pipe error = make_pipe();
#ifdef F_SETPIPE_SZ
    if (options.output_pipe_bytes > 0) {  // Linux: asked for; a size past the system's limit is refused
        const std::size_t bytes = std::min<std::size_t>(options.output_pipe_bytes, 1u << 30);
        fcntl(output.write.get(), F_SETPIPE_SZ, static_cast<int>(bytes));
    }
#endif

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, output.write.get(), STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, error.write.get(), STDERR_FILENO);
    const std::string folder = options.working_directory.string();
    if (!folder.empty()) {
        posix_spawn_file_actions_addchdir_np(&actions, folder.c_str());
    }
    // The program gets its three descriptors and none of the app's others,
    // whatever their close-on-exec flag: given another program's pipe, it
    // would make that program's output last as long as its own copy.
    // A process group of its own, which terminate() ends as one, unless
    // this process's own group holds the programs it starts.
    const bool own_group = !programs_in_this_group.load();
    if (own_group) {
        static std::once_flag started;
        std::call_once(started, [] { start_reaper(); });
    }
    short flags = own_group ? POSIX_SPAWN_SETPGROUP : 0;
#ifdef __APPLE__
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
    for (const int descriptor : options.inherited_descriptors) {
        posix_spawn_file_actions_addinherit_np(&actions, descriptor);
    }
#else
    // These actions run in order: what is closed is closed around the
    // inherited ones, which dup2 onto themselves then keeps past the exec.
    std::vector<int> inherited = options.inherited_descriptors;
    std::sort(inherited.begin(), inherited.end());
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    int next_to_close = STDERR_FILENO + 1;
    for (const int descriptor : inherited) {
        for (; next_to_close < descriptor; ++next_to_close) {
            posix_spawn_file_actions_addclose(&actions, next_to_close);
        }
        next_to_close = descriptor + 1;
    }
    posix_spawn_file_actions_addclosefrom_np(&actions, next_to_close);
#endif
    for (const int descriptor : inherited) {
        posix_spawn_file_actions_adddup2(&actions, descriptor, descriptor);
    }
#endif
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    // SIGPIPE as a program starts with it, whatever this process does with
    // it (the command line ignores it, and an ignored signal is inherited):
    // a program writing to a reader that has gone then ends, as it should.
    sigset_t defaults;
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
    posix_spawnattr_setsigdefault(&attributes, &defaults);
    flags |= POSIX_SPAWN_SETSIGDEF;
    posix_spawnattr_setflags(&attributes, flags);
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
    state->own_group = own_group;
    {
        const std::lock_guard running_lock(running_mutex);
        running_programs()[id] = own_group;
        if (own_group) {
            tell_reaper('+', id);
        }
    }
    // The program has its copies; ours must close, or the pipes never end.
    output.write.reset();
    error.write.reset();
    state->output = std::move(output.read);
    state->error_reader = std::thread([raw = state.get(), pipe = std::move(error.read),
                                       lines = ErrorLines(options.on_error_line)]() mutable {
        char buffer[4096];
        ssize_t count = 0;
        while ((count = ::read(pipe.get(), buffer, sizeof buffer)) > 0 || (count < 0 && errno == EINTR)) {
            if (count > 0) {
                const std::string_view written(buffer, static_cast<std::size_t>(count));
                lines.add(written);
                const std::lock_guard lock(raw->error_mutex);
                keep_error_tail(raw->error_text, written);
            }
        }
        lines.finish();
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
        if (std::optional<std::string> line = take_line(state_->unread)) {
            return line;
        }
        if (state_->output_ended) {
            if (state_->unread.empty()) {
                return std::nullopt;
            }
            return std::exchange(state_->unread, std::string());
        }
        char buffer[4096];
        const ssize_t count = ::read(state_->output.get(), buffer, sizeof buffer);
        if (count > 0) {
            state_->unread.append(buffer, static_cast<std::size_t>(count));
        } else if (count == 0 || errno != EINTR) {
            state_->output_ended = true;
            state_->output_error = count == 0 ? 0 : errno;
        }
    }
}

std::size_t ChildProcess::read(void* into, std::size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    if (!state_->unread.empty()) {  // what read_line read ahead
        return take_bytes(state_->unread, into, bytes);
    }
    while (!state_->output_ended) {
        const ssize_t count = ::read(state_->output.get(), into, bytes);
        if (count > 0) {
            return static_cast<std::size_t>(count);
        }
        if (count == 0 || errno != EINTR) {
            state_->output_ended = true;
            state_->output_error = count == 0 ? 0 : errno;
        }
    }
    return 0;
}

int ChildProcess::output_error() const { return state_->output_error; }

void ChildProcess::gather(std::size_t bytes, const std::atomic<bool>& stop) {
    // Each read wakes the reading thread here too, but the Python app
    // gathers on Windows only, and so does this.
    static_cast<void>(bytes);
    static_cast<void>(stop);
}

std::string exit_code_text(int exit_code) { return std::to_string(exit_code); }

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

void ChildProcess::terminate() { state_->signal_group(SIGKILL); }

void ChildProcess::suspend() {
    // Sent again each time: a process stopped already stays so, and one that
    // joined the group since stops too.
    const std::lock_guard lock(state_->hold_mutex);
    ++state_->holds;
    if (state_->signal_group(SIGSTOP)) {
        state_->stopped = true;
    }
}

void ChildProcess::resume() {
    const std::lock_guard lock(state_->hold_mutex);
    if (state_->holds == 0 || --state_->holds > 0) {
        return;  // not held, or still held by another
    }
    if (state_->stopped) {
        state_->signal_group(SIGCONT);
    }
    state_->stopped = false;
}

std::string ChildProcess::standard_error() {
    std::unique_lock lock(state_->error_mutex);
    state_->error_ended_signal.wait_for(lock, std::chrono::seconds(5), [this] { return state_->error_ended; });
    return universal_newlines(error_tail(state_->error_text));
}

int ChildProcess::process_id() const { return static_cast<int>(state_->id); }

void start_programs_in_this_group() { programs_in_this_group = true; }

void end_programs_now() {
    // Not waiting for the lock: the process is ending, and a thread that
    // holds it may be the one that will not come back.
    std::unique_lock lock(running_mutex, std::try_to_lock);
    for (const auto& [id, own_group] : running_programs()) {
        kill(own_group ? -id : id, SIGKILL);
    }
}

}  // namespace vml::platform
