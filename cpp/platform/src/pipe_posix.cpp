// InboundPipe and OutboundPipe on Linux and macOS: a FIFO in the temporary
// folder.
#include "vml/platform/pipe.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>

namespace vml::platform {

struct InboundPipe::State {
    int descriptor = -1;  // the FIFO, once its writer opened it
};

namespace {

// 12 random hexadecimal digits, which make a pipe's name unique.
std::string random_part() {
    std::random_device random;
    static constexpr char digits[] = "0123456789abcdef";
    std::string text;
    for (int i = 0; i < 12; ++i) {
        text += digits[random() & 0xF];
    }
    return text;
}

// The start of the names of a process's FIFOs: the ones it reads, and the
// ones it names for a process it starts to write (OutboundPipe::unique_path).
std::string pipe_prefix(int process_id) { return "vml-vmaf-" + std::to_string(process_id) + "-"; }
std::string feed_prefix(int process_id) { return "vml-feed-" + std::to_string(process_id) + "-"; }

}  // namespace

InboundPipe::InboundPipe(std::string_view name, std::size_t)
    : path_((std::filesystem::temp_directory_path() / (pipe_prefix(getpid()) +
                                                        random_part() + "-" + std::string(name)))
                .string()),
      state_(std::make_unique<State>()) {
    if (mkfifo(path_.c_str(), 0600) != 0) {
        throw PipeError("The pipe " + path_ + " could not be made: " + std::strerror(errno));
    }
}

InboundPipe::~InboundPipe() {
    close();
    unlink(path_.c_str());
}

void InboundPipe::connect() {
    // Opening a FIFO to read waits for its writer. A copy of the reading end
    // in a program started meanwhile would keep the FIFO open after this
    // side closes it, so none goes to another program (close on exec).
    do {
        state_->descriptor = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    } while (state_->descriptor < 0 && errno == EINTR);
    if (state_->descriptor < 0) {
        throw PipeError("Waiting for the writer of " + path_ + " failed: " + std::strerror(errno));
    }
}

void InboundPipe::gather(std::size_t, const std::atomic<bool>&, const std::atomic<bool>&) {}

std::size_t InboundPipe::read(void* into, std::size_t bytes) {
    if (state_->descriptor < 0) {
        return 0;
    }
    while (true) {
        const ssize_t got = ::read(state_->descriptor, into, bytes);
        if (got >= 0) {
            return static_cast<std::size_t>(got);
        }
        if (errno != EINTR) {
            return 0;
        }
    }
}

void InboundPipe::close() {
    if (state_->descriptor >= 0) {
        ::close(state_->descriptor);
        state_->descriptor = -1;
    }
}

void InboundPipe::touch(const std::string& path) {
    // A reader waiting in open() counts as the FIFO's reader, so a writer
    // that does not wait for one can open it then (and fails, ENXIO, before).
    const int writer = open(path.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (writer >= 0) {
        ::close(writer);
    }
}

void InboundPipe::remove_left_by(int process_id) {
    const std::string prefix = pipe_prefix(process_id);
    const std::string feeds = feed_prefix(process_id);
    // Never throws: a folder that cannot be read leaves them (the run's
    // result stands).
    std::error_code error;
    const std::filesystem::path folder = std::filesystem::temp_directory_path(error);
    if (error) {
        return;
    }
    for (std::filesystem::directory_iterator entry(folder, error), end; !error && entry != end;
         entry.increment(error)) {
        const std::string name = entry->path().filename().string();
        std::error_code type_error;
        if ((name.starts_with(prefix) || name.starts_with(feeds)) && entry->is_fifo(type_error)) {
            unlink(entry->path().c_str());
        }
    }
}

// OutboundPipe. A FIFO holds what the system gives it (64 KB on macOS and
// Linux), whatever buffer_bytes asks: each write waits for the reader to make
// room, as a full named pipe's does.
//
// Its writing end is open from the start, so that the reader's open() never
// waits: on macOS a reader waiting there for a writer misses one that opens
// and closes again before it runs, and would wait for ever (a feeding process
// that failed, or a touch). A writer that ends, however it ends, then leaves
// its reader the end of its input. Whether the reader has come, connect()
// asks the FIFO without waiting in open() either.
struct OutboundPipe::State {
    std::string path;
    int descriptor = -1;  // the writing end, until finish() or the reader went
    bool connected = false;  // connect() found the reader
};

namespace {

// How often connect() asks whether the reader has opened the FIFO: FFmpeg
// takes a tenth of a second or more to open its inputs.
constexpr std::chrono::milliseconds reader_poll(2);

// How long a pipe whose reader never came keeps its writing end once its name
// is gone: a reader that found the name just before is then let through
// open(). Closed at once, it would wait there for ever: the system counts it
// as a reader only inside open(), after the last writer may have closed, and
// no other writer can reach a FIFO with no name.
constexpr std::chrono::milliseconds late_reader_grace(50);

}  // namespace

std::string OutboundPipe::unique_path(std::string_view name) {
    return (std::filesystem::temp_directory_path() / (feed_prefix(getpid()) + random_part() + "-" + std::string(name)))
        .string();
}

OutboundPipe::OutboundPipe(const std::string& path, std::size_t) : state_(std::make_unique<State>()) {
    state_->path = path;
    if (mkfifo(path.c_str(), 0600) != 0) {
        throw PipeError("The pipe " + path + " could not be made: " + std::strerror(errno));
    }
    // A writer opens without waiting only where there is a reader: this
    // one's own, for that moment. Neither goes to a program started
    // meanwhile (close on exec).
    const int own_reader = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    const int writer = own_reader >= 0 ? open(path.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC) : -1;
    const int error = errno;
    if (own_reader >= 0) {
        ::close(own_reader);
    }
    if (writer < 0) {
        unlink(path.c_str());
        throw PipeError("The pipe " + path + " could not be opened: " + std::strerror(error));
    }
    // Writes wait for room; one after the reader closed its end fails
    // (EPIPE) instead of ending the process with SIGPIPE.
    fcntl(writer, F_SETFL, fcntl(writer, F_GETFL) & ~O_NONBLOCK);
#ifdef F_SETNOSIGPIPE
    fcntl(writer, F_SETNOSIGPIPE, 1);
#else
    std::signal(SIGPIPE, SIG_IGN);
#endif
    state_->descriptor = writer;
}

OutboundPipe::~OutboundPipe() {
    // The name first: a reader that opens it after this fails, and one that
    // opened it before finds the end of its input when the writer closes
    // (late_reader_grace for one that never connected).
    unlink(state_->path.c_str());
    if (state_->descriptor >= 0) {
        if (!state_->connected) {
            std::this_thread::sleep_for(late_reader_grace);
        }
        ::close(state_->descriptor);
    }
}

void OutboundPipe::connect() {
    // Another writer opens without waiting once there is a reader. One that
    // came and went (touch) leaves the FIFO's name gone: the reader is gone,
    // and the first write finds it so, as on Windows.
    while (true) {
        const int probe = open(state_->path.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
        if (probe >= 0) {
            ::close(probe);
            state_->connected = true;
            return;
        }
        if (errno == ENOENT) {
            ::close(state_->descriptor);
            state_->descriptor = -1;
            return;
        }
        if (errno != ENXIO && errno != EINTR) {
            throw PipeError("Waiting for the reader of " + state_->path + " failed: " + std::strerror(errno));
        }
        std::this_thread::sleep_for(reader_poll);
    }
}

bool OutboundPipe::write(const void* data, std::size_t bytes) {
    if (state_->descriptor < 0) {
        return false;  // the reader came and went before connect
    }
    // A megabyte at most at a time, as on Windows.
    constexpr std::size_t piece = 1 << 20;
    const auto* next = static_cast<const std::uint8_t*>(data);
    std::size_t left = bytes;
    while (left > 0) {
        const ssize_t written = ::write(state_->descriptor, next, std::min(left, piece));
        if (written >= 0) {
            next += written;
            left -= static_cast<std::size_t>(written);
            continue;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EPIPE) {
            return false;
        }
        throw PipeError("Writing into " + state_->path + " failed: " + std::strerror(errno));
    }
    return true;
}

bool OutboundPipe::finish() {
    // What waits in a FIFO stays there for the reader once its writer has
    // closed it, and the reader then finds the end of its input. A reader
    // that closed its end after the last write is not seen here.
    if (state_->descriptor < 0) {
        return false;
    }
    ::close(state_->descriptor);
    state_->descriptor = -1;
    return true;
}

void OutboundPipe::touch(const std::string& path) {
    // The writer's connect() finds the name gone. Its writing end is open,
    // so no reader waits in open() for it.
    unlink(path.c_str());
}

}  // namespace vml::platform
