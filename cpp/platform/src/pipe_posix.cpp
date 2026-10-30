// InboundPipe on Linux and macOS: a FIFO in the temporary folder.
#include "vml/platform/pipe.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <random>

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

// The start of the names of a process's FIFOs.
std::string pipe_prefix(int process_id) { return "vml-vmaf-" + std::to_string(process_id) + "-"; }

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
        if (name.starts_with(prefix) && entry->is_fifo(type_error)) {
            unlink(entry->path().c_str());
        }
    }
}

// OutboundPipe: Windows only, for want of a decoder in this process to feed
// it from (native/software_frames.cpp is Windows' own).
struct OutboundPipe::State {};

std::string OutboundPipe::unique_path(std::string_view name) {
    return (std::filesystem::temp_directory_path() / ("vml-feed-" + std::to_string(getpid()) + "-" + random_part() +
                                                      "-" + std::string(name)))
        .string();
}

OutboundPipe::OutboundPipe(const std::string&, std::size_t) {
    throw PipeError("The app's own decoding into FFmpeg is made on Windows only");
}

OutboundPipe::~OutboundPipe() = default;
void OutboundPipe::connect() {}
bool OutboundPipe::write(const void*, std::size_t) { return false; }
bool OutboundPipe::finish() { return false; }
void OutboundPipe::touch(const std::string&) {}

}  // namespace vml::platform
