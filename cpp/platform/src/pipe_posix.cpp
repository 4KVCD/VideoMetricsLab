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

}  // namespace

InboundPipe::InboundPipe(std::string_view name, std::size_t buffer_bytes)
    : path_((std::filesystem::temp_directory_path() / ("vml-vmaf-" + std::to_string(getpid()) + "-" +
                                                        random_part() + "-" + std::string(name)))
                .string()),
      buffer_bytes_(buffer_bytes) {
    if (mkfifo(path_.c_str(), 0600) != 0) {
        throw PipeError("The pipe " + path_ + " could not be made: " + std::strerror(errno));
    }
}

InboundPipe::~InboundPipe() {
    close();
    unlink(path_.c_str());
}

void InboundPipe::connect() {
    // Opening a FIFO to read waits for its writer.
    do {
        descriptor_ = open(path_.c_str(), O_RDONLY);
    } while (descriptor_ < 0 && errno == EINTR);
    if (descriptor_ < 0) {
        throw PipeError("Waiting for the writer of " + path_ + " failed: " + std::strerror(errno));
    }
}

void InboundPipe::gather(std::size_t, const std::atomic<bool>&, const std::atomic<bool>&) {}

std::size_t InboundPipe::read(void* into, std::size_t bytes) {
    if (descriptor_ < 0) {
        return 0;
    }
    while (true) {
        const ssize_t got = ::read(descriptor_, into, bytes);
        if (got >= 0) {
            return static_cast<std::size_t>(got);
        }
        if (errno != EINTR) {
            return 0;
        }
    }
}

void InboundPipe::close() {
    if (descriptor_ >= 0) {
        ::close(descriptor_);
        descriptor_ = -1;
    }
}

void InboundPipe::touch(const std::string& path) {
    // A reader waiting in open() counts as the FIFO's reader, so a writer
    // that does not wait for one can open it then (and fails, ENXIO, before).
    const int writer = open(path.c_str(), O_WRONLY | O_NONBLOCK);
    if (writer >= 0) {
        ::close(writer);
    }
}

}  // namespace vml::platform
