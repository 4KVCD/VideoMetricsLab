// A pipe this process reads and another program writes, opening it by its
// path as it opens a file: how FFmpeg hands its frames over for VMAF on the
// GPU. A named pipe (\\.\pipe\...) on Windows, a FIFO in the temporary
// folder elsewhere.
#pragma once

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vml::platform {

// The pipe could not be made, or its writer could not be waited for.
class PipeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InboundPipe {
public:
    // A new pipe, its path unique to this process and ending in `name`.
    // `buffer_bytes` is how much a named pipe holds; a FIFO holds what the
    // system gives it. Throws PipeError.
    InboundPipe(std::string_view name, std::size_t buffer_bytes);
    ~InboundPipe();
    InboundPipe(const InboundPipe&) = delete;
    InboundPipe& operator=(const InboundPipe&) = delete;

    // The path the writer opens.
    const std::string& path() const { return path_; }

    // Waits for the writer to open the pipe. A writer that opened it, wrote
    // and closed it already counts: what it wrote is still there to read.
    // Throws PipeError.
    void connect();

    // Waits until `bytes` are in the pipe (or as many as it holds), until
    // `writer_gone` says no more will come, or until `stop` is set, so that
    // a frame is taken in one read instead of in the pieces it arrives in.
    // Elsewhere than on Windows a FIFO holds too little for that, and this
    // returns at once.
    void gather(std::size_t bytes, const std::atomic<bool>& stop, const std::atomic<bool>& writer_gone);

    // Reads up to `bytes` into `into`: how many were read, 0 at the end.
    std::size_t read(void* into, std::size_t bytes);

    // Closes this end of the pipe, once the reading is over: a writer still
    // writing then fails, instead of waiting for room that would never come.
    void close();

    // Opens the pipe at `path` to write and closes it at once. A reader
    // waiting in connect() for a writer that never comes (FFmpeg opens no
    // output that gets no frame) then finds the pipe empty. Does nothing to
    // a pipe whose writer has it already, and, for a FIFO, to one whose
    // reader is not waiting in connect() yet.
    static void touch(const std::string& path);

private:
    std::string path_;
    // Each system uses its own of these.
    [[maybe_unused]] std::size_t buffer_bytes_ = 0;
    [[maybe_unused]] void* handle_ = nullptr;  // the pipe (Windows)
    [[maybe_unused]] void* timer_ = nullptr;   // a waitable timer for gather's short waits (Windows)
    [[maybe_unused]] int descriptor_ = -1;     // the FIFO, once its writer opened it (elsewhere)
};

}  // namespace vml::platform
