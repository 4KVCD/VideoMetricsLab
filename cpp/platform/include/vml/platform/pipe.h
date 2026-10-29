// A pipe this process reads and another program writes, opening it by its
// path as it opens a file: how FFmpeg hands its frames over for VMAF on the
// GPU. A named pipe (\\.\pipe\...) on Windows, a FIFO in the temporary
// folder elsewhere.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vml::platform {

// The pipe could not be made, or its writer could not be waited for.
class PipeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// How much a pipe that brings an FFmpeg's decoded frames holds (this one's,
// or a ChildProcess's standard output). On Windows what waits in a pipe is
// the writer's nonpaged memory, and a busy PC refuses FFmpeg's writes now
// and then (ERROR_NO_SYSTEM_RESOURCES, FFmpeg's "Invalid argument"
// mid-frame): with whole frames waiting in 64 MB pipes, far more often than
// with 8 MB. Read with read_full, which takes each half-pipe as it comes,
// 8 MB takes a 4K 10-bit stream at least as fast as 64 MB did.
inline constexpr std::size_t frame_pipe_bytes = 8 * 1024 * 1024;

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

    // Waits until `bytes` are in the pipe (or half of what it holds), until
    // `writer_gone` says no more will come, or until `stop` is set, so that
    // a read takes them at once instead of in the pieces they arrive in.
    // Elsewhere than on Windows a FIFO holds too little for that, and this
    // returns at once.
    void gather(std::size_t bytes, const std::atomic<bool>& stop, const std::atomic<bool>& writer_gone);

    // Reads up to `bytes` into `into`: how many were read, 0 at the end.
    std::size_t read(void* into, std::size_t bytes);

    // Fills `bytes` at `into`, gathering before each read: a frame comes in
    // a few large reads, however much bigger than the pipe it is. How many
    // were read: fewer only at the end.
    std::size_t read_full(void* into, std::size_t bytes, const std::atomic<bool>& stop,
                          const std::atomic<bool>& writer_gone);

    // Closes this end of the pipe, once the reading is over: a writer still
    // writing then fails, instead of waiting for room that would never come.
    void close();

    // Opens the pipe at `path` to write and closes it at once. A reader
    // waiting in connect() for a writer that never comes (FFmpeg opens no
    // output that gets no frame) then finds the pipe empty. Does nothing to
    // a pipe whose writer has it already, and, for a FIFO, to one whose
    // reader is not waiting in connect() yet.
    static void touch(const std::string& path);

    // Removes the pipes a process that has ended left: its FIFOs (named by
    // its id), which an ending it did not choose (a Cancel's SIGKILL) left in
    // the temporary folder. Windows' pipes go with their last handle: there
    // it does nothing.
    static void remove_left_by(int process_id);

private:
    std::string path_;
    struct State;  // each system's own (pipe_windows.cpp, pipe_posix.cpp)
    std::unique_ptr<State> state_;
};

}  // namespace vml::platform
