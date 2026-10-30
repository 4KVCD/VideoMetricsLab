// Pipes another program opens by their path as it opens a file. InboundPipe,
// which this process reads: how FFmpeg hands its frames over for VMAF on the
// GPU; a named pipe (\\.\pipe\...) on Windows, a FIFO in the temporary
// folder elsewhere. OutboundPipe, which this process writes: how FFmpeg gets
// pictures the app decodes itself; a named pipe, on Windows only.
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

// What a write that failed means: the reader closed its end (FFmpeg stopped
// reading), the system is short of memory for a moment (on Windows what
// waits in a pipe is nonpaged memory, refused now and then on a busy PC:
// worth waiting for and writing again), or anything else.
enum class PipeWriteError { reader_gone, try_again, failed };
// The meaning of Windows' error `code` from a write into a named pipe.
PipeWriteError pipe_write_error(unsigned long code);

// A pipe this process writes and another program reads, opening it by its
// path as it opens a file: how pictures the app decodes itself reach an
// FFmpeg as a rawvideo input. A named pipe; only on Windows (elsewhere making
// one throws PipeError).
class OutboundPipe {
public:
    // A path unique to this process for a pipe ending in `name`, made before
    // the pipe: by the process that starts the reader, for the writer to make.
    static std::string unique_path(std::string_view name);

    // A new pipe at `path` (unique_path), holding `buffer_bytes`. Throws
    // PipeError.
    OutboundPipe(const std::string& path, std::size_t buffer_bytes);
    ~OutboundPipe();
    OutboundPipe(const OutboundPipe&) = delete;
    OutboundPipe& operator=(const OutboundPipe&) = delete;

    // Waits for the reader to open the pipe; one that opened it and closed
    // it already (touch) counts, and the next write finds it gone. Throws
    // PipeError.
    void connect();

    // Writes `bytes` at `data`, in pieces of at most a megabyte, waiting and
    // writing again what the system refused for a moment: false once the
    // reader has closed its end, true when everything was written. Throws
    // PipeError on any other failure.
    bool write(const void* data, std::size_t bytes);

    // Waits until the reader has read everything written (false when it
    // closed its end first), then closes this end: the reader then finds the
    // end of its input.
    bool finish();

    // Opens the pipe at `path` to read and closes it at once: a writer
    // waiting in connect() for a reader that never comes (an FFmpeg that
    // failed before opening its inputs) goes on, and finds it gone. Does
    // nothing to a pipe whose reader has it.
    static void touch(const std::string& path);

private:
    struct State;  // pipe_windows.cpp
    std::unique_ptr<State> state_;
};

}  // namespace vml::platform
