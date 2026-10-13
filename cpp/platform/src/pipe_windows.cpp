// InboundPipe on Windows: a named pipe, as the Python app makes it
// (vmaf_gpu._StreamReader).
#include "vml/platform/pipe.h"

#include "text_windows.h"  // first: it sets <windows.h> up

#include <algorithm>
#include <random>

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace vml::platform {

struct InboundPipe::State {
    HANDLE pipe = nullptr;
    HANDLE timer = nullptr;  // for gather's waits of a millisecond
    std::size_t buffer_bytes = 0;
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

}  // namespace

InboundPipe::InboundPipe(std::string_view name, std::size_t buffer_bytes)
    : path_(R"(\\.\pipe\vml-vmaf-)" + std::to_string(GetCurrentProcessId()) + "-" + random_part() + "-" +
            std::string(name)),
      state_(std::make_unique<State>()) {
    state_->buffer_bytes = buffer_bytes;
    const DWORD size = static_cast<DWORD>(std::min<std::size_t>(buffer_bytes, 1u << 30));
    const HANDLE pipe =
        CreateNamedPipeW(widen(path_).c_str(), PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_WAIT, 1, size, size, 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) {
        throw PipeError("The pipe " + path_ + " could not be made (Windows error " + std::to_string(GetLastError()) +
                        ")");
    }
    state_->pipe = pipe;
    // gather's waits of a millisecond, as Python's time.sleep(0.001) makes
    // them: a plain Sleep(1) can last a whole 15.6 ms tick of the clock.
    state_->timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (state_->timer == nullptr) {  // before Windows 10 1803
        state_->timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    }
}

InboundPipe::~InboundPipe() {
    close();
    if (state_->timer != nullptr) {
        CloseHandle(state_->timer);
    }
}

void InboundPipe::connect() {
    if (ConnectNamedPipe(state_->pipe, nullptr)) {
        return;
    }
    // The writer opened the pipe first; or opened it, wrote and closed it
    // already. Neither is an error: what it wrote is still there to read.
    const DWORD error = GetLastError();
    if (error != ERROR_PIPE_CONNECTED && error != ERROR_NO_DATA) {
        throw PipeError("Waiting for the writer of " + path_ + " failed (Windows error " + std::to_string(error) + ")");
    }
}

void InboundPipe::gather(std::size_t bytes, const std::atomic<bool>& stop, const std::atomic<bool>& writer_gone) {
    const std::size_t wanted = std::min(bytes, state_->buffer_bytes / 2);
    while (!stop) {
        DWORD available = 0;
        if (!PeekNamedPipe(state_->pipe, nullptr, 0, nullptr, &available, nullptr)) {
            return;  // closed and empty: the next read says so
        }
        if (available >= wanted || writer_gone) {
            return;
        }
        LARGE_INTEGER due;
        due.QuadPart = -10'000;  // a millisecond from now, in 100 ns units
        if (state_->timer != nullptr && SetWaitableTimer(state_->timer, &due, 0, nullptr, nullptr, FALSE)) {
            WaitForSingleObject(state_->timer, INFINITE);
        } else {
            Sleep(1);
        }
    }
}

std::size_t InboundPipe::read(void* into, std::size_t bytes) {
    DWORD got = 0;
    const DWORD asked = static_cast<DWORD>(std::min<std::size_t>(bytes, 1u << 30));
    // It fails (ERROR_BROKEN_PIPE) once the writer has closed it and it is empty.
    if (state_->pipe == nullptr || !ReadFile(state_->pipe, into, asked, &got, nullptr)) {
        return 0;
    }
    return got;
}

void InboundPipe::close() {
    if (state_->pipe != nullptr) {
        CloseHandle(state_->pipe);
        state_->pipe = nullptr;
    }
}

void InboundPipe::touch(const std::string& path) {
    const HANDLE writer = CreateFileW(widen(path).c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (writer != INVALID_HANDLE_VALUE) {
        CloseHandle(writer);
    }
}

}  // namespace vml::platform
