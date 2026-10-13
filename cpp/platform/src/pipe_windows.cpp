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
      buffer_bytes_(buffer_bytes) {
    const DWORD size = static_cast<DWORD>(std::min<std::size_t>(buffer_bytes, 1u << 30));
    const HANDLE pipe =
        CreateNamedPipeW(widen(path_).c_str(), PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_WAIT, 1, size, size, 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) {
        throw PipeError("The pipe " + path_ + " could not be made (Windows error " + std::to_string(GetLastError()) +
                        ")");
    }
    handle_ = pipe;
    // gather's waits of a millisecond, as Python's time.sleep(0.001) makes
    // them: a plain Sleep(1) can last a whole 15.6 ms tick of the clock.
    timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (timer_ == nullptr) {  // before Windows 10 1803
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    }
}

InboundPipe::~InboundPipe() {
    close();
    if (timer_ != nullptr) {
        CloseHandle(timer_);
    }
}

void InboundPipe::connect() {
    if (ConnectNamedPipe(handle_, nullptr)) {
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
    const std::size_t wanted = std::min(bytes, buffer_bytes_ / 2);
    while (!stop) {
        DWORD available = 0;
        if (!PeekNamedPipe(handle_, nullptr, 0, nullptr, &available, nullptr)) {
            return;  // closed and empty: the next read says so
        }
        if (available >= wanted || writer_gone) {
            return;
        }
        LARGE_INTEGER due;
        due.QuadPart = -10'000;  // a millisecond from now, in 100 ns units
        if (timer_ != nullptr && SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE)) {
            WaitForSingleObject(timer_, INFINITE);
        } else {
            Sleep(1);
        }
    }
}

std::size_t InboundPipe::read(void* into, std::size_t bytes) {
    DWORD got = 0;
    const DWORD asked = static_cast<DWORD>(std::min<std::size_t>(bytes, 1u << 30));
    // It fails (ERROR_BROKEN_PIPE) once the writer has closed it and it is empty.
    if (handle_ == nullptr || !ReadFile(handle_, into, asked, &got, nullptr)) {
        return 0;
    }
    return got;
}

void InboundPipe::close() {
    if (handle_ != nullptr) {
        CloseHandle(handle_);
        handle_ = nullptr;
    }
}

void InboundPipe::touch(const std::string& path) {
    const HANDLE writer = CreateFileW(widen(path).c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (writer != INVALID_HANDLE_VALUE) {
        CloseHandle(writer);
    }
}

}  // namespace vml::platform
