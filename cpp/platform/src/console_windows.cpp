// The console on Windows.
#include "vml/platform/console.h"

#include "text_windows.h"  // first: it sets <windows.h> up

#include <shellapi.h>

#include <mutex>

namespace vml::platform {

namespace {

std::function<void()>& interrupt_handler() {
    static std::function<void()> handler;
    return handler;
}

BOOL WINAPI console_event(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
        if (interrupt_handler()) {
            interrupt_handler()();
        }
        if (event == CTRL_CLOSE_EVENT) {
            // Windows ends a closed console's process as soon as this returns,
            // or some 5 s on (SPI_GETHUNGAPPTIMEOUT): this thread waits, so that
            // the cancelled run writes its results as after Ctrl+C, and the
            // process's own exit ends this thread.
            Sleep(INFINITE);
        }
        return TRUE;  // handled: the run ends itself, after its cleanup
    }
    return FALSE;
}

HANDLE handle_of(Stream stream) {
    return GetStdHandle(stream == Stream::output ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
}

}  // namespace

void write_text(Stream stream, std::string_view text) {
    static std::mutex writing;
    const std::lock_guard lock(writing);
    const HANDLE handle = handle_of(stream);
    DWORD mode = 0;
    if (GetConsoleMode(handle, &mode)) {
        // A console: as UTF-16, which shows every character whatever the
        // console's code page.
        const std::wstring wide = widen(text);
        DWORD written = 0;
        WriteConsoleW(handle, wide.data(), static_cast<DWORD>(wide.size()), &written, nullptr);
        return;
    }
    // A file or a pipe: UTF-8 bytes, lines ended with CR LF.
    std::string bytes;
    bytes.reserve(text.size());
    for (const char c : text) {
        if (c == '\n') {
            bytes += '\r';
        }
        bytes += c;
    }
    DWORD written = 0;
    WriteFile(handle, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr);
}

bool is_terminal(Stream stream) {
    DWORD mode = 0;
    return GetConsoleMode(handle_of(stream), &mode) != 0;
}

std::vector<std::string> utf8_arguments(int, char**) {
    int count = 0;
    LPWSTR* wide = CommandLineToArgvW(GetCommandLineW(), &count);
    std::vector<std::string> arguments;
    for (int i = 1; i < count; ++i) {
        arguments.push_back(narrow(wide[i]));
    }
    LocalFree(wide);
    return arguments;
}

void on_interrupt(std::function<void()> handler) {
    interrupt_handler() = std::move(handler);
    SetConsoleCtrlHandler(console_event, TRUE);
}

}  // namespace vml::platform
