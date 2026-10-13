// The console on Linux and macOS.
#include "vml/platform/console.h"

#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <mutex>
#include <thread>

namespace vml::platform {

namespace {

std::function<void()>& interrupt_handler() {
    static std::function<void()> handler;
    return handler;
}

// A signal handler may do almost nothing, so it writes a byte to this pipe,
// and a thread of ours, waiting on it, calls the handler.
int interrupt_pipe[2] = {-1, -1};

extern "C" void on_signal(int) {
    const char byte = 1;
    [[maybe_unused]] const ssize_t written = write(interrupt_pipe[1], &byte, 1);
}

}  // namespace

void write_text(Stream stream, std::string_view text) {
    static std::mutex writing;
    const std::lock_guard lock(writing);
    FILE* file = stream == Stream::output ? stdout : stderr;
    std::fwrite(text.data(), 1, text.size(), file);
    std::fflush(file);
}

bool is_terminal(Stream stream) { return isatty(stream == Stream::output ? STDOUT_FILENO : STDERR_FILENO) != 0; }

std::vector<std::string> utf8_arguments(int argc, char** argv) {
    return std::vector<std::string>(argv + (argc > 0 ? 1 : 0), argv + argc);
}

void on_interrupt(std::function<void()> handler) {
    interrupt_handler() = std::move(handler);
    if (interrupt_pipe[0] < 0) {
        if (pipe(interrupt_pipe) != 0) {
            return;  // Ctrl+C keeps its default: it ends the program
        }
        std::thread([] {
            char byte = 0;
            while (true) {
                const ssize_t count = read(interrupt_pipe[0], &byte, 1);
                if (count == 1 && interrupt_handler()) {
                    interrupt_handler()();
                } else if (count < 0 && errno != EINTR) {
                    return;
                }
            }
        }).detach();
    }
    struct sigaction action {};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;  // calls a signal interrupts carry on
    sigaction(SIGINT, &action, nullptr);
}

}  // namespace vml::platform
