// The command line's own console: writing text out, reading its arguments,
// and Ctrl+C.
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vml::platform {

enum class Stream { output, error };

// Writes UTF-8 text to standard output or error. Text written to a
// Windows console shows every character; text written to a file or a pipe
// is UTF-8, with "\n" made "\r\n" on Windows. (The Python app's print
// writes Windows' ANSI code page there, with "?" for a character the code
// page lacks: UTF-8 keeps every file name as it is.)
void write_text(Stream stream, std::string_view text);

// Whether the stream is a terminal (where a progress line can be redrawn).
bool is_terminal(Stream stream);

// The program's arguments as UTF-8, without the program's own name (on
// Windows read from the wide command line, so every character survives).
std::vector<std::string> utf8_arguments(int argc, char** argv);

// Calls `handler` (once per press, on another thread) on Ctrl+C, and on
// Windows on Ctrl+Break and when the console window is closed.
void on_interrupt(std::function<void()> handler);

}  // namespace vml::platform
