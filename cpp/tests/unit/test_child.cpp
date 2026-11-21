// The program the process tests start.
//
//     vml_test_child echo ARG...   each ARG on a line of its own, ended with
//                                  CR LF, in UTF-8; "to standard error" on
//                                  standard error; then exits with code 3
//     vml_test_child sleep         sleeps for a minute
//     vml_test_child hold FIFO     (not Windows) writes a byte into the FIFO,
//                                  keeps it open for a minute, then exits
//     vml_test_child spawn-hold FIFO  (not Windows) starts "hold FIFO" as the
//                                  app starts its programs (a group of its
//                                  own, the reaper told), then sleeps for a
//                                  minute
//     vml_test_child warn N        "warning i of N" on N lines of standard
//                                  error, then "the last line"
//     vml_test_child data N        N bytes on standard output, byte i being
//                                  i modulo 251, in writes of 32 KiB as
//                                  FFmpeg's
//
// With VML_TEST_PLAYBACK set, it stands in for FFmpeg decoding a video for
// Video Compare's playback (build_playback_command). It tells the way of
// decoding from its arguments (interop: a CUDA device; transfer: a Vulkan
// device and -hwaccel; software: a Vulkan device alone; cpu: neither), and
// does what VML_TEST_PLAYBACK says for that way, "way=FRAMES:EXIT:PARTIAL:
// MESSAGE" each, separated by ";": FRAMES frames of VML_TEST_FRAME_BYTES
// bytes, each filled with its number (modulo 251), counted from the frame
// its -ss gives at VML_TEST_FPS frames a second; then PARTIAL bytes of one
// more; MESSAGE on standard error; and exits with EXIT. It writes "way
// start" into a file of its own in VML_TEST_TOOL_LOG.
//
// On Windows it reads its arguments as Unicode, as FFmpeg does, and writes
// bytes as they are (no text-mode line endings added).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>

#include "vml/platform/process.h"
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

// A path from UTF-8 text, as the arguments are.
std::filesystem::path path_of(const std::string& utf8) {
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

std::string process_number() {
#ifdef _WIN32
    return std::to_string(GetCurrentProcessId());
#else
    return std::to_string(getpid());
#endif
}

// An environment variable's value in UTF-8; empty when it is not set.
std::string variable(const char* name) {
#ifdef _WIN32
    const std::wstring wide_name(name, name + std::strlen(name));  // the names here are ASCII
    const DWORD size = GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
    if (size == 0) {
        return {};
    }
    std::wstring value(size, L'\0');
    value.resize(GetEnvironmentVariableW(wide_name.c_str(), value.data(), size));
    const int length = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0,
                                           nullptr, nullptr);
    std::string utf8(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), utf8.data(), length, nullptr,
                        nullptr);
    return utf8;
#else
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
#endif
}

// A file of this process's own in the folder VML_TEST_TOOL_LOG names,
// holding `text`.
void log_file(const std::string& name, const std::string& text) {
    const std::string folder = variable("VML_TEST_TOOL_LOG");
    if (!folder.empty()) {
        std::ofstream(path_of(folder) / (name + "-" + process_number() + ".txt"), std::ios::binary) << text;
    }
}

// FFmpeg decoding for playback, stood in for.
int playback(const std::vector<std::string>& arguments) {
    const auto has = [&arguments](const std::string& argument) {
        return std::find(arguments.begin(), arguments.end(), argument) != arguments.end();
    };
    const std::string way = has("cuda=decode")      ? "interop"
                            : has("vulkan=preview") ? (has("-hwaccel") ? "transfer" : "software")
                                                    : "cpu";
    const auto seek = std::find(arguments.begin(), arguments.end(), "-ss");
    const double seconds = seek != arguments.end() && seek + 1 != arguments.end() ? std::atof((seek + 1)->c_str()) : 0;
    const auto start = static_cast<long long>(std::llround(seconds * std::atof(variable("VML_TEST_FPS").c_str()) + 0.125));
    log_file("way", way + " " + std::to_string(start));
    // This way's part of the spec: "way=FRAMES:EXIT:PARTIAL:MESSAGE".
    const std::string spec = variable("VML_TEST_PLAYBACK");
    const std::size_t at = spec.find(way + "=");
    if (at == std::string::npos) {
        return 1;
    }
    const std::string part = spec.substr(at + way.size() + 1, spec.find(';', at) - at - way.size() - 1);
    std::vector<std::string> fields;
    for (std::size_t from = 0, to = 0; to != std::string::npos; from = to + 1) {
        to = part.find(':', from);
        fields.push_back(part.substr(from, to == std::string::npos ? std::string::npos : to - from));
    }
    fields.resize(4);
    const long long frames = std::atoll(fields[0].c_str());
    const auto bytes = static_cast<std::size_t>(std::atoll(variable("VML_TEST_FRAME_BYTES").c_str()));
    for (long long frame = start; frame < start + frames; ++frame) {
        const std::vector<char> picture(bytes, static_cast<char>(frame % 251));
        std::fwrite(picture.data(), 1, picture.size(), stdout);
    }
    const std::vector<char> partial(static_cast<std::size_t>(std::atoll(fields[2].c_str())), 'x');
    std::fwrite(partial.data(), 1, partial.size(), stdout);
    std::fflush(stdout);
    if (!fields[3].empty()) {
        std::fputs((fields[3] + "\n").c_str(), stderr);
    }
    return std::atoi(fields[1].c_str());
}

int run(const std::vector<std::string>& arguments) {
    if (!variable("VML_TEST_PLAYBACK").empty()) {
        return playback(arguments);
    }
    if (arguments.size() >= 2 && arguments[1] == "echo") {
        for (std::size_t i = 2; i < arguments.size(); ++i) {
            const std::string line = arguments[i] + "\r\n";
            std::fwrite(line.data(), 1, line.size(), stdout);
        }
        std::fputs("to standard error\n", stderr);
        return 3;
    }
    if (arguments.size() >= 2 && arguments[1] == "sleep") {
        std::this_thread::sleep_for(std::chrono::minutes(1));
        return 0;
    }
#ifndef _WIN32
    if (arguments.size() >= 3 && arguments[1] == "hold") {
        const int fifo = open(arguments[2].c_str(), O_WRONLY);
        if (fifo < 0 || write(fifo, "h", 1) != 1) {
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::minutes(1));
        return 0;
    }
    if (arguments.size() >= 3 && arguments[1] == "spawn-hold") {
        const vml::platform::ChildProcess held =
            vml::platform::ChildProcess::start({arguments[0], "hold", arguments[2]});
        std::this_thread::sleep_for(std::chrono::minutes(1));
        return 0;
    }
#endif
    if (arguments.size() >= 3 && arguments[1] == "warn") {
        const int count = std::stoi(arguments[2]);
        for (int i = 0; i < count; ++i) {
            const std::string line = "warning " + std::to_string(i) + " of " + arguments[2] + "\n";
            std::fwrite(line.data(), 1, line.size(), stderr);
        }
        std::fputs("the last line\n", stderr);
        return 0;
    }
    if (arguments.size() >= 3 && arguments[1] == "data") {
        const auto count = static_cast<std::size_t>(std::atoll(arguments[2].c_str()));
        std::vector<char> piece(32 * 1024);
        for (std::size_t written = 0; written < count;) {
            const std::size_t size = (std::min)(piece.size(), count - written);  // (): windows.h's min
            for (std::size_t i = 0; i < size; ++i) {
                piece[i] = static_cast<char>((written + i) % 251);
            }
            std::fwrite(piece.data(), 1, size, stdout);
            std::fflush(stdout);
            written += size;
        }
        return 0;
    }
    return 1;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        const int length = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string utf8(static_cast<std::size_t>(length), '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, utf8.data(), length, nullptr, nullptr);
        utf8.pop_back();  // the terminating zero
        arguments.push_back(utf8);
    }
    return run(arguments);
}
#else
int main(int argc, char** argv) { return run(std::vector<std::string>(argv, argv + argc)); }
#endif
