// The program the process tests start.
//
//     vml_test_child echo ARG...   each ARG on a line of its own, ended with
//                                  CR LF, in UTF-8; "to standard error" on
//                                  standard error; then exits with code 3
//     vml_test_child sleep         sleeps for a minute
//     vml_test_child warn N        "warning i of N" on N lines of standard
//                                  error, then "the last line"
//
// With the environment variable VML_TEST_TOOL set, it stands in for
// libjxl's tools instead, started as the app starts them (REFERENCE TEST,
// then Butteraugli's options), and for --version prints "test tool 1.0":
//
//     VML_TEST_TOOL=score   prints the number of the test picture's pair
//                           as its score (test-00000007.png scores 7); with
//                           --rawdistmap MAP, writes a distortion map whose
//                           3-norm is that number. It writes the pictures'
//                           folder into a file of its own in the folder
//                           VML_TEST_TOOL_LOG names.
//     VML_TEST_TOOL=hang    writes a file of its own into the folder
//                           VML_TEST_TOOL_LOG names, then sleeps for ten
//                           minutes
//
// With VML_TEST_FFMPEG_PICTURE set, it stands in for FFmpeg writing the CPU
// tools' frame pairs (png_pairs_command): it writes VML_TEST_FFMPEG_PAIRS
// pairs, each picture a copy of the picture VML_TEST_FFMPEG_PICTURE names,
// written under a temporary name and then renamed, as -atomic_writing does.
// On Windows it holds the first pair's second picture for
// VML_TEST_FFMPEG_HOLD_MS milliseconds once it has its name, as FFmpeg can
// (elsewhere, a file held open does not keep others from reading it).
//
// On Windows it reads its arguments as Unicode, as FFmpeg does, and writes
// bytes as they are (no text-mode line endings added).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

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

// libjxl's tools, stood in for.
int tool(const std::string& mode, const std::vector<std::string>& arguments) {
    if (arguments.size() == 2 && arguments[1] == "--version") {
        std::fputs("test tool 1.0\n", stdout);
        return 0;
    }
    if (arguments.size() < 3) {
        return 1;
    }
    if (mode == "hang") {
        log_file("started", "started");
        std::this_thread::sleep_for(std::chrono::minutes(10));
        return 0;
    }
    const std::filesystem::path test = path_of(arguments[2]);
    // "test-00000007.png" is pair 7.
    const std::string name = test.filename().string();
    const int number = std::atoi(name.substr(name.find('-') + 1).c_str());
    const std::u8string folder = test.parent_path().u8string();
    log_file("folder", std::string(folder.begin(), folder.end()));
    for (std::size_t i = 3; i + 1 < arguments.size(); ++i) {
        if (arguments[i] == "--rawdistmap") {
            // One value, little-endian: its 3-norm is the value.
            const auto value = static_cast<float>(number);
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof bits);
            std::string map = "Pf\n1 1\n-1.0\n";
            for (const int shift : {0, 8, 16, 24}) {
                map.push_back(static_cast<char>((bits >> shift) & 0xFFu));
            }
            std::ofstream(path_of(arguments[i + 1]), std::ios::binary) << map;
        }
    }
    std::printf("score: %d\n", number);
    return 0;
}

// Holds the file `path` open for `milliseconds`, as FFmpeg can hold a
// picture it has just renamed: with delete access, which keeps the readers
// (who do not share deleting) from opening it.
void hold(const std::filesystem::path& path, int milliseconds) {
#ifdef _WIN32
    const HANDLE file = CreateFileW(path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, 0, nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    if (file != INVALID_HANDLE_VALUE) {
        CloseHandle(file);
    }
#else
    static_cast<void>(path);
    static_cast<void>(milliseconds);
#endif
}

// FFmpeg writing the frame pairs, stood in for.
int ffmpeg(const std::vector<std::string>& arguments) {
    std::ifstream source(path_of(variable("VML_TEST_FFMPEG_PICTURE")), std::ios::binary);
    const std::string picture{std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
    std::vector<std::string> patterns;  // the outputs, "...\test-%08d.png" and "...\reference-%08d.png"
    for (const std::string& argument : arguments) {
        if (argument.ends_with("%08d.png")) {
            patterns.push_back(argument.substr(0, argument.size() - std::strlen("%08d.png")));
        }
    }
    const int pairs = std::atoi(variable("VML_TEST_FFMPEG_PAIRS").c_str());
    for (int number = 1; number <= pairs; ++number) {
        for (const std::string& pattern : patterns) {
            char digits[16];
            std::snprintf(digits, sizeof digits, "%08d", number);
            const std::filesystem::path name = path_of(pattern + digits + ".png");
            std::filesystem::path temporary = name;
            temporary += ".tmp";
            std::ofstream(temporary, std::ios::binary) << picture;
            std::filesystem::rename(temporary, name);
            if (number == 1 && pattern == patterns.back()) {
                hold(name, std::atoi(variable("VML_TEST_FFMPEG_HOLD_MS").c_str()));
            }
        }
    }
    return 0;
}

int run(const std::vector<std::string>& arguments) {
    // While it stands in for FFmpeg it stands in for the tools too: FFmpeg's
    // command is the one that writes pictures atomically.
    if (!variable("VML_TEST_FFMPEG_PICTURE").empty() &&
        std::find(arguments.begin(), arguments.end(), "-atomic_writing") != arguments.end()) {
        return ffmpeg(arguments);
    }
    if (const std::string mode = variable("VML_TEST_TOOL"); !mode.empty()) {
        return tool(mode, arguments);
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
    if (arguments.size() >= 3 && arguments[1] == "warn") {
        const int count = std::stoi(arguments[2]);
        for (int i = 0; i < count; ++i) {
            const std::string line = "warning " + std::to_string(i) + " of " + arguments[2] + "\n";
            std::fwrite(line.data(), 1, line.size(), stderr);
        }
        std::fputs("the last line\n", stderr);
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
