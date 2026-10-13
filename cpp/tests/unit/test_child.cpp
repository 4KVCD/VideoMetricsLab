// The program the process tests start.
//
//     vml_test_child echo ARG...   each ARG on a line of its own, ended with
//                                  CR LF, in UTF-8; "to standard error" on
//                                  standard error; then exits with code 3
//     vml_test_child sleep         sleeps for a minute
//
// On Windows it reads its arguments as Unicode, as FFmpeg does, and writes
// bytes as they are (no text-mode line endings added).
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

namespace {

int run(const std::vector<std::string>& arguments) {
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
    return 1;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    _setmode(_fileno(stdout), _O_BINARY);
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
