// The system on Linux and macOS.
#include "vml/platform/system.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <thread>

namespace vml::platform {

std::optional<std::string> environment_variable(std::string_view name) {
    const char* value = std::getenv(std::string(name).c_str());
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string(value);
}

std::filesystem::path home_directory() { return environment_variable("HOME").value_or("/"); }

std::string executable_name(std::string_view name) { return std::string(name); }

std::string shared_library_name(std::string_view name) {
#ifdef __APPLE__
    return std::string(name) + ".dylib";
#else
    return std::string(name) + ".so";
#endif
}

std::optional<std::filesystem::path> find_on_path(std::string_view name) {
    const std::string path = environment_variable("PATH").value_or("");
    for (std::size_t start = 0; start <= path.size();) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) {
            end = path.size();
        }
        if (end > start) {
            std::filesystem::path candidate = std::filesystem::path(path.substr(start, end - start)) / name;
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error) && access(candidate.c_str(), X_OK) == 0) {
                return candidate;
            }
        }
        start = end + 1;
    }
    return std::nullopt;
}

unsigned logical_cores() {
    const unsigned cores = std::thread::hardware_concurrency();
    return cores == 0 ? 1 : cores;
}

std::vector<unsigned> gpu_vendor_ids() { return {}; }

std::optional<long long> modified_ns(const std::filesystem::path& file) {
    struct stat status {};
    if (stat(file.c_str(), &status) != 0) {
        return std::nullopt;
    }
#ifdef __APPLE__
    const timespec& written = status.st_mtimespec;
#else
    const timespec& written = status.st_mtim;
#endif
    return static_cast<long long>(written.tv_sec) * 1'000'000'000LL + written.tv_nsec;
}

std::string path_text(const std::filesystem::path& path) { return path.string(); }

std::filesystem::path path_from_text(std::string_view utf8) { return std::filesystem::path(utf8); }

}  // namespace vml::platform
