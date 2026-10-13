// The system on Linux and macOS.
#include "vml/platform/system.h"

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <mach/mach.h>
#endif

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

std::optional<unsigned long long> available_memory() {
#ifdef __APPLE__
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    const kern_return_t result =
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics), &count);
    mach_port_deallocate(mach_task_self(), host);
    if (result != KERN_SUCCESS) {
        return std::nullopt;
    }
    const auto page = static_cast<unsigned long long>(sysconf(_SC_PAGESIZE));
    return (static_cast<unsigned long long>(statistics.free_count) + statistics.inactive_count) * page;
#else
    std::ifstream meminfo("/proc/meminfo");
    for (std::string line; std::getline(meminfo, line);) {
        if (line.starts_with("MemAvailable:")) {  // "MemAvailable:   12345678 kB"
            return std::strtoull(line.c_str() + 13, nullptr, 10) * 1024;
        }
    }
    return std::nullopt;
#endif
}

void raise_thread_priority() {}

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

void crash_quietly() {}  // a crash shows no dialog here

void end_process_now(int exit_code) { _exit(exit_code); }

int process_id() { return static_cast<int>(getpid()); }

bool process_running(int pid) {
    if (pid <= 0) {
        return pid == 0;  // as psutil counts it
    }
    return kill(pid, 0) == 0 || errno == EPERM;  // EPERM: it runs, but is not ours to signal
}

void close_native_handle(void* handle) { close(static_cast<int>(reinterpret_cast<std::intptr_t>(handle))); }

std::filesystem::path executable_path() {
#ifdef __APPLE__
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string path(size, '\0');
    if (_NSGetExecutablePath(path.data(), &size) != 0) {
        return {};
    }
    path.resize(std::strlen(path.c_str()));
    return std::filesystem::weakly_canonical(path);
#else
    std::error_code error;
    return std::filesystem::read_symlink("/proc/self/exe", error);
#endif
}

std::string path_text(const std::filesystem::path& path) { return path.string(); }

std::filesystem::path path_from_text(std::string_view utf8) { return std::filesystem::path(utf8); }

std::optional<std::string> fopen_path(const std::filesystem::path& path) { return path.string(); }

}  // namespace vml::platform
