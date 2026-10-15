// The system on Linux and macOS.
#include "vml/platform/system.h"

#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
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

std::optional<unsigned long long> total_memory() {
#ifdef __APPLE__
    std::uint64_t bytes = 0;
    std::size_t size = sizeof bytes;
    if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) != 0) {
        return std::nullopt;
    }
    return bytes;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) {
        return std::nullopt;
    }
    return static_cast<unsigned long long>(pages) * static_cast<unsigned long long>(page);
#endif
}

std::tm local_time(std::time_t time) {
    std::tm local{};
    localtime_r(&time, &local);
    return local;
}

std::string system_description() {
    utsname name{};
    if (uname(&name) != 0) {
        return "unknown";
    }
    return std::string(name.sysname) + "-" + name.release + "-" + name.machine;
}

std::string processor_description() {
    utsname name{};
    return uname(&name) == 0 ? std::string(name.machine) : std::string();
}

DisplayHdrInfo query_display_hdr(void* /*window*/) { return {}; }  // not known here: SDR at 100 nits

void raise_thread_priority() {}

bool keep_system_awake(bool /*awake*/) { return false; }

std::vector<unsigned> gpu_vendor_ids() { return {}; }

unsigned first_gpu_adapter() { return 0; }

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

namespace {

// The file crashes are reported in (report_crashes_to).
int crash_file = -1;

void write_crash_text(const char* text) {
    [[maybe_unused]] const ssize_t written = write(crash_file, text, std::strlen(text));
}

// Writes what happened and the calling stack, with only what a signal
// handler may call, then lets the signal end the process as it would have.
void report_signal(int signal_number) {
    write_crash_text(signal_number == SIGSEGV   ? "Fatal signal: segmentation fault\nStack:\n"
                     : signal_number == SIGBUS  ? "Fatal signal: bus error\nStack:\n"
                     : signal_number == SIGILL  ? "Fatal signal: illegal instruction\nStack:\n"
                     : signal_number == SIGFPE  ? "Fatal signal: arithmetic error\nStack:\n"
                                                : "Aborted\nStack:\n");
    void* frames[64];
    const int count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, crash_file);
    write_crash_text("\n");
    raise(signal_number);  // SA_RESETHAND made its handling the default again
}

}  // namespace

bool report_crashes_to(const std::filesystem::path& file) {
    crash_file = open(file.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (crash_file < 0) {
        return false;
    }
    // Loaded now: the first call may load the unwinder, which a signal
    // handler must not.
    void* warm[1];
    backtrace(warm, 1);
    struct sigaction action {};
    action.sa_handler = report_signal;
    action.sa_flags = SA_RESETHAND;
    sigemptyset(&action.sa_mask);
    for (const int signal_number : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) {
        sigaction(signal_number, &action, nullptr);
    }
    return true;
}

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
