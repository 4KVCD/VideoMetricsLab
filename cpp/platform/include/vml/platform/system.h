// What the app asks of the system it runs on: folders, the PATH, the
// environment, the CPU and the GPUs.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vml::platform {

// The user's home folder, as Python's Path.home() finds it: USERPROFILE on
// Windows, HOME elsewhere.
std::filesystem::path home_directory();

// This program's own file, to start it again (for work done in a process
// of its own).
std::filesystem::path executable_path();

// Makes a crash end this process at once, without the system's dialog
// waiting on the user's screen (Windows), for a process whose crash its
// parent reports.
void crash_quietly();

// Ends this process at once with `exit_code`, without unloading its
// libraries: for a process one of whose threads may hold a driver's locks
// forever, which a normal exit could wait on.
[[noreturn]] void end_process_now(int exit_code);

// This process's id.
int process_id();

// Whether a process with id `pid` is running, as psutil's pid_exists says:
// one that runs but may not be looked at counts, and so does 0.
bool process_running(int pid);

// Closes a handle a library gave this process to pass on, such as the Vulkan
// engine's GPU memory exported for a decoder to import: a HANDLE on Windows,
// a file descriptor elsewhere.
void close_native_handle(void* handle);

// An environment variable's value, or nothing when it is not set.
std::optional<std::string> environment_variable(std::string_view name);

// A program's file name on this system: "ffmpeg.exe" on Windows, "ffmpeg"
// elsewhere.
std::string executable_name(std::string_view name);

// A shared library's file name on this system: "libvmaf.dll" on Windows,
// "libvmaf.so" on Linux, "libvmaf.dylib" on macOS, for "libvmaf".
std::string shared_library_name(std::string_view name);

// The program `name` as Python's shutil.which finds it: in the current
// folder and then each folder on the PATH on Windows (with PATHEXT's
// extensions), on the PATH elsewhere. Nothing when it is not found.
std::optional<std::filesystem::path> find_on_path(std::string_view name);

// How many logical cores the CPU has (Python's os.cpu_count()), at least 1.
unsigned logical_cores();

// How many bytes of memory programs can be given now without the system
// swapping, as psutil's virtual_memory().available gives it: the available
// physical memory on Windows, MemAvailable on Linux, the free and inactive
// pages on macOS. Nothing when the system does not say.
std::optional<unsigned long long> available_memory();

// Lets the calling thread run ahead of ordinary work (Windows'
// THREAD_PRIORITY_HIGHEST), for a light thread whose timing matters. It does
// nothing on other systems, as in the Python app, or if Windows refuses.
void raise_thread_priority();

// The PCI vendor IDs of the hardware GPUs DirectX lists (Windows only; empty
// elsewhere): 0x10DE NVIDIA, 0x1002 and 0x1022 AMD, 0x8086 Intel.
std::vector<unsigned> gpu_vendor_ids();

// When a file was last written, in nanoseconds since 1970 (Python's
// st_mtime_ns), or nothing when it cannot be read.
std::optional<long long> modified_ns(const std::filesystem::path& file);

// The whole of a file, in one read, as Python's read_bytes() reads it, or
// nothing when it cannot be read. Read byte by byte through a stream, a 20 MB
// picture of a 4K frame took 64 ms rather than 15.
std::optional<std::string> read_file(const std::filesystem::path& path);

// A path as UTF-8 text, and back; the app keeps paths as UTF-8.
std::string path_text(const std::filesystem::path& path);
std::filesystem::path path_from_text(std::string_view utf8);

// An existing file's path as a library that opens files with fopen() needs
// it (libvmaf): on Windows in the ANSI code page, or, for a name with
// letters the code page lacks, as its short 8.3 path, where the drive keeps
// those; nothing when neither names it. Elsewhere, the path itself.
std::optional<std::string> fopen_path(const std::filesystem::path& path);

}  // namespace vml::platform
