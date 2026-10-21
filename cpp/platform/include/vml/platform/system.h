// What the app asks of the system it runs on: folders, the PATH, the
// environment, the CPU and the GPUs.
#pragma once

#include <ctime>
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

// From now on, a crash of this process (an access violation, an abort) is
// written to the end of `file` before the process ends: what happened, where
// (module and offset), the thread, and the calling stack, as modules and
// offsets (the Python app's faulthandler). False when the file cannot be
// opened.
bool report_crashes_to(const std::filesystem::path& file);

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

// How many bytes of physical memory the PC has, or nothing when the system
// does not say.
std::optional<unsigned long long> total_memory();

// The local time of `time` (seconds since 1970), for the log's timestamps.
std::tm local_time(std::time_t time);

// The system, as the Python app's log names it: on Windows Python's
// platform.platform() form, such as "Windows-11-10.0.26300-SP0"; elsewhere
// the kernel's name, release and machine, such as "Linux-6.8.0-x86_64".
std::string system_description();

// The processor, as Python's platform.processor() names it: on Windows the
// PROCESSOR_IDENTIFIER Windows sets, such as "Intel64 Family 6 Model 198
// Stepping 2, GenuineIntel"; elsewhere the machine, such as "x86_64".
std::string processor_description();

// The HDR state of a monitor, with nothing for what Windows does not say.
struct DisplayHdrInfo {
    std::string device_name;
    std::optional<bool> hdr_supported;
    std::optional<bool> hdr_enabled;
    std::optional<int> bits_per_color_channel;
    // Windows' SDR content brightness: the white SDR is shown at while HDR
    // is on.
    std::optional<double> sdr_white_nits;
    bool operator==(const DisplayHdrInfo&) const = default;
};

// The HDR state of the monitor nearest `window` (a native window handle),
// as Windows' display configuration says it (the Python app's
// display_hdr.py); all unknown elsewhere, or when a driver does not say.
// Only read: nothing about the display is changed.
DisplayHdrInfo query_display_hdr(void* window);

// Lets the calling thread run ahead of ordinary work (Windows'
// THREAD_PRIORITY_HIGHEST), for a light thread whose timing matters. It does
// nothing on other systems, as in the Python app, or if Windows refuses.
void raise_thread_priority();

// Holds (`awake`) or releases Windows' "system required" request, which
// keeps the PC from going to sleep while a run takes hours; the screen may
// still turn off (the Python app's power.keep_system_awake). The request is
// the calling thread's, and ends with it at the latest. On macOS a power
// assertion against idle sleep (the process's), which also keeps App Nap
// off. False where nothing was done: Linux, or the system refused.
bool keep_system_awake(bool awake);

// The PCI vendor IDs of the GPU makers the app knows. AMD's GPUs report
// either of its two.
inline constexpr unsigned nvidia_pci_vendor = 0x10DE;
inline constexpr unsigned intel_pci_vendor = 0x8086;
inline constexpr unsigned amd_pci_vendors[] = {0x1002, 0x1022};

// The PCI vendor IDs of the hardware GPUs DirectX lists (Windows only; empty
// elsewhere). With
// VML_GPU_VENDOR set to nvidia, intel or amd, only that maker's, as on a PC
// with no other: for testing one maker's route on a PC that has several
// (CUDA_VISIBLE_DEVICES and VK_LOADER_DRIVERS_DISABLE hide the others from
// CUDA and Vulkan).
std::vector<unsigned> gpu_vendor_ids();
// Whether this system lists its GPUs in gpu_vendor_ids: Windows does; on
// macOS and Linux the list is empty whatever GPU there is.
bool lists_gpus();
// DirectX's number for the GPU that Video Compare decodes and draws on: 0,
// the first, unless VML_GPU_VENDOR names another maker's; nothing when it
// names a maker with no GPU here (as with no GPU, none to draw on).
std::optional<unsigned> first_gpu_adapter();

// The GPU memory free on the GPU named `name` (as DirectX names it, which is
// how Vship and the drivers name it too): its own memory less what every
// process holds, and no more than Windows' budget for this process. Nothing
// elsewhere, when no GPU has that name or Windows does not say, or for a GPU
// with less than 1 GB of its own memory, which shares the PC's (integrated).
std::optional<unsigned long long> gpu_memory_available(std::string_view name);

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
