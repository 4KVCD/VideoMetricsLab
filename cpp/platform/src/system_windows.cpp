// The system on Windows.
#include "vml/platform/system.h"

#include "text_windows.h"  // before <dxgi.h>: it sets <windows.h> up

#include <dxgi1_4.h>
#include <pdh.h>

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <thread>
#include <utility>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace vml::platform {

std::optional<std::string> environment_variable(std::string_view name) {
    const std::wstring wide_name = widen(name);
    const DWORD length = GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
    if (length == 0) {
        return std::nullopt;
    }
    std::wstring value(length, L'\0');
    const DWORD written = GetEnvironmentVariableW(wide_name.c_str(), value.data(), length);
    value.resize(written);
    return narrow(value);
}

std::filesystem::path home_directory() {
    if (const std::optional<std::string> profile = environment_variable("USERPROFILE")) {
        return path_from_text(*profile);
    }
    const std::string drive = environment_variable("HOMEDRIVE").value_or("");
    return path_from_text(drive + environment_variable("HOMEPATH").value_or(""));
}

std::string executable_name(std::string_view name) { return std::string(name) + ".exe"; }

std::string shared_library_name(std::string_view name) { return std::string(name) + ".dll"; }

std::optional<std::filesystem::path> find_on_path(std::string_view name) {
    // The extensions to try, from PATHEXT; none when the name has one of them.
    std::vector<std::wstring> extensions;
    std::wstring wanted = widen(name);
    std::wstring lowered = wanted;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](wchar_t c) { return std::towlower(c); });
    const std::wstring pathext = widen(environment_variable("PATHEXT").value_or(".COM;.EXE;.BAT;.CMD"));
    bool has_extension = false;
    for (std::size_t start = 0; start <= pathext.size();) {
        const std::size_t end = std::min(pathext.find(L';', start), pathext.size());
        std::wstring extension = pathext.substr(start, end - start);
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](wchar_t c) { return std::towlower(c); });
        if (!extension.empty()) {
            extensions.push_back(extension);
            if (lowered.size() > extension.size() && lowered.ends_with(extension)) {
                has_extension = true;
            }
        }
        start = end + 1;
    }
    if (has_extension) {
        extensions.assign(1, L"");
    }
    std::vector<std::filesystem::path> folders{std::filesystem::current_path()};
    const std::wstring path = widen(environment_variable("PATH").value_or(""));
    for (std::size_t start = 0; start <= path.size();) {
        const std::size_t end = std::min(path.find(L';', start), path.size());
        if (end > start) {
            folders.emplace_back(path.substr(start, end - start));
        }
        start = end + 1;
    }
    for (const std::filesystem::path& folder : folders) {
        for (const std::wstring& extension : extensions) {
            std::filesystem::path candidate = folder / (wanted + extension);
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error)) {
                return candidate;
            }
        }
    }
    return std::nullopt;
}

unsigned logical_cores() { return std::max(1u, std::thread::hardware_concurrency()); }

std::optional<unsigned long long> available_memory() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    if (!GlobalMemoryStatusEx(&status)) {
        return std::nullopt;
    }
    return status.ullAvailPhys;
}

std::optional<unsigned long long> total_memory() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    if (!GlobalMemoryStatusEx(&status)) {
        return std::nullopt;
    }
    return status.ullTotalPhys;
}

std::tm local_time(std::time_t time) {
    std::tm local{};
    localtime_s(&local, &time);
    return local;
}

std::string system_description() {
    // ntdll's RtlGetVersion gives the real version whatever this program's
    // manifest says (GetVersionEx gives the version the manifest names).
    using GetVersion = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    // Through void (*)(), which any function type may be cast from and to.
    const auto get_version = ntdll == nullptr ? nullptr
                                              : reinterpret_cast<GetVersion>(reinterpret_cast<void (*)()>(
                                                    GetProcAddress(ntdll, "RtlGetVersion")));
    RTL_OSVERSIONINFOEXW version{};
    version.dwOSVersionInfoSize = sizeof version;
    if (get_version == nullptr || get_version(reinterpret_cast<PRTL_OSVERSIONINFOW>(&version)) != 0) {
        return "Windows";
    }
    // Windows 11 still says 10.0, with builds from 22000.
    const std::string release = version.dwMajorVersion == 10 && version.dwBuildNumber >= 22000
                                    ? "11"
                                    : std::to_string(version.dwMajorVersion);
    return "Windows-" + release + "-" + std::to_string(version.dwMajorVersion) + "." +
           std::to_string(version.dwMinorVersion) + "." + std::to_string(version.dwBuildNumber) + "-SP" +
           std::to_string(version.wServicePackMajor);
}

std::string processor_description() { return environment_variable("PROCESSOR_IDENTIFIER").value_or(""); }

namespace {

// The display configuration's requests and answers, laid out as Windows
// lays them out (wingdi.h), for the two that older SDKs lack.
struct DeviceInfoHeader {
    UINT32 type;
    UINT32 size;
    LUID adapter;
    UINT32 id;
};

struct AdvancedColorInfo {
    DeviceInfoHeader header;
    UINT32 value;  // bit 0: HDR supported; bit 1: HDR on
    UINT32 color_encoding;
    UINT32 bits_per_color_channel;
};

struct SdrWhiteLevel {
    DeviceInfoHeader header;
    ULONG level;  // in thousandths of 80 nits
};

constexpr UINT32 get_source_name = 1;      // DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME
constexpr UINT32 get_advanced_color = 9;   // DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO
constexpr UINT32 get_sdr_white_level = 11;  // DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL

LONG device_info(void* request) {
    return DisplayConfigGetDeviceInfo(static_cast<DISPLAYCONFIG_DEVICE_INFO_HEADER*>(request));
}

}  // namespace

DisplayHdrInfo query_display_hdr(void* window) {
    DisplayHdrInfo info;
    const HMONITOR monitor = MonitorFromWindow(static_cast<HWND>(window), MONITOR_DEFAULTTONEAREST);
    MONITORINFOEXW monitor_info{};
    monitor_info.cbSize = sizeof monitor_info;
    if (monitor == nullptr || !GetMonitorInfoW(monitor, &monitor_info)) {
        return info;
    }

    // The active display paths; asked again if they change in between. As
    // in the Python app, nothing is known, not even the monitor's name, if
    // they cannot be listed.
    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    bool listed = false;
    for (int attempt = 0; attempt < 3 && !listed; ++attempt) {
        UINT32 path_count = 0;
        UINT32 mode_count = 0;
        if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS) {
            return info;
        }
        paths.resize(path_count);
        modes.resize(mode_count);
        const LONG result =
            QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr);
        if (result == ERROR_INSUFFICIENT_BUFFER) {
            continue;
        }
        if (result != ERROR_SUCCESS) {
            return info;
        }
        paths.resize(path_count);
        listed = true;
    }
    if (!listed) {
        return info;
    }
    info.device_name = narrow(monitor_info.szDevice);

    for (const DISPLAYCONFIG_PATH_INFO& path : paths) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
        source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source.header.size = sizeof source;
        source.header.adapterId = path.sourceInfo.adapterId;
        source.header.id = path.sourceInfo.id;
        static_assert(DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME == get_source_name);
        if (device_info(&source.header) != ERROR_SUCCESS ||
            CompareStringOrdinal(source.viewGdiDeviceName, -1, monitor_info.szDevice, -1, TRUE) != CSTR_EQUAL) {
            continue;
        }
        AdvancedColorInfo advanced{};
        advanced.header = {get_advanced_color, sizeof advanced, path.targetInfo.adapterId, path.targetInfo.id};
        if (device_info(&advanced) != ERROR_SUCCESS) {
            return info;
        }
        SdrWhiteLevel white{};
        white.header = {get_sdr_white_level, sizeof white, path.targetInfo.adapterId, path.targetInfo.id};
        if (device_info(&white) == ERROR_SUCCESS) {
            info.sdr_white_nits = white.level / 1000.0 * 80.0;
        }
        info.hdr_supported = (advanced.value & 0x1) != 0;
        info.hdr_enabled = (advanced.value & 0x2) != 0;
        info.bits_per_color_channel = static_cast<int>(advanced.bits_per_color_channel);
        return info;
    }
    return info;
}

void raise_thread_priority() { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); }

bool keep_system_awake(bool awake) {
    return SetThreadExecutionState(ES_CONTINUOUS | (awake ? ES_SYSTEM_REQUIRED : 0)) != 0;
}

namespace {

// The PCI vendor IDs of the maker VML_GPU_VENDOR names, or nothing when it
// is not set (every maker's GPUs are listed).
std::optional<std::vector<unsigned>> named_vendor_ids() {
    std::optional<std::string> maker = environment_variable("VML_GPU_VENDOR");
    if (!maker || maker->empty()) {
        return std::nullopt;
    }
    std::transform(maker->begin(), maker->end(), maker->begin(),
                   [](unsigned char letter) { return static_cast<char>(std::tolower(letter)); });
    if (*maker == "nvidia") {
        return std::vector<unsigned>{nvidia_pci_vendor};
    }
    if (*maker == "intel") {
        return std::vector<unsigned>{intel_pci_vendor};
    }
    if (*maker == "amd") {
        return std::vector<unsigned>(std::begin(amd_pci_vendors), std::end(amd_pci_vendors));
    }
    return std::vector<unsigned>{};  // a maker not known here: none of the GPUs
}

// The hardware GPUs DirectX lists, as their adapter number and PCI vendor
// ID, in its order; only the maker's that VML_GPU_VENDOR names, if set.
std::vector<std::pair<UINT, unsigned>> listed_gpus() {
    std::vector<std::pair<UINT, unsigned>> gpus;
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        return gpus;
    }
    const std::optional<std::vector<unsigned>> named = named_vendor_ids();
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description)) && !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!named || std::find(named->begin(), named->end(), description.VendorId) != named->end())) {
            gpus.emplace_back(index, description.VendorId);
        }
        adapter->Release();
    }
    factory->Release();
    return gpus;
}

}  // namespace

std::vector<unsigned> gpu_vendor_ids() {
    std::vector<unsigned> vendors;
    for (const auto& [index, vendor] : listed_gpus()) {
        vendors.push_back(vendor);
    }
    return vendors;
}

std::optional<unsigned> first_gpu_adapter() {
    if (!named_vendor_ids()) {
        return 0;
    }
    const std::vector<std::pair<UINT, unsigned>> gpus = listed_gpus();
    return gpus.empty() ? std::nullopt : std::optional<unsigned>(gpus.front().first);
}

namespace {

// What every process holds of a GPU's own memory, by Windows' "GPU Adapter
// Memory" counter for the adapter `luid` (as Task Manager shows it).
std::optional<unsigned long long> gpu_memory_used(const LUID& luid) {
    wchar_t path[96];
    swprintf(path, std::size(path), L"\\GPU Adapter Memory(luid_0x%08lx_0x%08lx_phys_0)\\Dedicated Usage",
             static_cast<unsigned long>(luid.HighPart), luid.LowPart);
    PDH_HQUERY query = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) {
        return std::nullopt;
    }
    std::optional<unsigned long long> used;
    PDH_HCOUNTER counter = nullptr;
    PDH_FMT_COUNTERVALUE value{};
    if (PdhAddEnglishCounterW(query, path, 0, &counter) == ERROR_SUCCESS && PdhCollectQueryData(query) == ERROR_SUCCESS &&
        PdhGetFormattedCounterValue(counter, PDH_FMT_LARGE, nullptr, &value) == ERROR_SUCCESS && value.largeValue >= 0) {
        used = static_cast<unsigned long long>(value.largeValue);
    }
    PdhCloseQuery(query);
    return used;
}

}  // namespace

std::optional<unsigned long long> gpu_memory_available(std::string_view name) {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        return std::nullopt;
    }
    std::optional<unsigned long long> available;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; !available && factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 description{};
        IDXGIAdapter3* queried = nullptr;
        // A GPU with little memory of its own shares the PC's (integrated):
        // its memory says nothing of how much a pass can take.
        constexpr SIZE_T own_memory_least = 1ULL << 30;
        if (SUCCEEDED(adapter->GetDesc1(&description)) && !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            description.DedicatedVideoMemory >= own_memory_least && narrow(description.Description) == name &&
            SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&queried)))) {
            // Windows' budget for this process alone is near the whole GPU's
            // memory, as it would push other programs' memory out to make
            // room; what is free is the GPU's memory less what every process
            // holds (4.3 GB of 32 GB on an idle desktop here).
            DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
            const std::optional<unsigned long long> used = gpu_memory_used(description.AdapterLuid);
            if (used && SUCCEEDED(queried->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory))) {
                const unsigned long long total = description.DedicatedVideoMemory;
                const unsigned long long budget = memory.Budget > memory.CurrentUsage ? memory.Budget - memory.CurrentUsage : 0;
                available = std::min(budget, total > *used ? total - *used : 0);
            }
            queried->Release();
        }
        adapter->Release();
    }
    factory->Release();
    return available;
}

std::optional<long long> modified_ns(const std::filesystem::path& file) {
    WIN32_FILE_ATTRIBUTE_DATA attributes{};
    if (!GetFileAttributesExW(file.c_str(), GetFileExInfoStandard, &attributes)) {
        return std::nullopt;
    }
    // A FILETIME counts 100 ns since 1601; 116444736000000000 of them come
    // before 1970.
    const long long since_1601 = (static_cast<long long>(attributes.ftLastWriteTime.dwHighDateTime) << 32) |
                                 attributes.ftLastWriteTime.dwLowDateTime;
    return (since_1601 - 116444736000000000LL) * 100;
}

namespace {

// Whether a crash ends this process at once (crash_quietly), once it is
// reported (report_crashes_to).
bool quiet_crashes = false;

}  // namespace

void crash_quietly() {
    quiet_crashes = true;
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);  // abort() without its message box
#ifdef _DEBUG
    // A debug build's failed checks go to standard error, not to a dialog.
    for (const int report : {_CRT_WARN, _CRT_ERROR, _CRT_ASSERT}) {
        _CrtSetReportMode(report, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(report, _CRTDBG_FILE_STDERR);
    }
#endif
#endif
}

namespace {

// The file crashes are reported in (report_crashes_to).
HANDLE crash_file = INVALID_HANDLE_VALUE;

void write_crash_text(const char* text) {
    DWORD written = 0;
    WriteFile(crash_file, text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
}

// An address as the module it is in and its offset there
// ("vmaf_vulkan.dll+0x1a2b3"), or as itself.
void describe_address(const void* address, char* out, std::size_t size) {
    HMODULE module = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module) &&
        module != nullptr) {
        GetModuleFileNameA(module, name, sizeof name);
        const char* base = std::strrchr(name, '\\');
        std::snprintf(out, size, "%s+0x%llx", base != nullptr ? base + 1 : name,
                      static_cast<unsigned long long>(static_cast<const char*>(address) -
                                                      reinterpret_cast<const char*>(module)));
    } else {
        std::snprintf(out, size, "0x%p", address);
    }
}

// Writes `headline`, the time and the thread, then the calling stack: from
// `from` on when it is on the stack (an exception's address, past this
// handler and Windows' own frames that called it), or all of it.
void report_crash(const char* headline, const void* from = nullptr) {
    SYSTEMTIME now{};
    GetLocalTime(&now);
    char line[512];
    std::snprintf(line, sizeof line, "%04u-%02u-%02u %02u:%02u:%02u %s in thread %lu\nStack:\n",
                  static_cast<unsigned>(now.wYear), static_cast<unsigned>(now.wMonth), static_cast<unsigned>(now.wDay),
                  static_cast<unsigned>(now.wHour), static_cast<unsigned>(now.wMinute),
                  static_cast<unsigned>(now.wSecond), headline, GetCurrentThreadId());
    write_crash_text(line);
    void* frames[62];
    const USHORT count = CaptureStackBackTrace(0, 62, frames, nullptr);
    USHORT first = 0;
    while (from != nullptr && first < count && frames[first] != from) {
        ++first;
    }
    if (first == count) {
        first = 0;
    }
    for (USHORT i = first; i < count; ++i) {
        char where[300];
        describe_address(frames[i], where, sizeof where);
        std::snprintf(line, sizeof line, "  %s\n", where);
        write_crash_text(line);
    }
    write_crash_text("\n");
    FlushFileBuffers(crash_file);
}

LONG WINAPI report_exception(EXCEPTION_POINTERS* exception) {
    char where[300];
    describe_address(exception->ExceptionRecord->ExceptionAddress, where, sizeof where);
    char headline[400];
    std::snprintf(headline, sizeof headline, "Fatal exception 0x%08lX at %s",
                  static_cast<unsigned long>(exception->ExceptionRecord->ExceptionCode), where);
    report_crash(headline, exception->ExceptionRecord->ExceptionAddress);
    if (quiet_crashes) {
        // Ended now, with the exception's code as its exit code, as Windows
        // would end it, without Windows' error reporting first.
        TerminateProcess(GetCurrentProcess(), exception->ExceptionRecord->ExceptionCode);
    }
    return EXCEPTION_CONTINUE_SEARCH;  // the process then ends as it would have
}

void report_abort(int /*signal*/) { report_crash("Aborted"); }

}  // namespace

bool report_crashes_to(const std::filesystem::path& file) {
    crash_file = CreateFileW(file.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (crash_file == INVALID_HANDLE_VALUE) {
        return false;
    }
    SetUnhandledExceptionFilter(report_exception);
    std::signal(SIGABRT, report_abort);
    return true;
}

void end_process_now(int exit_code) {
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(exit_code));
    std::_Exit(exit_code);  // not reached
}

int process_id() { return static_cast<int>(GetCurrentProcessId()); }

bool process_running(int pid) {
    if (pid <= 0) {
        return pid == 0;  // the System Idle Process
    }
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
    if (process == nullptr) {
        return GetLastError() == ERROR_ACCESS_DENIED;  // it runs, but is not ours to look at
    }
    // A process that has ended can still be opened while a handle to it is
    // open somewhere.
    const bool running = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
    CloseHandle(process);
    return running;
}

void close_native_handle(void* handle) {
    if (handle != nullptr) {
        CloseHandle(handle);
    }
}

std::filesystem::path executable_path() {
    std::wstring path(MAX_PATH, L'\0');
    while (true) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (length < path.size()) {
            path.resize(length);
            return std::filesystem::path(path);
        }
        path.resize(path.size() * 2);  // it was cut short
    }
}

std::string path_text(const std::filesystem::path& path) { return narrow(path.native()); }

std::filesystem::path path_from_text(std::string_view utf8) { return std::filesystem::path(widen(utf8)); }

std::optional<std::string> fopen_path(const std::filesystem::path& path) {
    // The ANSI code page's bytes for the name, when it has every letter.
    const auto in_code_page = [](const std::wstring& name) -> std::optional<std::string> {
        if (name.empty()) {
            return std::string();
        }
        BOOL lost = FALSE;
        const int bytes = WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, name.data(), static_cast<int>(name.size()),
                                              nullptr, 0, nullptr, &lost);
        if (bytes <= 0 || lost) {
            return std::nullopt;
        }
        std::string text(static_cast<std::size_t>(bytes), '\0');
        WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, name.data(), static_cast<int>(name.size()), text.data(),
                            bytes, nullptr, nullptr);
        return text;
    };
    if (std::optional<std::string> text = in_code_page(path.native())) {
        return text;
    }
    std::wstring short_name(32768, L'\0');
    const DWORD length = GetShortPathNameW(path.c_str(), short_name.data(), static_cast<DWORD>(short_name.size()));
    if (length == 0 || length >= short_name.size()) {
        return std::nullopt;
    }
    short_name.resize(length);
    return in_code_page(short_name);
}

}  // namespace vml::platform
