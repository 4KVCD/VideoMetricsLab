#include "vml/vship.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <system_error>
#include <tuple>

#include "vml/app_log.h"
#include "vml/colour.h"
#include "vml/geometry.h"
#include "vml/hardware_decoding.h"
#include "vml/isolated.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// Whether the app bundles Vship's builds for this system: Windows', and
// macOS's Vulkan build on MoltenVK (scripts/build_vship_macos.sh).
#if defined(_WIN32) || defined(__APPLE__)
constexpr bool vship_bundled = true;
#else
constexpr bool vship_bundled = false;
#endif

// The build the metrics run on, in tools/vship/vulkan: Vship-fast's Vulkan
// build (scripts/build_vship_vulkan.ps1), whose metrics compute as libjxl's
// and pycvvdp's do, and the fastest: on an RTX 5090, over 600 pairs of 4K
// 10-bit frames, it scored SSIMULACRA2, Butteraugli and CVVDP together at
// 45 pairs a second, where Vship 5.1.2 scored 36 with CUDA and 35 with
// Vulkan. In tools/vship/vship51, stock Vship 5.1.2's Vulkan build (v2.1's),
// which a metric Vship-fast failed is calculated with again.
constexpr const char* vship_backend = "vulkan";
constexpr const char* vship51_folder = "vship51";

// The largest frame, in pixels, a (GPU maker, metric) combination is given,
// with the GPUs it covers for the reason shown. Vship 5.1.2's Vulkan
// CVVDP hung Intel's GPU above 1080p (a Core Ultra 9 285K: Windows reset
// the GPU after 3 to 6 4K frames; 1920x1080 scored all 240). Vship-fast's
// is kept to the same limit: it has not been tried there, as a hang resets
// the GPU.
struct PixelLimit {
    GpuVendor vendor;
    const char* metric;
    long long pixels;
    const char* gpus;
};
const std::vector<PixelLimit> gpu_pixel_limits{
    {GpuVendor::intel, "cvvdp", 1920LL * 1080, "Intel GPUs above 1920x1080"},
};

// Vship_Sample_t for each integer bit depth. Vship takes any of these for
// any chroma subsampling; the layouts are FFmpeg's.
int sample_of(int depth) {
    switch (depth) {
    case 8:
        return 2;
    case 9:
        return 3;
    case 10:
        return 5;
    case 12:
        return 7;
    case 14:
        return 9;
    case 16:
        return 11;
    default:
        throw std::invalid_argument("no Vship sample for " + std::to_string(depth) + " bits");
    }
}

// Subsampling shifts (width, height) of FFmpeg's planar YUV layouts. 4:1:0
// (yuv410p) has one chroma sample per 4x4 block, so (2, 2): not (2, 1) as
// FFVship 5.1.1 has it (Vship issue 21), which reads chroma planes twice
// the height FFmpeg wrote.
std::pair<int, int> subsampling_of(const std::string& sampling) {
    static const std::map<std::string, std::pair<int, int>, std::less<>> shifts{
        {"410", {2, 2}}, {"411", {2, 0}}, {"420", {1, 1}}, {"422", {1, 0}}, {"440", {0, 1}}, {"444", {0, 0}}};
    return shifts.at(sampling);
}

// The planar YUV format for a subsampling ("420") and bit depth.
VshipImageFormat format_yuv(const std::string& sampling, int depth) {
    const auto [subw, subh] = subsampling_of(sampling);
    VshipImageFormat image;
    image.pixel_format = "yuv" + sampling + "p" + (depth == 8 ? "" : std::to_string(depth) + "le");
    image.family = 0;
    image.sample = sample_of(depth);
    image.subw = subw;
    image.subh = subh;
    return image;
}

// ----------------------------------------------------- the Vulkan loader

// Vulkan's structures for starting an instance, as vulkan_core.h lays them
// out, to ask the loader before Vship's Vulkan build is loaded.
struct VkApplicationInfo {
    int sType = 0;  // VK_STRUCTURE_TYPE_APPLICATION_INFO
    const void* pNext = nullptr;
    const char* pApplicationName = "VideoMetricsLab";
    std::uint32_t applicationVersion = 1;
    const char* pEngineName = nullptr;
    std::uint32_t engineVersion = 0;
    std::uint32_t apiVersion = 1u << 22;  // Vulkan 1.0
};

struct VkInstanceCreateInfo {
    int sType = 1;  // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    const void* pNext = nullptr;
    std::uint32_t flags = 0;
    const VkApplicationInfo* pApplicationInfo = nullptr;
    std::uint32_t enabledLayerCount = 0;
    const char* const* ppEnabledLayerNames = nullptr;
    std::uint32_t enabledExtensionCount = 0;
    const char* const* ppEnabledExtensionNames = nullptr;
};

// The Vulkan loader, with an instance started, or why not.
class VulkanLoader {
public:
    VulkanLoader() {
#ifdef __APPLE__
        // macOS has no Vulkan of its own: the app bundles Khronos' loader with
        // MoltenVK, Vulkan on Metal, as its driver, which Vship is linked to,
        // and loads the loader by its path.
        const std::filesystem::path vulkan = vulkan_library_path();
        try {
            library_.emplace(vulkan);
        } catch (const platform::LibraryError&) {
            problem_ = "Vulkan (MoltenVK) is not bundled (" + platform::path_text(vulkan.filename()) +
                       " was not found)";
            return;
        }
#else
#ifdef _WIN32
        const std::string loader = "vulkan-1.dll";
#else
        const std::string loader = "libvulkan.so.1";
#endif
        try {
            library_.emplace(platform::DynamicLibrary::system(loader));
        } catch (const platform::LibraryError&) {
            problem_ = "no Vulkan driver is installed (" + loader + " was not found)";
            return;
        }
#endif
        create_ = library_->function<int (*)(const VkInstanceCreateInfo*, const void*, void**)>("vkCreateInstance");
        enumerate_ =
            library_->function<int (*)(void*, std::uint32_t*, void**)>("vkEnumeratePhysicalDevices");
        properties_ = library_->function<void (*)(void*, void*)>("vkGetPhysicalDeviceProperties");
        destroy_ = library_->function<void (*)(void*, const void*)>("vkDestroyInstance");
        const VkApplicationInfo application;
        VkInstanceCreateInfo info;
        info.pApplicationInfo = &application;
#ifdef __APPLE__
        // MoltenVK is a portability driver: the loader shows its GPUs only to
        // an instance that asks for them, as Vship's does.
        static const char* const portability = "VK_KHR_portability_enumeration";
        info.flags = 0x00000001;  // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = &portability;
#endif
        const int result = create_(&info, nullptr, &instance_);
        if (result != 0) {
            instance_ = nullptr;
            problem_ = "no Vulkan driver answered (VkResult " + std::to_string(result) + ")";
        }
    }

    ~VulkanLoader() {
        if (instance_ != nullptr) {
            destroy_(instance_, nullptr);
        }
    }

    VulkanLoader(const VulkanLoader&) = delete;
    VulkanLoader& operator=(const VulkanLoader&) = delete;

    // Why no GPU is usable through Vulkan here, or nothing if one is.
    std::optional<std::string> unavailable() {
        if (problem_) {
            return problem_;
        }
        if (devices().empty()) {
            return "no GPU with a Vulkan driver was found";
        }
        return std::nullopt;
    }

    // Who made the GPU called `name` (Vship names a GPU by its
    // VkPhysicalDeviceProperties.deviceName): GpuVendor::none for another
    // maker, nothing if no GPU has the name.
    std::optional<GpuVendor> vendor_of(const std::string& name) {
        if (problem_) {
            return std::nullopt;
        }
        for (void* device : devices()) {
            // VkPhysicalDeviceProperties: apiVersion, driverVersion, vendorID,
            // deviceID, deviceType (4 bytes each), then deviceName[256].
            std::array<unsigned char, 1024> properties{};
            properties_(device, properties.data());
            const char* device_name = reinterpret_cast<const char*>(properties.data() + 20);
            if (replace_invalid_utf8(std::string(device_name, strnlen(device_name, 256))) == name) {
                const std::uint32_t vendor = static_cast<std::uint32_t>(properties[8]) |
                                             (static_cast<std::uint32_t>(properties[9]) << 8) |
                                             (static_cast<std::uint32_t>(properties[10]) << 16) |
                                             (static_cast<std::uint32_t>(properties[11]) << 24);
                return vendor_of_pci_id(vendor);
            }
        }
        return std::nullopt;
    }

private:
    std::vector<void*> devices() {
        std::uint32_t count = 0;
        if (enumerate_(instance_, &count, nullptr) != 0 || count == 0) {
            return {};
        }
        std::vector<void*> found(count);
        enumerate_(instance_, &count, found.data());
        found.resize(count);
        return found;
    }

    std::optional<platform::DynamicLibrary> library_;
    std::optional<std::string> problem_;
    int (*create_)(const VkInstanceCreateInfo*, const void*, void**) = nullptr;
    int (*enumerate_)(void*, std::uint32_t*, void**) = nullptr;
    void (*properties_)(void*, void*) = nullptr;
    void (*destroy_)(void*, const void*) = nullptr;
    void* instance_ = nullptr;
};

// ----------------------------------------------------- the loaded library

// The library once this process has loaded it, kept until it ends: a
// library that started a GPU driver is never unloaded, as Python's ctypes
// never unloads one.
struct LoadedLibrary {
    std::mutex mutex;
    std::unique_ptr<VshipLibrary> library;
    bool vship51 = false;  // stock Vship 5.1.2's rather than Vship-fast's
};

LoadedLibrary& loaded_library() {
    static LoadedLibrary loaded;
    return loaded;
}

// The API's functions, all of which a library must have.
constexpr std::array<const char*, 13> api_functions{
    "Vship_GetVersion",         "Vship_GetDeviceCount",         "Vship_GetDeviceInfo",
    "Vship_GPUFullCheck",       "Vship_GetErrorMessage",        "Vship_GetDetailedLastError",
    "Vship_PinnedMalloc2",      "Vship_PinnedFree2",            "Vship_InitHandler",
    "Vship_FreeHandler",        "Vship_ComputeHandler",         "Vship_GetDetailedLastErrorHandler",
    "Vship_ResetScore",
};

// A build without Vship 5.1's API, refused by the probe.
class OldVshipError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

bool has_api(const platform::DynamicLibrary& library) {
    return std::all_of(api_functions.begin(), api_functions.end(),
                       [&library](const char* name) { return library.find(name) != nullptr; });
}

// The library this process has loaded, or else `load`ed and kept: of the
// build asked for (stock Vship 5.1.2's with `vship51`), as a process keeps
// only one.
const VshipLibrary& kept_library(const std::function<platform::DynamicLibrary()>& load, bool vship51 = false) {
    LoadedLibrary& loaded = loaded_library();
    const std::lock_guard lock(loaded.mutex);
    if (!loaded.library) {
        loaded.library = std::make_unique<VshipLibrary>(load());
        loaded.vship51 = vship51;
    } else if (loaded.vship51 != vship51) {
        throw VshipUnavailableError(std::string("this process has loaded ") +
                                    (loaded.vship51 ? "Vship 5.1.2" : "Vship-fast") + " already");
    }
    return *loaded.library;
}

// ----------------------------------------------------- the probe's state

// One Vship build's probe (detect_vship_device).
struct ProbeState {
    std::mutex mutex;  // held for the whole of a probe
    std::optional<std::pair<std::optional<VshipDevice>, std::string>> probed;
    bool failed = false;  // no GPU, as a check crashed or failed (vship_check_failed)
    std::chrono::steady_clock::time_point probed_at;
    // The last finished probe's result, under a lock of its own that a probe
    // in progress does not hold (known_vship_device).
    std::mutex known_mutex;
    std::optional<std::pair<std::optional<VshipDevice>, std::string>> known;
};

// Who is told as either build's probe finishes (on_vship_probe_finished).
struct ProbeListener {
    std::mutex mutex;
    std::function<void()> on_finished;
};

// How long a probe that found no GPU is kept before a run probes again
// (forget_failed_vship_probe).
constexpr std::chrono::seconds failed_probe_retry{60};

ProbeState& probe_state(VshipBuild build) {
    static ProbeState fast;
    static ProbeState stock;
    return build == VshipBuild::stock ? stock : fast;
}

ProbeListener& probe_listener() {
    static ProbeListener listener;
    return listener;
}

std::string vendor_text(const std::optional<GpuVendor>& vendor) {
    if (!vendor) {
        return "";
    }
    switch (*vendor) {
    case GpuVendor::nvidia:
        return "nvidia";
    case GpuVendor::intel:
        return "intel";
    case GpuVendor::amd:
        return "amd";
    default:
        return "none";
    }
}

std::optional<GpuVendor> vendor_from(const std::string& text) {
    if (text == "nvidia") {
        return GpuVendor::nvidia;
    }
    if (text == "intel") {
        return GpuVendor::intel;
    }
    if (text == "amd") {
        return GpuVendor::amd;
    }
    if (text == "none") {
        return GpuVendor::none;
    }
    return std::nullopt;
}

// The probe, in a process of its own (a crash there would end the run:
// Vship 5.1.1's Vulkan build crashed as it closed, on a PC whose only GPU
// is Intel's). Any failure is an answer, no GPU, as VMAF's and VMAF v1's
// probes take one: one that escaped would leave the probe unasked, and a
// run that waits for it (MainWindow::wait_for_gpu_checks) asking again for
// good. With `vship51`, stock Vship 5.1.2's. `failed` is set when the probe
// crashed or failed rather than answering. A probe stopped as the app ends
// (stop_gpu_probes) is no answer: Cancelled goes on.
std::pair<std::optional<VshipDevice>, std::string> probe_isolated(bool vship51, bool& failed) {
    try {
        const nlohmann::json result =
            run_isolated("vship-probe", vship51 ? "Vship 5.1.2's GPU probe" : "Vship's GPU probe",
                         {{"vship51", vship51}}, {}, gpu_probes_token());
        return {vship_device_from(result.at("device")), result.at("reason").get<std::string>()};
    } catch (const IsolatedCrashError& error) {
        failed = true;
        return {std::nullopt, error.what()};
    } catch (const Cancelled&) {
        throw;
    } catch (const std::exception& error) {
        failed = true;
        return {std::nullopt, std::string("the GPU probe failed: ") + error.what()};
    }
}

}  // namespace

// ------------------------------------------------------------------ library

VshipLibrary::VshipLibrary(platform::DynamicLibrary loaded) : library(std::move(loaded)) {
    if (!has_api(library)) {
        throw VshipUnavailableError(platform::path_text(library.path()) + " is older than Vship 5.1");
    }
    get_version = library.function<decltype(get_version)>("Vship_GetVersion");
    get_device_count = library.function<decltype(get_device_count)>("Vship_GetDeviceCount");
    get_device_info = library.function<decltype(get_device_info)>("Vship_GetDeviceInfo");
    gpu_full_check = library.function<decltype(gpu_full_check)>("Vship_GPUFullCheck");
    get_error_message = library.function<decltype(get_error_message)>("Vship_GetErrorMessage");
    get_detailed_last_error = library.function<decltype(get_detailed_last_error)>("Vship_GetDetailedLastError");
    pinned_malloc = library.function<decltype(pinned_malloc)>("Vship_PinnedMalloc2");
    pinned_free = library.function<decltype(pinned_free)>("Vship_PinnedFree2");
    init_handler = library.function<decltype(init_handler)>("Vship_InitHandler");
    free_handler = library.function<decltype(free_handler)>("Vship_FreeHandler");
    compute_handler = library.function<decltype(compute_handler)>("Vship_ComputeHandler");
    get_detailed_last_error_handler =
        library.function<decltype(get_detailed_last_error_handler)>("Vship_GetDetailedLastErrorHandler");
    reset_score = library.function<decltype(reset_score)>("Vship_ResetScore");
}

std::string VshipLibrary::message(std::optional<int> code) const {
    std::array<char, 1024> buffer{};
    if (code) {
        get_error_message(*code, buffer.data(), static_cast<int>(buffer.size()));
    } else {
        get_detailed_last_error(buffer.data(), static_cast<int>(buffer.size()));
    }
    const std::string text(python_strip(replace_invalid_utf8(std::string(buffer.data(), strnlen(buffer.data(), buffer.size())))));
    return !text.empty() ? text : "Vship error " + (code ? std::to_string(*code) : std::string("None"));
}

std::string VshipLibrary::handler_error(void* handler, int code) const {
    std::array<char, 1024> detail{};
    get_detailed_last_error_handler(handler, detail.data(), static_cast<int>(detail.size()));
    if (detail[0] != '\0') {
        return std::string(
            python_strip(replace_invalid_utf8(std::string(detail.data(), strnlen(detail.data(), detail.size())))));
    }
    return message(code);
}

namespace {

// The bundled Vulkan's folder (macOS), its loader pointed at it once, before
// Vulkan is first loaded in this process: by the app's probe, or by Vship's
// library, which loads it itself (in a GPU process of its own, too). The
// loader's driver is MoltenVK, from its manifest beside it, which says only
// its errors on standard error. Its fast math stays on its default, on
// demand: Vship-fast turns it off in the kernels that need Metal's safe math
// and keeps it in CVVDP's and its YUV-to-RGB step (an M1 Pro: CVVDP 1.24x
// faster than with it off everywhere, scores within 2e-6; libvmaf-fast's
// Vulkan VMAF the same to the byte either way). A value the user set stays,
// the loader's older name for its drivers (VK_ICD_FILENAMES) too, which
// VK_DRIVER_FILES would otherwise override.
std::filesystem::path vulkan_folder() {
    const std::filesystem::path folder = resource_folder() / "tools" / "vulkan";
#ifdef __APPLE__
    static const bool prepared = [&folder] {
        if (std::getenv("VK_ICD_FILENAMES") == nullptr) {
            setenv("VK_DRIVER_FILES", platform::path_text(folder / "MoltenVK_icd.json").c_str(), 0);
        }
        setenv("MVK_CONFIG_LOG_LEVEL", "1", 0);
        return true;
    }();
    static_cast<void>(prepared);
#endif
    return folder;
}

}  // namespace

std::filesystem::path vulkan_library_path() { return vulkan_folder() / "libvulkan.1.dylib"; }

std::filesystem::path vship_library_path(bool vship51) {
#ifdef __APPLE__
    static_cast<void>(vulkan_folder());  // the loader pointed at MoltenVK before Vship loads it
#endif
    return resource_folder() / "tools" / "vship" / (vship51 ? vship51_folder : vship_backend) /
           platform::shared_library_name("libvship");
}

bool vship51_bundled() {
    std::error_code missing;
    return vship_bundled && std::filesystem::is_regular_file(vship_library_path(true), missing);
}

VshipBuild default_vship_build(std::string_view setting) {
    const VshipBuild chosen = vship_build_from(setting);
    return chosen == VshipBuild::stock && !vship51_bundled() ? VshipBuild::fast : chosen;
}

const VshipLibrary& vship_library(bool vship51) {
    try {
        return kept_library([vship51] { return platform::DynamicLibrary(vship_library_path(vship51)); }, vship51);
    } catch (const platform::LibraryError& error) {
        throw VshipUnavailableError(error.what());
    }
}

// ------------------------------------------------------------------ the GPU

std::optional<std::string> size_refusal(const VshipDevice& device, const std::string& key, std::pair<int, int> size,
                                        std::optional<std::pair<int, int>> computed) {
    const auto limit = std::find_if(gpu_pixel_limits.begin(), gpu_pixel_limits.end(), [&](const PixelLimit& entry) {
        return device.vendor && *device.vendor == entry.vendor && key == entry.metric;
    });
    const std::pair<int, int> at = computed.value_or(size);
    if (limit == gpu_pixel_limits.end() || static_cast<long long>(at.first) * at.second <= limit->pixels) {
        return std::nullopt;
    }
    const std::string shown = std::to_string(size.first) + "x" + std::to_string(size.second);
    const std::string detail = at == size ? "this comparison is " + shown
                                          : "this comparison is " + shown + ", scaled to the display: " +
                                                std::to_string(at.first) + "x" + std::to_string(at.second);
    return "on " + std::string(limit->gpus) + " it is not calculated, as Vship's Vulkan build hangs the GPU (" +
           detail + ")";
}

std::string vship_build_name(const VshipDevice& device) {
    return (device.vship51 ? "Vship " : "Vship-fast ") + device.version;
}

std::pair<std::optional<VshipDevice>, std::string> probe_vship_device(bool vship51) {
    // The rest is built on every system all the same, so that it stays
    // ready for the builds of the others.
    if constexpr (!vship_bundled) {
        return {std::nullopt, "Vship GPU acceleration is only bundled for Windows and macOS."};
    }
    const std::filesystem::path path = vship_library_path(vship51);
    std::error_code missing;
    if (!std::filesystem::is_regular_file(path, missing)) {
        return {std::nullopt, vship51 ? "Vship 5.1.2's Vulkan library is missing" : "Vship's Vulkan library is missing"};
    }
    if (const std::optional<std::string> unavailable = VulkanLoader().unavailable()) {
        return {std::nullopt, "Vulkan: " + *unavailable};
    }
    const VshipLibrary* library = nullptr;
    try {
        library = &kept_library(
            [&path] {
                platform::DynamicLibrary loaded(path);
                if (!has_api(loaded)) {
                    throw OldVshipError("Vship's Vulkan library is older than 5.1");
                }
                return loaded;
            },
            vship51);
    } catch (const OldVshipError& error) {
        return {std::nullopt, error.what()};
    } catch (const VshipUnavailableError& error) {  // the other build is loaded in this process
        return {std::nullopt, error.what()};
    } catch (const platform::LibraryError& error) {
        return {std::nullopt, std::string("Vship's Vulkan library could not be loaded: ") + error.what()};
    }
    int count = 0;
    if (const int error = library->get_device_count(&count); error != 0) {
        return {std::nullopt, "Vulkan: " + library->message(error)};
    }
    // (integrated, number, name): a discrete GPU first, then Vship's order.
    std::vector<std::tuple<bool, int, std::string>> usable;
    std::vector<std::string> failures;
    for (int gpu_id = 0; gpu_id < count; ++gpu_id) {
        if (const int error = library->gpu_full_check(gpu_id); error != 0) {
            failures.push_back("Vulkan GPU " + std::to_string(gpu_id) + ": " + library->message(error));
            continue;
        }
        vship_api::DeviceInfo info;
        if (const int error = library->get_device_info(&info, gpu_id); error != 0) {
            failures.push_back("Vulkan GPU " + std::to_string(gpu_id) + ": " + library->message(error));
            continue;
        }
        const std::string name = replace_invalid_utf8(std::string(info.name, strnlen(info.name, sizeof info.name)));
        usable.emplace_back(info.integrated != 0, gpu_id, name.empty() ? "Vulkan GPU " + std::to_string(gpu_id) : name);
    }
    if (usable.empty()) {
        if (count == 0) {
            failures.push_back("Vulkan: no GPU was found");
        }
        std::string reasons;
        for (const std::string& failure : failures) {
            reasons += (reasons.empty() ? "" : "; ") + failure;
        }
        return {std::nullopt, reasons};
    }
    const auto& [integrated, gpu_id, name] = *std::min_element(usable.begin(), usable.end());
    const vship_api::Version version = library->get_version();
    return {VshipDevice{vship_backend, name, gpu_id,
                        std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
                            std::to_string(version.minorMinor),
                        VulkanLoader().vendor_of(name), vship51},
            ""};
}

nlohmann::json vship_device_json(const VshipDevice& device) {
    return {{"backend", device.backend}, {"name", device.name},
            {"gpu_id", device.gpu_id},   {"version", device.version},
            {"vendor", vendor_text(device.vendor)}, {"vship51", device.vship51}};
}

std::optional<VshipDevice> vship_device_from(const nlohmann::json& text) {
    if (text.is_null()) {
        return std::nullopt;
    }
    VshipDevice device;
    text.at("backend").get_to(device.backend);
    text.at("name").get_to(device.name);
    text.at("gpu_id").get_to(device.gpu_id);
    text.at("version").get_to(device.version);
    device.vendor = vendor_from(text.at("vendor").get<std::string>());
    device.vship51 = text.value("vship51", false);
    return device;
}

nlohmann::json vship_probe_result(const std::pair<std::optional<VshipDevice>, std::string>& probed) {
    return {{"device", probed.first ? vship_device_json(*probed.first) : nlohmann::json()}, {"reason", probed.second}};
}

std::pair<std::optional<VshipDevice>, std::string> detect_vship_device(VshipBuild build) {
    // One probe of each build serves the whole process: callers that arrive
    // while it runs wait for it rather than starting their own.
    ProbeState& state = probe_state(build);
    const std::lock_guard lock(state.mutex);
    if (!state.probed) {
        const bool vship51 = build == VshipBuild::stock;
        bool failed = false;
        // Kept only once whole: a probe stopped as the app ends (Cancelled,
        // 5.1.2's below too) leaves this build unasked, not without a GPU.
        std::pair<std::optional<VshipDevice>, std::string> probed =
            isolation_program() ? probe_isolated(vship51, failed) : probe_vship_device(vship51);
        if (!vship51 && !probed.first && isolation_program() && vship51_bundled()) {
            // Vship-fast compiles every shader as it starts on a GPU: one of
            // its own that the driver refuses, or crashes on, leaves it no
            // GPU where stock Vship 5.1.2 may have one. Its reason stays the
            // answer if 5.1.2 finds none either. 5.1.2's own probe answers,
            // so that it runs once for both builds.
            std::pair<std::optional<VshipDevice>, std::string> stock = detect_vship_device(VshipBuild::stock);
            if (stock.first) {
                app_log::write(app_log::Level::warning, "vml.perceptual_vship",
                               "Vship-fast can use no GPU (" + probed.second + "); Vship 5.1.2 can");
                stock.first->fast_failure = probed.second;
                probed = std::move(stock);
            } else {
                failed = failed || vship_check_failed(VshipBuild::stock);
            }
        }
        state.probed = std::move(probed);
        state.failed = failed && !state.probed->first;
        state.probed_at = std::chrono::steady_clock::now();
        const auto& [device, reason] = *state.probed;
        if (device) {
            app_log::write(app_log::Level::info, "vml.perceptual_vship",
                           vship_build_name(*device) + " (Vulkan) GPU: " + device->name);
        } else {
            app_log::write(app_log::Level::warning, "vml.perceptual_vship",
                           std::string(vship51 ? "Vship 5.1.2" : "Vship-fast") + " GPU unavailable: " + reason);
        }
        {
            const std::lock_guard known_lock(state.known_mutex);
            state.known = state.probed;
        }
        ProbeListener& listener = probe_listener();
        const std::lock_guard listener_lock(listener.mutex);
        if (listener.on_finished) {
            listener.on_finished();
        }
    }
    return *state.probed;
}

bool vship_check_failed(VshipBuild build) {
    ProbeState& state = probe_state(build);
    const std::lock_guard lock(state.mutex);
    return state.probed && state.failed;
}

void on_vship_probe_finished(std::function<void()> callback) {
    ProbeListener& listener = probe_listener();
    const std::lock_guard lock(listener.mutex);
    listener.on_finished = std::move(callback);
}

std::optional<std::pair<std::optional<VshipDevice>, std::string>> known_vship_device(VshipBuild build) {
    ProbeState& state = probe_state(build);
    const std::lock_guard lock(state.known_mutex);
    return state.known;
}

bool vship_probe_ready(VshipBuild build) {
    ProbeState& state = probe_state(build);
    const std::unique_lock lock(state.mutex, std::try_to_lock);
    return lock.owns_lock() && state.probed.has_value();
}

void forget_failed_vship_probe() {
    for (const VshipBuild build : {VshipBuild::fast, VshipBuild::stock}) {
        ProbeState& state = probe_state(build);
        const std::lock_guard lock(state.mutex);
        if (state.probed && !state.probed->first &&
            std::chrono::steady_clock::now() - state.probed_at >= failed_probe_retry) {
            state.probed.reset();
        }
    }
}

VshipBuild vship_build_in_use(VshipBuild chosen) {
    if (chosen == VshipBuild::fast) {
        const auto known = known_vship_device(VshipBuild::fast);
        if (known && known->first && known->first->vship51) {
            return VshipBuild::stock;
        }
    }
    return chosen;
}

// ------------------------------------------------------------- frame layouts

VshipImageFormat::Layout VshipImageFormat::frame_layout(int width, int height) const {
    const std::size_t sample_bytes = wide() ? 2 : 1;
    const auto w = static_cast<std::size_t>(width), h = static_cast<std::size_t>(height);
    Layout layout;
    if (family == 1) {
        layout.plane_bytes = {w * h * sample_bytes, w * h * sample_bytes, w * h * sample_bytes};
        const auto stride = static_cast<std::int64_t>(w * sample_bytes);
        layout.strides = {stride, stride, stride};
    } else {
        const std::size_t chroma_width = (w + (std::size_t{1} << subw) - 1) >> subw;
        const std::size_t chroma_height = (h + (std::size_t{1} << subh) - 1) >> subh;
        const std::size_t chroma = chroma_width * chroma_height * sample_bytes;
        layout.plane_bytes = {w * h * sample_bytes, chroma, chroma};
        layout.strides = {static_cast<std::int64_t>(w * sample_bytes),
                          static_cast<std::int64_t>(chroma_width * sample_bytes),
                          static_cast<std::int64_t>(chroma_width * sample_bytes)};
    }
    layout.frame_bytes = layout.plane_bytes[0] + layout.plane_bytes[1] + layout.plane_bytes[2];
    return layout;
}

bool VshipImageFormat::wide() const { return sample != sample_of(8); }

VshipImageFormat vship_image_format(const VideoInfo& info) {
    const std::string name = ascii_lower(python_strip(info.pix_fmt));
    if (name == "nv12" || name == "nv21") {
        return format_yuv("420", 8);
    }
    static const std::map<std::string, std::pair<std::string, int>, std::less<>> semi_planar{
        {"p010le", {"420", 10}}, {"p010be", {"420", 10}}, {"p012le", {"420", 12}}, {"p016le", {"420", 16}},
        {"p016be", {"420", 16}}, {"nv16", {"422", 8}},     {"p210le", {"422", 10}}, {"p212le", {"422", 12}},
        {"p216le", {"422", 16}}, {"nv24", {"444", 8}},     {"p410le", {"444", 10}}, {"p412le", {"444", 12}},
        {"p416le", {"444", 16}},
    };
    if (const auto found = semi_planar.find(name); found != semi_planar.end()) {
        return format_yuv(found->second.first, found->second.second);
    }
    std::smatch match;
    // yuv, yuvj (full range when untagged) and yuva (alpha dropped).
    static const std::regex yuv("yuv(j|a)?(410|411|420|422|440|444)p(?:(9|10|12|14|16)(?:le|be)?)?");
    if (std::regex_match(name, match, yuv)) {
        VshipImageFormat image = format_yuv(match[2].str(), match[3].matched ? std::stoi(match[3].str()) : 8);
        image.full_range = match[1].matched && match[1].str() == "j";
        return image;
    }
    // Monochrome (AV1 can be): the luma as it is, with neutral chroma.
    static const std::regex gray("gray(?:(9|10|12|14|16)(?:le|be)?)?");
    if (std::regex_match(name, match, gray)) {
        return format_yuv("420", match[1].matched ? std::stoi(match[1].str()) : 8);
    }
    int depth = 0;
    static const std::regex planar_rgb("gbra?p(?:(9|10|12|14|16)(?:le|be)?)?");
    static const std::vector<std::string> rgb8{"rgb24", "bgr24", "rgba", "bgra", "argb",
                                               "abgr",  "rgb0",  "bgr0", "0rgb", "0bgr"};
    static const std::vector<std::string> rgb16{"rgb48le",   "rgb48be",   "bgr48le",   "bgr48be",
                                                "rgba64le",  "rgba64be",  "bgra64le",  "bgra64be"};
    if (std::regex_match(name, match, planar_rgb)) {
        depth = match[1].matched ? std::stoi(match[1].str()) : 8;
    } else if (std::find(rgb8.begin(), rgb8.end(), name) != rgb8.end()) {
        depth = 8;
    } else if (std::find(rgb16.begin(), rgb16.end(), name) != rgb16.end()) {
        depth = 16;
    } else {
        throw VshipUnavailableError("The GPU metrics cannot read the decoded pixel format " +
                                    (info.pix_fmt.empty() ? std::string("(unknown)") : info.pix_fmt) + ".");
    }
    VshipImageFormat image;
    image.pixel_format = "gbrp" + (depth > 8 ? std::to_string(depth) + "le" : std::string());
    image.family = 1;
    image.sample = sample_of(depth);
    image.full_range = true;
    image.plane_order = {2, 0, 1};
    return image;
}

std::optional<std::pair<VshipImageFormat, std::string>> passthrough_format(const VideoInfo& info,
                                                                           const std::optional<std::string>& hwaccel) {
    if (!hwaccel || hwaccel->empty()) {
        return std::nullopt;
    }
    const VshipImageFormat image = vship_image_format(info);
    const std::string native = hw_native_format(info.pix_fmt);
    if (image.pixel_format == "yuv420p" && native == "nv12") {
        return std::pair(image, std::string("nv12"));
    }
    // P010 keeps each 10-bit sample in the top bits of 16. Declared to Vship
    // as 16-bit and limited range, that is exactly the same picture: Vship
    // brings limited-range samples to 8-bit scale by dividing by
    // 2^(depth-8), so (v << 6) / 256 and v / 4 are the same float. Full
    // range divides by 2^depth - 1, where the two differ.
    const std::string range = ascii_lower(info.color_range);
    if (image.pixel_format == "yuv420p10le" && native == "p010le" && !image.full_range && range != "pc" &&
        range != "jpeg" && range != "full") {
        VshipImageFormat sixteen = format_yuv("420", 16);
        return std::pair(sixteen, std::string("p010le"));
    }
    return std::nullopt;
}

namespace {

// A chroma siting as Vship's Vship_ChromaLocation_t, which has no bottom or
// bottom-left siting; none given is read as left.
std::optional<int> vship_siting(const std::string& location) {
    std::string name = ascii_lower(location.empty() ? "left" : location);
    std::erase(name, '-');
    static const std::map<std::string, int, std::less<>> locations{
        {"left", 0}, {"center", 1}, {"topleft", 2}, {"top", 3}, {"unspecified", 0}, {"unknown", 0}};
    const auto found = locations.find(name);
    return found == locations.end() ? std::nullopt : std::optional(found->second);
}

bool siting_given(const std::string& location) {
    const std::string name = ascii_lower(location);
    return !name.empty() && name != "unspecified" && name != "unknown";
}

}  // namespace

std::pair<std::string, std::string> shared_chroma_location(const std::string& source, const std::string& distorted) {
    const bool source_given = siting_given(source);
    const bool distorted_given = siting_given(distorted);
    return {source_given || !distorted_given ? source : distorted,
            distorted_given || !source_given ? distorted : source};
}

bool chroma_location_shared(const std::string& source, const std::string& distorted) {
    const auto [source_read, distorted_read] = shared_chroma_location(source, distorted);
    return vship_siting(source_read) != vship_siting(source) || vship_siting(distorted_read) != vship_siting(distorted);
}

vship_api::Colorspace vship_colorspace(const VideoInfo& info, const VshipImageFormat& image, int width, int height) {
    VideoColour colour;
    try {
        colour = video_colour(info, image.family == 1, image.full_range);
    } catch (const UnsupportedColourError& error) {
        const std::string& kind = error.kind();
        throw VshipUnavailableError(
            kind == "matrix"      ? "Vship does not support the " + error.value() + " color matrix."
            : kind == "transfer"  ? "Vship does not support the " + error.value() + " transfer function."
            : kind == "primaries" ? "Vship does not support the " + error.value() + " color primaries."
                                  : "Vship does not support the " + error.value() + " range tag.");
    }
    const std::optional<int> siting = vship_siting(info.chroma_location);
    if (!siting) {
        throw VshipUnavailableError("Vship does not support " + info.chroma_location + " chroma siting.");
    }
    vship_api::Colorspace space;
    space.width = width;
    space.height = height;
    space.target_width = -1;
    space.target_height = -1;
    space.sample = image.sample;
    space.range = colour.full_range ? 1 : 0;
    space.subsampling = {image.subw, image.subh};
    space.chromaLocation = *siting;
    space.colorFamily = image.family;
    space.YUVMatrix = colour.matrix;
    space.transferFunction = colour.transfer;
    space.primaries = colour.primaries;
    return space;
}

std::pair<std::pair<int, int>, std::pair<int, int>> vship_scaled_sizes(const VideoInfo& source,
                                                                       const VideoInfo& distorted,
                                                                       ScaleDirection direction,
                                                                       const std::optional<CropBox>& source_crop,
                                                                       const std::optional<CropBox>& distorted_crop) {
    const Size source_size = content_size(source, source_crop);
    const Size distorted_size = content_size(distorted, distorted_crop);
    if (source_size == distorted_size) {
        return {source_size, distorted_size};
    }
    if (direction == ScaleDirection::distorted_to_source) {
        return {source_size, source_size};
    }
    return {distorted_size, distorted_size};
}

std::string vship_filter_chain(const VideoInfo& info, const std::optional<CropBox>& crop,
                               std::pair<int, int> target_size, const std::string& pixel_format, int step,
                               const std::string& algorithm, const std::optional<std::string>& hwaccel,
                               const std::optional<Rational>& numbered_at, bool full_range) {
    std::vector<std::string> operations;
    if (numbered_at) {
        operations = numbering_filters(*numbered_at);
    }
    if (step > 1) {
        // select comes before the rest, so that the frames it drops are not
        // downloaded, cropped or scaled. Nothing after it drops a frame, so n
        // counts the same frames wherever it stands.
        operations.push_back("select=not(mod(n\\," + std::to_string(step) + "))");
    }
    if (hwaccel && !hwaccel->empty()) {
        // A hardware-decoded surface is brought to system memory in its
        // native layout first; crop, scale and the final conversion then run
        // as they do for a software-decoded input, so the pictures Vship
        // scores do not depend on which decoder made them.
        operations.push_back("hwdownload,format=" + hw_native_format(info.pix_fmt));
    }
    if (crop && !crop->is_noop(info.width, info.height)) {
        operations.push_back(crop->as_filter());
    }
    if (content_size(info, crop) != target_size) {
        operations.push_back("scale=" + std::to_string(target_size.first) + ":" + std::to_string(target_size.second) +
                             ":flags=" + algorithm);
    }
    if (!numbered_at) {
        operations.push_back("setpts=PTS-STARTPTS");
    }
    if (full_range) {
        // FFmpeg brings full-range gray to the limited range as it gives it
        // chroma for Vship's 4:2:0 (and as scale resizes it), though Vship is
        // told it is full range: it read 30-219 where the video held 16-236.
        // Asked for the full range, the conversion keeps it; full-range 4:2:0,
        // NV12 and 4:2:0 with alpha kept theirs either way, and are untouched.
        operations.push_back("scale=out_range=pc");
    }
    operations.push_back("format=" + pixel_format);
    std::string chain;
    for (const std::string& operation : operations) {
        chain += (chain.empty() ? "" : ",") + operation;
    }
    return chain;
}

std::vector<std::string> vship_piped_as(const std::string& pixel_format) {
    // The planar layouts FFmpeg's YUV4MPEG muxer writes (with -strict -1
    // above 8 bits). YUV4MPEG holds the same planes as raw video, each frame
    // after a "FRAME" line, and FFmpeg writes it straight from its frames,
    // while the rawvideo encoder first copies every frame into a packet: on
    // a 4K 10-bit HEVC video, 23 ms of FFmpeg's CPU per frame against 35.
    static const std::regex y4m("yuv(411|420|422|444)p(?:(9|10|12|14|16)le)?");
    if (std::regex_match(pixel_format, y4m)) {
        return {"-pix_fmt", pixel_format, "-f", "yuv4mpegpipe", "-strict", "-1", "pipe:1"};
    }
    return {"-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"};
}

}  // namespace vml
