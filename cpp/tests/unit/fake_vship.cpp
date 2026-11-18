// A stand-in for Vship's library (Vship 5.1's API), for testing how the app
// probes its GPUs without a GPU: the GPUs it lists and which of them fail
// their check come from the environment.
//
//     VML_FAKE_VSHIP_GPUS   name|integrated;name|integrated...  (integrated: 1 or 0)
//     VML_FAKE_VSHIP_FAILS  the numbers of the GPUs whose check fails, as "2;5"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define FAKE_EXPORT extern "C" __declspec(dllexport)
#else
#define FAKE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct Version {
    int major, minor, minorMinor, backend;
};

struct DeviceInfo {
    char name[256];
    std::uint64_t VRAMSize;
    int integrated;
    int MultiProcessorCount;
    int WarpSize;
    bool vulkanFeatureMatrix[26];
};

struct Gpu {
    std::string name;
    bool integrated;
};

std::vector<Gpu> gpus() {
    std::vector<Gpu> listed;
    const char* text = std::getenv("VML_FAKE_VSHIP_GPUS");
    std::string rest = text != nullptr ? text : "";
    while (!rest.empty()) {
        const std::size_t end = rest.find(';');
        const std::string item = rest.substr(0, end);
        const std::size_t bar = item.find('|');
        listed.push_back({item.substr(0, bar), bar != std::string::npos && item.substr(bar + 1) == "1"});
        rest = end == std::string::npos ? "" : rest.substr(end + 1);
    }
    return listed;
}

bool fails(int gpu_id) {
    const char* text = std::getenv("VML_FAKE_VSHIP_FAILS");
    const std::string list = std::string(";") + (text != nullptr ? text : "") + ";";
    return list.find(";" + std::to_string(gpu_id) + ";") != std::string::npos;
}

void copy_text(const char* text, char* out, int bytes) {
    if (out != nullptr && bytes > 0) {
        std::strncpy(out, text, static_cast<std::size_t>(bytes) - 1);
        out[bytes - 1] = '\0';
    }
}

}  // namespace

FAKE_EXPORT Version Vship_GetVersion() { return {5, 1, 2, 0}; }

FAKE_EXPORT int Vship_GetDeviceCount(int* count) {
    *count = static_cast<int>(gpus().size());
    return 0;
}

FAKE_EXPORT int Vship_GetDeviceInfo(DeviceInfo* info, int gpu_id) {
    const std::vector<Gpu> listed = gpus();
    if (gpu_id < 0 || gpu_id >= static_cast<int>(listed.size())) {
        return 1;
    }
    *info = DeviceInfo{};
    copy_text(listed[static_cast<std::size_t>(gpu_id)].name.c_str(), info->name, sizeof info->name);
    info->integrated = listed[static_cast<std::size_t>(gpu_id)].integrated ? 1 : 0;
    return 0;
}

FAKE_EXPORT int Vship_GPUFullCheck(int gpu_id) { return fails(gpu_id) ? 2 : 0; }

FAKE_EXPORT int Vship_GetErrorMessage(int code, char* out, int bytes) {
    copy_text(code == 2 ? "the fake check failed" : "fake error", out, bytes);
    return 0;
}

FAKE_EXPORT int Vship_GetDetailedLastError(char* out, int bytes) {
    copy_text("", out, bytes);
    return 0;
}

// The rest of the API, which the app checks is there; a probe calls none of it.
FAKE_EXPORT int Vship_PinnedMalloc2(void**, std::uint64_t, int) { return 1; }
FAKE_EXPORT int Vship_PinnedFree2(void*, int) { return 1; }
FAKE_EXPORT int Vship_InitHandler(void**, void*) { return 1; }
FAKE_EXPORT int Vship_FreeHandler(void*) { return 1; }
FAKE_EXPORT int Vship_ComputeHandler(void*, void*, const std::uint8_t* const*, const std::uint8_t* const*,
                                     const std::int64_t*, const std::int64_t*) {
    return 1;
}
FAKE_EXPORT int Vship_GetDetailedLastErrorHandler(void*, char* out, int bytes) {
    copy_text("", out, bytes);
    return 0;
}
FAKE_EXPORT int Vship_ResetScore(void*) { return 1; }
