// The devices command: FFmpeg's version, and where each metric is
// calculated on this PC, as the Python app's command line says it
// (vmaf_app/cli.py's devices).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vml/ffmpeg_tools.h"
#include "vml/gpu_vmaf.h"
#include "vml/settings.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship.h"

namespace vml::cli {

// What the command finds on this PC.
struct DeviceFindings {
    std::optional<std::string> chosen_gpu;  // the GPU chosen to score on (gpu_choice.h), if any
    ToolsStatus tools;
    std::optional<VshipDevice> vship;  // the GPU the app's Vship build uses, if there is one
    std::string no_vship;              // why there is none
    // Where both Vship builds are bundled, the other one's (--vship
    // `other_build`) check: its GPU, or why it has none.
    std::optional<std::pair<std::optional<VshipDevice>, std::string>> other_vship;
    VshipBuild other_build = VshipBuild::stock;
    GpuVmafBackend vmaf;
    V1Probe v1;
    bool cpu_metrics_in_app = false;  // PSNR, SSIM and XPSNR scored by the app's libvmaf
    std::string libvmaf_fast_version;
};

// The command's lines, after the app's name and version.
std::vector<std::string> device_lines(const DeviceFindings& found);

// The same as JSON (--json; docs/schemas/devices.schema.json): document_head
// (devices-1), the GPUs this PC lists, the one chosen, FFmpeg, then for each
// metric where it would be calculated ("on": gpu or cpu, null where it
// cannot be), on which GPU, and the line's words as "detail".
nlohmann::ordered_json devices_json(const DeviceFindings& found, const std::vector<std::string>& gpus);

// devices --json's format.
inline constexpr const char* devices_format = "devices-1";

// Runs the devices command: prints the app's name and version, then
// device_lines for this PC, by `settings` (the app's, --settings' or the
// defaults), or with `json_path` devices_json ("-": printed in their place).
// Returns the exit code: 0, 1 when the JSON file could not be written, or 2
// when FFmpeg cannot be used.
int devices(const Settings& settings, const std::optional<std::string>& json_path);

}  // namespace vml::cli
