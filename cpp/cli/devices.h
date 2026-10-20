// The devices command: FFmpeg's version, and where each metric is
// calculated on this PC, as the Python app's command line says it
// (vmaf_app/cli.py's devices).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "vml/ffmpeg_tools.h"
#include "vml/gpu_vmaf.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship.h"

namespace vml::cli {

// What the command finds on this PC.
struct DeviceFindings {
    ToolsStatus tools;
    std::optional<VshipDevice> vship;  // the GPU Vship uses, if there is one
    std::string no_vship;              // why there is none
    GpuVmafBackend vmaf;
    V1Probe v1;
    bool cpu_metrics_in_app = false;  // PSNR, SSIM and XPSNR scored by the app's libvmaf
    std::string libvmaf_fast_version;
};

// The command's lines, after the app's name and version.
std::vector<std::string> device_lines(const DeviceFindings& found);

// Runs the devices command: prints the app's name and version, then
// device_lines for this PC. Returns the exit code: 0, or 2 when FFmpeg
// cannot be used.
int devices();

}  // namespace vml::cli
