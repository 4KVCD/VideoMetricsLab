#include "devices.h"

#include <future>
#include <optional>
#include <string>
#include <utility>

#include "arguments.h"
#include "vml/ffmpeg_run.h"
#include "vml/isolated.h"
#include "vml/libvmaf.h"
#include "vml/metrics.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/settings.h"

namespace vml::cli {

std::vector<std::string> device_lines(const DeviceFindings& found) {
    std::vector<std::string> lines;
    if (found.tools.ok()) {
        lines.push_back("FFmpeg: " + format_version(found.tools.ffmpeg.version));
    } else {
        for (const std::string& problem : found.tools.problems()) {
            lines.push_back("FFmpeg: NOT USABLE: " + problem);
        }
    }
    if (found.vship) {
        const std::string vship = "GPU: " + found.vship->name + " (" + vship_build_name(*found.vship) + ", Vulkan)";
        lines.push_back("CVVDP: " + vship);
        for (const std::string key : {"ssimulacra2", "butteraugli"}) {
            lines.push_back(std::string(metric_definition(key).label) + ": " + vship);
        }
    } else {
        lines.push_back("CVVDP: not available (" + found.no_vship + ")");
        for (const std::string key : {"ssimulacra2", "butteraugli"}) {
            lines.push_back(std::string(metric_definition(key).label) + ": CPU (libjxl)");
        }
    }
    if (found.other_vship) {
        const auto& [device, reason] = *found.other_vship;
        lines.push_back("With --vship " + std::string(vship_build_value(found.other_build)) + ": " +
                        (device ? "GPU: " + device->name + " (" + vship_build_name(*device) + ", Vulkan)"
                                : "not available (" + reason + ")"));
    }
    if (!found.vmaf.backend.empty()) {
        lines.push_back("VMAF v0.6.1, VMAF NEG: GPU (" +
                        (found.vmaf.backend == "cuda" ? std::string("libvmaf CUDA")
                                                      : "Vulkan, GPU " + std::to_string(found.vmaf.device.value_or(0))) +
                        ")");
    } else {
        lines.push_back("VMAF v0.6.1, VMAF NEG: CPU (" + found.vmaf.text + ")");
    }
    lines.push_back("VMAF v1: " + std::string(found.v1.available ? "GPU (" : "CPU (") + found.v1.text + ")");
    lines.push_back(found.cpu_metrics_in_app ? "PSNR, SSIM, XPSNR: CPU (libvmaf-fast " + found.libvmaf_fast_version +
                                                   " in the app; FFmpeg beside VMAF on the CPU)"
                                             : "PSNR, SSIM, XPSNR: CPU (FFmpeg)");
    return lines;
}

int devices() {
    // Ctrl+C ends the command at once with exit code 130, as the Python
    // app's KeyboardInterrupt ends it; a check under way ends with it.
    platform::on_interrupt([] { platform::end_process_now(exit_cancelled); });
    const Settings settings = Settings::load();
    // The GPU checks together rather than one after another: each starts a
    // process of its own (Vship's takes 1.8 s, VMAF's and VMAF v1's 0.5 s).
    // The metrics' lines are for the app's Vship build; the other bundled
    // build is checked for a line of its own.
    const VshipBuild build = default_vship_build(settings.default_vship_build);
    const VshipBuild other = build == VshipBuild::fast ? VshipBuild::stock : VshipBuild::fast;
    std::future<std::pair<std::optional<VshipDevice>, std::string>> vship_check =
        std::async(std::launch::async, [build] { return detect_vship_device(build, gpu_probes_token()); });
    std::future<std::pair<std::optional<VshipDevice>, std::string>> other_check;
    if (vship51_bundled()) {
        other_check = std::async(std::launch::async, [other] { return detect_vship_device(other, gpu_probes_token()); });
    }
    std::future<GpuVmafBackend> vmaf_check = std::async(std::launch::async, [] { return gpu_vmaf_backend(gpu_probes_token()); });
    std::future<V1Probe> v1_check = std::async(std::launch::async, [] { return v1_available(gpu_probes_token()); });
    DeviceFindings found;
    found.tools = check_tools(settings);
    const auto [vship, no_vship] = vship_check.get();
    found.vship = vship;
    found.no_vship = no_vship;
    if (other_check.valid()) {
        found.other_vship = other_check.get();
        found.other_build = other;
    }
    found.vmaf = vmaf_check.get();
    found.v1 = v1_check.get();
    found.cpu_metrics_in_app = cpu_metrics_in_app();
    found.libvmaf_fast_version = libvmaf_fast_version();
    std::string text = version_line() + "\n";
    for (const std::string& line : device_lines(found)) {
        text += line + "\n";
    }
    platform::write_text(platform::Stream::output, text);
    return found.tools.ok() ? exit_ok : exit_usage;
}

}  // namespace vml::cli
