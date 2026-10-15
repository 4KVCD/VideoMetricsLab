#include "devices.h"

#include <future>
#include <optional>
#include <string>
#include <utility>

#include "arguments.h"
#include "vml/ffmpeg_run.h"
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
    lines.push_back("GPU backend setting: " + found.gpu_backend);
    if (found.vship) {
        const std::string vship = "GPU: " + found.vship->name + " (Vship " + found.vship->version + ", " +
                                  backend_label(found.vship->backend) + ")";
        lines.push_back("CVVDP: " + vship);
        for (const std::string key : {"ssimulacra2", "butteraugli"}) {
            lines.push_back(std::string(metric_definition(key).label) + ": " +
                            (scores_correctly(*found.vship, key)
                                 ? vship
                                 : "CPU (libjxl): this Vship build scores it wrongly on this GPU"));
        }
    } else {
        lines.push_back("CVVDP: not available (" + found.no_vship + ")");
        for (const std::string key : {"ssimulacra2", "butteraugli"}) {
            lines.push_back(std::string(metric_definition(key).label) + ": CPU (libjxl)");
        }
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
    set_gpu_backend(settings.gpu_backend);
    set_vship_backend(settings.gpu_backend);
    // The GPU checks together rather than one after another: each starts a
    // process of its own (Vship's took 1.8 s, VMAF's and VMAF v1's 0.5 s).
    std::future<std::pair<std::optional<VshipDevice>, std::string>> vship_check =
        std::async(std::launch::async, [] { return detect_vship_device(); });
    std::future<GpuVmafBackend> vmaf_check = std::async(std::launch::async, [] { return gpu_vmaf_backend(); });
    std::future<V1Probe> v1_check = std::async(std::launch::async, [] { return v1_available(); });
    DeviceFindings found;
    found.tools = check_tools(settings);
    found.gpu_backend = settings.gpu_backend;
    const auto [vship, no_vship] = vship_check.get();
    found.vship = vship;
    found.no_vship = no_vship;
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
