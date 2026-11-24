// The CPU route's metrics that the bundled Vulkan engine also calculates,
// CIEDE2000 and CAMBI, on the GPU from the pictures CpuScorer gives libvmaf's
// extractors. On the CPU they were the slowest of its metrics: at 4K 10-bit
// beside VMAF on an RTX 5090 a 240-frame run took 10.0 s with CIEDE2000 and
// 5.8 s with CAMBI against 3.7 s for VMAF alone.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vml/libvmaf.h"

namespace vml {

// Which of `metrics` (CpuScorer's keys) the GPU may calculate: CIEDE2000
// unless the size is odd (the engine copies chroma planes (w + 1) / 2 x
// (h + 1) / 2, libvmaf's pictures hold w / 2 x h / 2), and CAMBI where the
// GPU takes all of its libvmaf options (`cambi_options`: not a test video
// scored at another size or bit depth).
std::vector<std::string> gpu_metrics_of(const std::vector<std::string>& metrics, bool odd_size,
                                        const std::vector<std::pair<std::string, std::string>>& cambi_options);

// Calculates CIEDE2000 and CAMBI of the pairs given, in a VMAF v1 context of
// the engine without VMAF's own features.
class GpuMetricsScorer {
public:
    // `metrics` from gpu_metrics_of; `device` is Vulkan's number for the GPU,
    // -1 for the one VMAF would run on (chosen_vulkan_device). A metric the
    // engine does not start is left out (metrics()). Throws VmafVulkanError
    // when it starts none.
    GpuMetricsScorer(int width, int height, int bit_depth, int device, std::vector<std::string> metrics,
                     const std::vector<std::pair<std::string, std::string>>& cambi_options);
    ~GpuMetricsScorer();
    GpuMetricsScorer(const GpuMetricsScorer&) = delete;
    GpuMetricsScorer& operator=(const GpuMetricsScorer&) = delete;

    // The metrics it calculates.
    const std::vector<std::string>& metrics() const;

    // Scores one more pair, libvmaf's pictures of it (4:2:0 where CIEDE2000
    // is calculated; CAMBI reads the test picture's luma). The planes are
    // copied before it returns.
    void add(const VmafPicture& reference, const VmafPicture& distorted);

    // Each metric's values for the pairs given, in order: CIEDE2000 as
    // libvmaf's 45 - 20 log10(dE00) (+infinity for identical pictures),
    // CAMBI's score. Throws VmafVulkanError.
    std::map<std::string, std::vector<double>> finish();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace vml
