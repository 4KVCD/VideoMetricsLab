#include "vml/gpu_cpu_metrics.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

#include "vml/app_log.h"
#include "vml/chroma_vmaf.h"
#include "vml/metrics.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vmaf_vulkan.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.gpu_cpu_metrics");

// What vv_v1_cambi answers for options it does not calculate.
constexpr int unsupported_options = -4;

// vv_create_v1's ADM and motion options, libvmaf's defaults: unused, as the
// context calculates neither.
constexpr std::array<double, 9> unused_adm_motion{3.0, 1080, 0, 0.03125, 1.0, 0.0, 10000.0, 0, 0};

bool has(const std::vector<std::string>& metrics, const std::string& key) {
    return std::find(metrics.begin(), metrics.end(), key) != metrics.end();
}

}  // namespace

std::vector<std::string> gpu_metrics_of(const std::vector<std::string>& metrics, bool odd_size,
                                        const std::vector<std::pair<std::string, std::string>>& cambi_options) {
    std::vector<std::string> on_gpu;
    if (has(metrics, "ciede2000") && !odd_size) {
        on_gpu.emplace_back("ciede2000");
    }
    if (has(metrics, "cambi") && cambi_gpu_values(cambi_options)) {
        on_gpu.emplace_back("cambi");
    }
    return on_gpu;
}

std::vector<std::string> may_run_on_gpu(const std::vector<std::string>& keys, bool cie_cambi_offered,
                                        bool chroma_vmaf_offered) {
    const std::vector<std::string> cie_cambi =
        cie_cambi_offered ? gpu_metrics_of(keys, false, {}) : std::vector<std::string>{};
    // VMAF YUV, as a task names U and V asked for it alone (job_runner's as_asked), is VMAF's Y and those planes.
    const auto chroma_vmaf = [](const std::string& key) {
        return key == vmaf_u_key || key == vmaf_v_key || key == "vmaf_yuv";
    };
    std::vector<std::string> maybe;
    for (const std::string& key : keys) {
        if (has(cie_cambi, key) || (chroma_vmaf_offered && chroma_vmaf(key))) {
            maybe.push_back(key);
        }
    }
    return maybe;
}

std::string cpu_metrics_text(const std::vector<std::string>& keys, const std::vector<std::string>& maybe_gpu) {
    std::vector<std::string> here;
    std::vector<std::string> gpu;  // in `keys`' order
    for (const std::string& key : keys) {
        (has(maybe_gpu, key) ? gpu : here).push_back(key);
    }
    std::string text = metric_labels(here);
    if (!gpu.empty()) {
        text += (text.empty() ? "" : "; ") + metric_labels(gpu, " and ") + " on the GPU where it can";
    }
    return text;
}

std::string cpu_metrics_text(const std::vector<std::string>& keys, bool gpu_offered) {
    return cpu_metrics_text(keys, may_run_on_gpu(keys, gpu_offered, gpu_offered));
}

struct GpuMetricsScorer::State {
    const VulkanEngine& engine = vulkan_engine();
    VulkanContext* gpu = nullptr;
    std::vector<std::string> metrics;
    unsigned count = 0;

    void check(int error, const std::string& what) const { check_vulkan(engine, error, what); }
    bool ciede() const { return has(metrics, "ciede2000"); }

    ~State() {
        if (gpu != nullptr) {
            engine.destroy(gpu);  // waits for its frames
        }
    }
};

GpuMetricsScorer::GpuMetricsScorer(int width, int height, int bit_depth, int device, std::vector<std::string> metrics,
                                   const std::vector<std::pair<std::string, std::string>>& cambi_options)
    : state_(std::make_unique<State>()) {
    State& s = *state_;
    const VulkanEngine& engine = s.engine;
    if (engine.create_v1 == nullptr || engine.submit_v1 == nullptr) {
        throw VmafVulkanError("the bundled Vulkan engine has no VMAF v1 context");
    }
    if (device < 0) {
        const std::optional<VulkanDevice> chosen = chosen_vulkan_device(vulkan_devices());
        if (!chosen) {
            throw VmafVulkanError("no GPU that Vulkan can calculate on");
        }
        device = chosen->index;
    }
    s.check(engine.create_v1(&s.gpu, device, width, height, bit_depth,
                             vulkan_context_flags(false, false) | vulkan_no_vmaf_features_flags,
                             unused_adm_motion.data()),
            "Starting Vulkan");
    // Each metric the engine refuses (one it lacks, or cambi.c refusing the
    // options or the size) is left to libvmaf.
    const auto start = [&](const std::string& key, auto&& begin) {
        if (!has(metrics, key)) {
            return;
        }
        if (const int code = begin(); code == 0) {
            s.metrics.push_back(key);
        } else {
            logger.warning(key + " is calculated on the CPU: the GPU did not start it (" +
                           (code == unsupported_options ? std::string("options it does not take")
                                                        : std::string(engine.error())) +
                           ")");
        }
    };
    start("ciede2000", [&] {
        return engine.v1_ciede != nullptr && engine.v1_ciede_scores != nullptr ? engine.v1_ciede(s.gpu)
                                                                               : unsupported_options;
    });
    start("cambi", [&] {
        const std::optional<std::array<double, 9>> values = cambi_gpu_values(cambi_options);
        return engine.v1_cambi != nullptr && engine.v1_cambi_scores != nullptr && values
                   ? engine.v1_cambi(s.gpu, values->data())
                   : unsupported_options;
    });
    if (s.metrics.empty()) {
        throw VmafVulkanError("the GPU started none of " + std::to_string(metrics.size()) + " metrics");
    }
}

GpuMetricsScorer::~GpuMetricsScorer() = default;

const std::vector<std::string>& GpuMetricsScorer::metrics() const { return state_->metrics; }

bool GpuMetricsScorer::reads_reference() const { return state_->ciede(); }

void GpuMetricsScorer::add(const VmafPicture& reference, const VmafPicture& distorted) {
    State& s = *state_;
    const auto plane = [](const VmafPicture& picture, int index) {
        return static_cast<const std::uint8_t*>(picture.data[index]);
    };
    int code = 0;
    if (s.ciede()) {  // CIEDE2000 reads both pictures' chroma
        const std::array<const std::uint8_t*, 6> planes{plane(reference, 0), plane(reference, 1),
                                                        plane(reference, 2), plane(distorted, 0),
                                                        plane(distorted, 1), plane(distorted, 2)};
        const std::array<std::ptrdiff_t, 6> strides{reference.stride[0], reference.stride[1], reference.stride[2],
                                                    distorted.stride[0], distorted.stride[1], distorted.stride[2]};
        code = s.engine.submit_v1(s.gpu, planes.data(), strides.data(), 1);
    } else {
        // CAMBI alone, which reads only the test picture: the engine is given
        // its luma for the reference's too, as the caller need not have
        // filled the reference picture (reads_reference).
        code = s.engine.submit(s.gpu, plane(distorted, 0), distorted.stride[0], plane(distorted, 0),
                               distorted.stride[0], 1);
    }
    s.check(code, "Scoring pair " + std::to_string(s.count));
    ++s.count;
}

std::map<std::string, std::vector<double>> GpuMetricsScorer::finish() {
    State& s = *state_;
    std::map<std::string, std::vector<double>> scores;
    if (s.count == 0) {
        for (const std::string& key : s.metrics) {
            scores[key];
        }
        return scores;
    }
    s.check(s.engine.flush(s.gpu), "Scoring");
    for (const std::string& key : s.metrics) {
        std::vector<double>& column = scores[key];
        column.resize(s.count);
        s.check(key == "ciede2000" ? s.engine.v1_ciede_scores(s.gpu, column.data(), s.count)
                                   : s.engine.v1_cambi_scores(s.gpu, column.data(), s.count),
                "Reading " + key);
    }
    return scores;
}

}  // namespace vml
