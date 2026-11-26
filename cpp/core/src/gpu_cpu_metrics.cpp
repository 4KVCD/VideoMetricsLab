#include "vml/gpu_cpu_metrics.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
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

// MS-SSIM YUV or one of its planes.
bool of_ms_ssim_yuv(const std::string& key) { return scored_by(key) == "ms_ssim_yuv"; }

// The engine's columns of MS-SSIM's planes, Y, Cb and Cr, by libvmaf's feature names.
constexpr std::array<const char*, 3> ms_ssim_features{"float_ms_ssim", "float_ms_ssim_cb", "float_ms_ssim_cr"};

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
    // MS-SSIM YUV's chroma as CIEDE2000's. Where it stays on the CPU, libvmaf
    // calculates MS-SSIM's luma in the same pass, so MS-SSIM stays with it.
    const bool ms_ssim_yuv = std::any_of(metrics.begin(), metrics.end(), of_ms_ssim_yuv);
    if (has(metrics, "ms_ssim") && !(ms_ssim_yuv && odd_size)) {
        on_gpu.emplace_back("ms_ssim");
    }
    if (!odd_size) {
        std::copy_if(metrics.begin(), metrics.end(), std::back_inserter(on_gpu), of_ms_ssim_yuv);
    }
    return on_gpu;
}

bool engine_takes_cpu_metric(std::string_view key) {
    return key == "ciede2000" || key == "cambi" || key == "ms_ssim" || key == "ms_ssim_yuv" ||
           key == "ms_ssim_yuv_y" || key == "ms_ssim_yuv_u" || key == "ms_ssim_yuv_v";
}

std::vector<std::string> may_run_on_gpu(const std::vector<std::string>& keys, bool extra_gpu_offered,
                                        bool chroma_vmaf_offered) {
    const std::vector<std::string> extra =
        extra_gpu_offered ? gpu_metrics_of(keys, false, {}) : std::vector<std::string>{};
    // VMAF YUV, as a task names U and V asked for it alone (job_runner's as_asked), is VMAF's Y and those planes.
    const auto chroma_vmaf = [](const std::string& key) {
        return key == vmaf_u_key || key == vmaf_v_key || key == "vmaf_yuv";
    };
    std::vector<std::string> maybe;
    for (const std::string& key : keys) {
        if (has(extra, key) || (chroma_vmaf_offered && chroma_vmaf(key))) {
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
    std::vector<std::string> maybe_gpu = may_run_on_gpu(keys, gpu_offered, gpu_offered);
    // HDR VMAF, scored in the app rather than by CpuScorer, follows VMAF's place too: its features on the engine
    // (hdr_plan). maybe_gpu is a set to cpu_metrics_text, so its order does not matter.
    if (gpu_offered && has(keys, "vmaf_hdr")) {
        maybe_gpu.emplace_back("vmaf_hdr");
    }
    return cpu_metrics_text(keys, maybe_gpu);
}

struct GpuMetricsScorer::State {
    const VulkanEngine& engine = vulkan_engine();
    VulkanContext* gpu = nullptr;
    std::vector<std::string> metrics;
    int ms_ssim_planes = 0;  // 1: the luma (MS-SSIM), 3: Y, Cb and Cr (MS-SSIM YUV); 0: none
    unsigned count = 0;

    void check(int error, const std::string& what) const { check_vulkan(engine, error, what); }
    // CIEDE2000 and MS-SSIM YUV read both pictures' chroma.
    bool reads_chroma() const { return has(metrics, "ciede2000") || ms_ssim_planes == 3; }

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
    // options or the size) is left to libvmaf. `keys` are started together.
    const auto start = [&](std::vector<std::string> keys, auto&& begin) {
        std::erase_if(keys, [&](const std::string& key) { return !has(metrics, key); });
        if (keys.empty()) {
            return false;
        }
        const int code = begin();
        if (code == 0) {
            s.metrics.insert(s.metrics.end(), keys.begin(), keys.end());
            return true;
        }
        logger.warning(metric_labels(keys, " and ") + " on the CPU: the GPU did not start " +
                       (keys.size() == 1 ? "it" : "them") + " (" +
                       (code == unsupported_options ? std::string("options it does not take")
                                                    : std::string(engine.error())) +
                       ")");
        return false;
    };
    start({"ciede2000"}, [&] {
        return engine.v1_ciede != nullptr && engine.v1_ciede_scores != nullptr ? engine.v1_ciede(s.gpu)
                                                                               : unsupported_options;
    });
    start({"cambi"}, [&] {
        const std::optional<std::array<double, 9>> values = cambi_gpu_values(cambi_options);
        return engine.v1_cambi != nullptr && engine.v1_cambi_scores != nullptr && values
                   ? engine.v1_cambi(s.gpu, values->data())
                   : unsupported_options;
    });
    // MS-SSIM and MS-SSIM YUV share the luma's scales.
    const int ms_ssim_planes = std::any_of(metrics.begin(), metrics.end(), of_ms_ssim_yuv) ? 3 : 1;
    if (start({"ms_ssim", "ms_ssim_yuv", "ms_ssim_yuv_y", "ms_ssim_yuv_u", "ms_ssim_yuv_v"}, [&] {
            return engine.v1_ms_ssim != nullptr && engine.v1_ms_ssim_scores != nullptr
                       ? engine.v1_ms_ssim(s.gpu, ms_ssim_planes)
                       : unsupported_options;
        })) {
        s.ms_ssim_planes = ms_ssim_planes;
    }
    if (s.metrics.empty()) {
        throw VmafVulkanError("the GPU started none of " + std::to_string(metrics.size()) + " metrics");
    }
}

GpuMetricsScorer::~GpuMetricsScorer() = default;

const std::vector<std::string>& GpuMetricsScorer::metrics() const { return state_->metrics; }

bool GpuMetricsScorer::reads_reference() const {
    return has(state_->metrics, "ciede2000") || state_->ms_ssim_planes != 0;
}

void GpuMetricsScorer::add(const VmafPicture& reference, const VmafPicture& distorted) {
    State& s = *state_;
    const auto plane = [](const VmafPicture& picture, int index) {
        return static_cast<const std::uint8_t*>(picture.data[index]);
    };
    int code = 0;
    if (s.reads_chroma()) {
        const std::array<const std::uint8_t*, 6> planes{plane(reference, 0), plane(reference, 1),
                                                        plane(reference, 2), plane(distorted, 0),
                                                        plane(distorted, 1), plane(distorted, 2)};
        const std::array<std::ptrdiff_t, 6> strides{reference.stride[0], reference.stride[1], reference.stride[2],
                                                    distorted.stride[0], distorted.stride[1], distorted.stride[2]};
        code = s.engine.submit_v1(s.gpu, planes.data(), strides.data(), 1);
    } else if (s.ms_ssim_planes != 0) {  // MS-SSIM's luma
        code = s.engine.submit(s.gpu, plane(reference, 0), reference.stride[0], plane(distorted, 0),
                               distorted.stride[0], 1);
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
    if (s.count != 0) {
        s.check(s.engine.flush(s.gpu), "Scoring");
    }
    const auto read = [&](const std::string& column_name, auto&& into) {
        std::vector<double>& column = scores[column_name];
        column.resize(s.count);
        if (s.count != 0) {
            s.check(into(column.data()), "Reading " + column_name);
        }
    };
    if (has(s.metrics, "ciede2000")) {
        read("ciede2000", [&](double* out) { return s.engine.v1_ciede_scores(s.gpu, out, s.count); });
    }
    if (has(s.metrics, "cambi")) {
        read("cambi", [&](double* out) { return s.engine.v1_cambi_scores(s.gpu, out, s.count); });
    }
    for (int plane = 0; plane < s.ms_ssim_planes; ++plane) {
        read(ms_ssim_features[plane],
             [&](double* out) { return s.engine.v1_ms_ssim_scores(s.gpu, out, s.count, plane); });
    }
    return scores;
}

}  // namespace vml
