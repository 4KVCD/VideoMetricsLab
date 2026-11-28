// The CPU route's metrics that the bundled Vulkan engine also calculates,
// CIEDE2000, CAMBI, MS-SSIM and MS-SSIM YUV, on the GPU from the pictures
// CpuScorer gives libvmaf's extractors. On the CPU they were the slowest of its metrics: at 4K 10-bit
// beside VMAF on an RTX 5090 a 240-frame run took 10.0 s with CIEDE2000 and
// 5.8 s with CAMBI against 3.7 s for VMAF alone.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/libvmaf.h"
#include "vml/options.h"
#include "vml/pair_scorer.h"

namespace vml {

class GpuFrameStream;  // gpu_frames.h

// Which of `metrics` (CpuScorer's keys) the GPU may calculate: CIEDE2000
// and MS-SSIM YUV with its planes unless the size is odd (the engine copies
// chroma planes (w + 1) / 2 x (h + 1) / 2, libvmaf's pictures hold w / 2 x
// h / 2), MS-SSIM unless MS-SSIM YUV stays on the CPU (libvmaf calculates its
// luma in the same pass), and CAMBI where the GPU takes all of its libvmaf
// options (`cambi_options`: not a test video scored at another size or bit
// depth). None of `kept_on_cpu` (those set to the CPU: kept_on_cpu_of).
std::vector<std::string> gpu_metrics_of(const std::vector<std::string>& metrics, bool odd_size,
                                        const std::vector<std::pair<std::string, std::string>>& cambi_options,
                                        const std::vector<std::string>& kept_on_cpu = {});

// Whether `key` is one of the CPU metrics the Vulkan engine may calculate (gpu_metrics_of): CIEDE2000, CAMBI,
// MS-SSIM, and MS-SSIM YUV with its planes.
bool engine_takes_cpu_metric(std::string_view key);

// Those of `keys` the engine may take that `options` set to the CPU (VmafOptions::on_gpu_chosen), every MS-SSIM
// key together: libvmaf calculates MS-SSIM's luma in MS-SSIM YUV's pass.
std::vector<std::string> kept_on_cpu_of(const std::vector<std::string>& keys, const VmafOptions& options);

// Which of `keys` (CpuScorer's) may go to the GPU, in `keys`' order: gpu_metrics_of's with `extra_gpu_offered` (the
// size and CAMBI's options aside), VMAF YUV's U and V (and VMAF YUV, as a task names them) with
// `chroma_vmaf_offered` (VMAF's GPU is the Vulkan engine's: GpuPlan::vmaf_on_vulkan). Where they go is decided only
// as the scorer starts: it keeps a metric the GPU cannot take (ChromaVmaf's U and V above 10 bits or with a model
// file, say) on the CPU. None of `kept_on_cpu` (gpu_metrics_of's).
std::vector<std::string> may_run_on_gpu(const std::vector<std::string>& keys, bool extra_gpu_offered,
                                        bool chroma_vmaf_offered, const std::vector<std::string>& kept_on_cpu = {});

// The CPU scorer's metrics `keys` as a task's name in the log and vml-cli's list of what a run calculates gives
// them: "PSNR; CAMBI and CIEDE2000 (ΔE) on the GPU where it can", those in `maybe_gpu` (may_run_on_gpu) named so,
// as where they go is decided only as the scorer starts (it says so then: StatusKind::metrics_on_gpu).
std::string cpu_metrics_text(const std::vector<std::string>& keys, const std::vector<std::string>& maybe_gpu);

// The same before VMAF's GPU backend is known: CAMBI, CIEDE2000 and MS-SSIM named so where `options` set them
// to the GPU, VMAF YUV's U and V and HDR VMAF (its features) with `vmaf_offered` (VMAF's place is the GPU).
std::string cpu_metrics_text(const std::vector<std::string>& keys, const VmafOptions& options, bool vmaf_offered);

// gpu_metrics_of's metrics on the GPU failed after they started (a lost device, say),
// as CpuScorer says it: the pass is made again with them in libvmaf on the
// CPU (score_plan), its other metrics failing for this alone otherwise.
class GpuMetricsFailed : public LibvmafError {
public:
    using LibvmafError::LibvmafError;
};

// Calculates CIEDE2000, CAMBI, MS-SSIM and MS-SSIM YUV of the pairs given, in
// a VMAF v1 context of the engine without VMAF's own features.
class GpuMetricsScorer {
public:
    // `metrics` from gpu_metrics_of; `device` is Vulkan's number for the GPU,
    // -1 for the one VMAF would run on (chosen_vulkan_device). A metric the
    // engine does not start is left out (metrics()). Throws VmafVulkanError
    // when it starts none. With `shared` (the test video's decoder, which
    // shares its GPU memory with the source's) and metrics that read only the
    // luma (MS-SSIM, CAMBI), the decoders copy the pairs' luma planes on the
    // GPU into the engine's memory, which `shared` imports (SharedSlots,
    // takes_shared): the pictures never go through system memory. Where the
    // import fails, the pairs go through system memory as without it (the log
    // says why).
    GpuMetricsScorer(int width, int height, int bit_depth, int device, std::vector<std::string> metrics,
                     const std::vector<std::pair<std::string, std::string>>& cambi_options,
                     GpuFrameStream* shared = nullptr);
    ~GpuMetricsScorer();
    GpuMetricsScorer(const GpuMetricsScorer&) = delete;
    GpuMetricsScorer& operator=(const GpuMetricsScorer&) = delete;

    // The metrics it calculates.
    const std::vector<std::string>& metrics() const;

    // Whether it reads the reference picture: CIEDE2000 and MS-SSIM do, CAMBI
    // reads the test picture's luma alone. Without it, add takes any
    // reference picture, as its planes are not read.
    bool reads_reference() const;

    // Scores one more pair, libvmaf's pictures of it (4:2:0 where CIEDE2000
    // or MS-SSIM YUV is calculated; CAMBI reads the test picture's luma). The planes are
    // copied before it returns.
    void add(const VmafPicture& reference, const VmafPicture& distorted);

    // Whether it reads nothing but the two luma planes: MS-SSIM (with CAMBI
    // or not) without MS-SSIM YUV or CIEDE2000. A caller whose pictures
    // nothing else reads then gives it the planes alone (add_luma,
    // add_decoded), skipping the copy into libvmaf's picture.
    bool takes_luma_alone() const;

    // Scores one more pair from its two luma planes (takes_luma_alone), which
    // the engine copies before it returns.
    void add_luma(Plane reference, Plane distorted);

    // Scores one more pair (takes_luma_alone) whose luma planes `write` puts
    // straight into the engine's memory: it is given where the source's and
    // the test video's go, each to be written packed (width x sample bytes a
    // row), as a decoder downloads it.
    void add_decoded(const std::function<void(std::uint8_t* reference, std::uint8_t* distorted)>& write);

    // Whether the decoders' pictures reach the engine on the GPU (the
    // constructor's `shared`): the pairs are then given only to add_shared
    // (add, add_luma and add_decoded throw std::logic_error).
    bool takes_shared() const;

    // Scores one more pair (takes_shared) held by the decoders: each copies
    // its luma plane on the GPU into the engine's next slot (the source's
    // only where it is read: not for CAMBI alone), then the pair is
    // committed.
    void add_shared(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot);

    // The values for the pairs given, in order, by libvmaf's feature names:
    // "ciede2000" as libvmaf's 45 - 20 log10(dE00) (+infinity for identical
    // pictures), "float_ms_ssim" and with MS-SSIM YUV "float_ms_ssim_cb" and
    // "float_ms_ssim_cr"; CAMBI's score as "cambi" (libvmaf names it from its
    // options). Throws VmafVulkanError.
    std::map<std::string, std::vector<double>> finish();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace vml
