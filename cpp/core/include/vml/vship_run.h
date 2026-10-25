// SSIMULACRA2, Butteraugli and CVVDP scored on the GPU by Vship (vship.h),
// as the Python app scores them (the rest of
// vmaf_app/core/perceptual_vship.py): each video's frames are decoded,
// cropped, sampled and scaled into a ring of Vship's pinned memory, by a
// decoder of the app's in the pass's process where one takes the video
// (gpu_frames.h), else by FFmpeg; each frame pair is handed to every
// metric's handlers, two per metric on threads of their own, and CVVDP's
// one, which takes the pairs in order; a pass scores one or more metrics
// over both videos, in a process of its own; and a task's metrics are
// scored in one pass or several. Frames are paired frame N with frame N, or
// by timestamp (frame_sync_by_time.h) when VML_FRAME_PAIRING says so.
//
// A task's metrics go where each is set to run, and to the CPU's tools
// (perceptual_cpu.h) where the GPU cannot calculate them
// (run_perceptual_task).
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/cvvdp.h"
#include "vml/ffmpeg_run.h"
#include "vml/perceptual.h"
#include "vml/vship.h"

namespace vml {

// The tasks of the perceptual metrics, as the job runner names them:
// Vship's on the GPU, and SSIMULACRA2 and Butteraugli chosen for the CPU.
inline constexpr std::string_view perceptual_task = "perceptual";
inline constexpr std::string_view perceptual_cpu_task = "perceptual_cpu";

// Whether `key` is one of the perceptual metrics: SSIMULACRA2, Butteraugli
// and CVVDP.
bool is_perceptual_metric(std::string_view key);

// What a task reports beside its progress and statuses: each GPU pass's
// scores as soon as the pass finishes (on_pass_done), so that they are
// shown and kept at once; each pass as it starts, with its number, how
// many there are and its metrics (on_pass); and the metrics the CPU is
// about to calculate (on_cpu).
struct PerceptualHooks {
    ProgressCallback on_progress;
    StatusCallback on_status;
    std::function<void(const PerceptualTaskOutput&)> on_pass_done;
    std::function<void(int number, int count, const std::vector<std::string>& keys)> on_pass;
    std::function<void(const std::vector<std::string>& keys)> on_cpu;
};

// The passes `keys` are scored in, in order: one per metric, or with
// `together` (Settings > GPU metrics) one per set of frames scored, so that
// every metric scoring the same frames shares one pass and the videos are
// decoded once for them all.
std::vector<std::vector<std::string>> vship_passes(const std::vector<std::string>& keys, int n_subsample,
                                                   bool together);

// The steps run_perceptual_task takes a task's metrics through, each a
// function, so that the tests can stand in for them (the Python app's tests
// replace the same functions): finding the GPU Vship uses, and whether a GPU
// check failed where none was found (none: it did not), finding the black
// bars, Vship-fast's passes on that GPU, the same passes on stock Vship
// 5.1.2 (none where it is not bundled), and the CPU's tools with the black
// bars already found, if they were.
struct PerceptualRoutes {
    using GpuRoute = std::function<PerceptualTaskOutput(const PerceptualRequest&, const std::vector<std::string>& keys,
                                                        const VshipDevice&, const CropPair&, bool together,
                                                        const PerceptualHooks&, const CancelToken&)>;
    std::function<std::pair<std::optional<VshipDevice>, std::string>()> find_gpu;
    std::function<bool()> gpu_check_failed;
    std::function<CropPair(const PerceptualRequest&, const StatusCallback&, const CancelToken&)> find_crops;
    GpuRoute run_gpu;
    GpuRoute run_vship51;
    std::function<PerceptualTaskOutput(const PerceptualRequest&, const std::vector<std::string>& keys,
                                       const std::optional<CropPair>&, const ProgressCallback&,
                                       const StatusCallback&, const CancelToken&)>
        run_cpu;
};

// The app's routes: the Vship probe (after forgetting one that failed a
// while ago, forget_failed_vship_probe), resolve_crops, Vship's passes on
// Vship-fast and, where it is bundled, on stock Vship 5.1.2, and
// run_cpu_perceptual_task.
PerceptualRoutes app_perceptual_routes();

// Calculates `keys` (SSIMULACRA2, Butteraugli, CVVDP) where each is set to
// run (the Python app's apply_vship_cpu_fallback): on the GPU through
// Vship-fast, each in a pass of its own, or with `together` in one pass; or
// on the CPU. What Vship-fast fails on the GPU is calculated again with
// stock Vship 5.1.2 where it is bundled, but a metric refused for the size
// compared at (size_refusal); on a GPU only 5.1.2 found
// (VshipDevice::vship51) every metric runs on 5.1.2. A metric moves from
// the GPU to the CPU where the GPU cannot calculate it, whatever the
// video's length: without a GPU Vship can use, and after it failed on the
// GPU with both builds.
// CVVDP runs on the GPU only, and fails with its own reason. A metric that
// fails while others finish is in the output's failures. `hooks.on_cpu`
// gets the metrics the CPU is about to calculate, just before it starts;
// the progress after it is the CPU's own, from 0. Throws Cancelled,
// ComparisonCutShortError (a video ends long before its stated length),
// PerceptualRunError, and CropDetectError.
PerceptualTaskOutput run_perceptual_task(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                         bool together, const PerceptualHooks& hooks, const CancelToken& cancel,
                                         const PerceptualRoutes& routes = app_perceptual_routes());

// The isolated "vship-pass" task's work (isolated.cpp): one pass, in this
// process, from the request the parent wrote; the scores are left in
// `folder`.
nlohmann::json vship_pass_task(const nlohmann::json& request, const std::filesystem::path& folder);

}  // namespace vml
