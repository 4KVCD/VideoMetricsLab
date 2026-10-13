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
// Not yet: the CPU calculating SSIMULACRA2 and Butteraugli (the command line
// refuses such a run before it starts).
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
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

// What a perceptual task is given: the two videos, the run's options (black
// bars, scaling, the duration limit, frame subsampling, decoding), where
// SSIMULACRA2 and Butteraugli are each set to run, and the display CVVDP
// models.
struct PerceptualRequest {
    RunEnvironment environment;
    VideoInfo source;
    VideoInfo distorted;
    VmafOptions options;
    // "gpu" or "cpu" for SSIMULACRA2 and Butteraugli; "gpu" where not given.
    std::map<std::string, std::string, std::less<>> backends;
    CvvdpSettings cvvdp;

    // Where `key` is set to run: CVVDP on the GPU only (its official CPU
    // implementation needs PyTorch, almost 1 GB, and takes seconds per 4K
    // frame); the others as set.
    std::string backend_of(std::string_view key) const;

    // Every how many frames `key` is scored (perceptual_step).
    int step_of(std::string_view key) const;
};

// Every how many frames perceptual metric `key` is scored: every
// `n_subsample`-th frame; CVVDP, which models motion over time, every frame.
int perceptual_step(std::string_view key, int n_subsample);

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

// Calculates `keys` (SSIMULACRA2, Butteraugli, CVVDP) where each is set to
// run (the Python app's apply_vship_cpu_fallback): on the GPU through
// Vship, each in a pass of its own, or with `together` in one pass. A
// metric that fails while the others finish is in the output's failures;
// when every pass fails, the PerceptualRunError thrown says why, CVVDP's
// own reason apart, as it runs on the GPU only. Throws Cancelled,
// ComparisonCutShortError (a video ends long before its stated length),
// PerceptualRunError, and CropDetectError.
PerceptualTaskOutput run_perceptual_task(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                         bool together, const PerceptualHooks& hooks, const CancelToken& cancel);

// The isolated "vship-pass" task's work (isolated.cpp): one pass, in this
// process, from the request the parent wrote; the scores are left in
// `folder`.
nlohmann::json vship_pass_task(const nlohmann::json& request, const std::filesystem::path& folder);

}  // namespace vml
