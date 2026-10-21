// What SSIMULACRA2, Butteraugli and CVVDP share wherever they are
// calculated: what a task is given, their errors, a task's output, and the
// checks made before a run. A port of the shared parts of
// vmaf_app/core/perceptual_cpu.py.
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/cvvdp.h"
#include "vml/ffmpeg_run.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml {

// What a perceptual task is given: the two videos, the run's options (black
// bars, scaling, the duration limit, frame subsampling, decoding), where
// SSIMULACRA2 and Butteraugli are each set to run, the display CVVDP models,
// and how many tasks the CPU's queue runs at once, this one included, which
// sets the share of the cores and memory SSIMULACRA2 and Butteraugli take
// on the CPU (scoring_workers in perceptual_cpu.h).
struct PerceptualRequest {
    RunEnvironment environment;
    VideoInfo source;
    VideoInfo distorted;
    VmafOptions options;
    // "gpu" or "cpu" for SSIMULACRA2 and Butteraugli; "gpu" where not given.
    std::map<std::string, std::string, std::less<>> backends;
    CvvdpSettings cvvdp;
    int concurrent_cpu_tasks = 1;

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

// The black-bar crops of the source and the test video.
using CropPair = std::pair<std::optional<CropBox>, std::optional<CropBox>>;

// SSIMULACRA2, Butteraugli or CVVDP could not be calculated.
class PerceptualRunError : public std::runtime_error {
public:
    // `stderr_tail` is the end of the tool's error output.
    explicit PerceptualRunError(const std::string& message, std::string stderr_tail = "")
        : std::runtime_error(message), stderr_tail_(std::move(stderr_tail)) {}

    const std::string& stderr_tail() const { return stderr_tail_; }

private:
    std::string stderr_tail_;
};

// A video's pictures end long before its stated length (frame coverage):
// a fault of the file, which no other backend can mend, so it is not a GPU
// failure to retry on the CPU.
class ComparisonCutShortError : public PerceptualRunError {
public:
    using PerceptualRunError::PerceptualRunError;
};

// Each metric that could not be scored, with the reason, in the order they
// failed; a key appears once (the Python app's dict).
using PerceptualFailures = std::vector<std::pair<std::string, std::string>>;

// Sets `key`'s reason, in its place if it is there, else at the end.
void set_failure(PerceptualFailures& failures, const std::string& key, const std::string& reason);

// `key`'s reason, or nullptr.
const std::string* failure_of(const PerceptualFailures& failures, const std::string& key);

// The result of a SSIMULACRA2, Butteraugli or CVVDP task: the scores, the
// black-bar crops used, how many frames were compared, and the metrics that
// could not be scored while the others were (CVVDP failing beside
// SSIMULACRA2, say), which the run reports as a partial failure.
struct PerceptualTaskOutput {
    MetricResultSet metrics;
    std::optional<CropBox> source_crop;
    std::optional<CropBox> distorted_crop;
    long long compared_frame_count = 0;
    PerceptualFailures failures;
};

// For comparisons longer than this, SSIMULACRA2 and Butteraugli run on the
// CPU only when the user has agreed: the CPU tools take a second or two per
// 4K pair, which is days for a film. (A failed GPU run falls back to the
// CPU whatever its length.)
inline constexpr double long_cpu_run_seconds = 10 * 60;

// How much video a comparison scores: the shorter input, capped by the
// duration limit when one is set.
double compared_seconds(const VideoInfo& source, const VideoInfo& distorted, double duration_limit);

// Throws PerceptualRunError if the two videos cannot be compared by the
// window's rules (geometry.h's pair_problem), such as durations that do not
// match.
void validate_pair(const VideoInfo& source, const VideoInfo& distorted, double duration_limit);

// The black-bar crops of the source and the test video, or none when black
// bars are kept (`crop_mode`). Sends a status first, worded as for FFmpeg's
// metrics, because a video's tasks share one detection. Throws
// PerceptualRunError when the two pictures are different shapes after
// cropping (geometry.h's shape_problem), CropDetectError and Cancelled.
CropPair resolve_crops(const std::string& ffmpeg, const VideoInfo& source, const VideoInfo& distorted,
                       CropMode crop_mode, const CancelToken& cancel, const StatusCallback& on_status);

}  // namespace vml
