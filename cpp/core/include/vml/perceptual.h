// What SSIMULACRA2, Butteraugli and CVVDP share wherever they are
// calculated: their errors, a task's output, and the checks made before a
// run. A port of the shared parts of vmaf_app/core/perceptual_cpu.py.
#pragma once

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/ffmpeg_run.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml {

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
// 4K pair, which is days for a film. A failed GPU run does not fall back to
// the CPU by itself past this length.
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
// CropDetectError and Cancelled.
std::pair<std::optional<CropBox>, std::optional<CropBox>> resolve_crops(
    const std::string& ffmpeg, const VideoInfo& source, const VideoInfo& distorted, CropMode crop_mode,
    const CancelToken& cancel, const StatusCallback& on_status);

}  // namespace vml
