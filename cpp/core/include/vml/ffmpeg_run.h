// Comparing two videos: with FFmpeg's own metric filters, libvmaf (VMAF
// v0.6.1, NEG and v1, PSNR, SSIM) and xpsnr, on the CPU; or with VMAF, NEG
// and VMAF v1 scored on the GPU by the Vulkan engine (gpu_vmaf.h,
// vmaf_v1_gpu.h), from frames the app decodes itself (gpu_frames.h) or
// FFmpeg's, and FFmpeg's filters taking over if that fails. A port of
// vmaf_app/core/vmaf_runner.py: the same checks, the same filter graphs and
// commands, the same fallback from GPU decoding, and the same reading of
// the logs.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/hardware_decoding.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml {

// A run failed. stderr_tail() is the end of FFmpeg's error output, when
// there is one.
class VmafRunError : public std::runtime_error {
public:
    explicit VmafRunError(const std::string& message, std::string stderr_tail = "")
        : std::runtime_error(message), stderr_tail_(std::move(stderr_tail)) {}
    const std::string& stderr_tail() const { return stderr_tail_; }

private:
    std::string stderr_tail_;
};

// Called with (frames done, frames expected, frames per second).
using ProgressCallback = std::function<void(long long, long long, double)>;

// What a status tells the run line, beside its text (the Python app's
// status kinds).
enum class StatusKind {
    step,             // a step of the run
    starting,         // FFmpeg is starting: nothing to show as a step
    gpu_pass,         // one of several GPU metric passes begins
    gpu_wait,         // waiting for another video's GPU pass to end
    // VMAF on the GPU failed, or its check did (MetricTask::fallbacks); the
    // CPU takes over.
    gpu_vmaf_failed,
    // Perceptual metrics (SSIMULACRA2, Butteraugli, CVVDP) failed on the GPU,
    // or Vship's check did: stock Vship 5.1.2 or the CPU takes over within
    // the same task.
    gpu_perceptual_failed,
};

// Whether a status says that a GPU route failed (or its check did) and its
// metrics went elsewhere: to the CPU, or to stock Vship 5.1.2. Each such is a
// note of the video's (fallback_note), in the command line's output and on
// the window's row.
inline bool is_gpu_fallback(StatusKind kind) {
    return kind == StatusKind::gpu_vmaf_failed || kind == StatusKind::gpu_perceptual_failed;
}

// A fallback's status text as a note: without the "…" it ends with, as the
// step is over by the time the note is read.
std::string fallback_note(const std::string& status_text);

// A run's status, as the Python app words it.
struct RunStatus {
    std::string text;
    StatusKind kind = StatusKind::step;
    // The text without the decode plan, which the run line shows apart;
    // empty when the text has none.
    std::string brief{};
    // The decode plan the text names, for the window's "Decoder: ..."
    // (the Python app's Status.decode).
    std::optional<HardwareDecoding> decoding{};
};
using StatusCallback = std::function<void(const RunStatus&)>;

// The status of a step that decodes: `brief`, then the decode plan, then
// `ending`, as in "Running ffmpeg (GPU decode: source cuda, distorted
// cpu)...".
RunStatus decoding_status(const std::string& brief, const HardwareDecoding& decoding,
                          StatusKind kind = StatusKind::step, std::string_view ending = "...");

// A status as a process of its own sends it to the run (isolated.h), and
// back.
nlohmann::json status_json(const RunStatus& status);
RunStatus status_from_json(const nlohmann::json& json);

// The status a run sends when VMAF on the GPU fails, after which FFmpeg's
// libvmaf calculates it, starts with this.
inline constexpr std::string_view vmaf_gpu_failed = "VMAF on the GPU failed";

// What a run needs to know about this PC, found once.
struct RunEnvironment {
    std::string ffmpeg;          // the program
    std::string ffmpeg_version;  // "9.0.1", or "unknown"
    std::string ffprobe;         // the program, which reads a stream's configuration for the app's decoders
    DecodingCapabilities decoding;
    unsigned cores = 1;          // libvmaf's threads for "Auto"
};

// Throws VmafRunError when the two videos' timelines do not allow a
// comparison (geometry.h's pair_problem).
void validate_video_pair(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options);

// Throws VmafRunError when the two pictures are different shapes after
// cropping: scaling one to the other would stretch it, and the score would
// look like any other while meaning nothing.
void validate_display_geometry(const VideoInfo& source, const VideoInfo& distorted,
                               const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop);

// How many frames FFmpeg will pass (it stops at the shorter video and at the
// limit), which sizes the progress and is the result's compared frames.
long long estimate_total_frames(const VideoInfo& reference, const VmafOptions& options, const VideoInfo* other);

// Why `compared` frames are too few to stand for the video (one of the two
// was cut short), or nothing.
std::optional<std::string> short_comparison(long long expected, long long compared, double fps, int step);

// Whether the app may score PSNR, SSIM and XPSNR itself: its libvmaf is
// bundled, and the environment variable VML_CPU_METRICS does not send them
// to FFmpeg ("ffmpeg").
bool cpu_metrics_in_app();

// The filter graph of a comparison: each video downloaded from its GPU
// decoder if need be, cropped, converted and scaled to the compared size,
// its frames numbered, then libvmaf and xpsnr. With XPSNR beside libvmaf's
// metrics there are two graphs, one per line, each run on a thread of its
// own. `model` is the model= value (by its bare file name for a file);
// `cores` decides libvmaf's threads for "Auto".
std::string build_filtergraph(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              const HardwareDecoding& decoding, const std::string& log_name, const std::string& model,
                              const std::optional<std::string>& xpsnr_log_name, unsigned cores);

// FFmpeg's options before each video's -i (VideoInfo::input_options): a
// raw stream's frame rate.
struct InputOptions {
    std::vector<std::string> distorted;
    std::vector<std::string> source;
};

// FFmpeg's command for a filter graph, the test video first.
std::vector<std::string> build_ffmpeg_command(const std::string& ffmpeg, const std::filesystem::path& distorted,
                                              const std::filesystem::path& source, const std::string& filtergraph,
                                              const HardwareDecoding& decoding, double duration_limit,
                                              const InputOptions& inputs = {});

// The filter graph of a resolution test (VmafOptions::resample_test), which
// has a single input: the source is decoded once and split into an
// untouched reference and a test video, which is scaled down to the target
// width (keeping the source's shape) and back up to the source's own size.
// `hwaccel` is the source's decoder.
std::string build_resample_filtergraph(const VideoInfo& source, const VmafOptions& options,
                                       const std::optional<CropBox>& source_crop,
                                       const std::optional<std::string>& hwaccel, const std::string& log_name,
                                       const std::string& model, const std::optional<std::string>& xpsnr_log_name,
                                       unsigned cores);

// FFmpeg's command for a resolution test's filter graph; `source_options`
// go before the source's -i.
std::vector<std::string> build_resample_command(const std::string& ffmpeg, const std::filesystem::path& source,
                                                const std::string& filtergraph,
                                                const std::optional<std::string>& hwaccel, double duration_limit,
                                                const std::vector<std::string>& source_options = {});

// For VMAF on the GPU, the filter graph of each video's own FFmpeg (the test
// video's, then the source's): its chain as build_filtergraph's, giving its
// frames as [vmaf_dist] or [vmaf_ref], which the app pairs itself.
std::pair<std::string, std::string> build_stream_graphs(const VideoInfo& source, const VideoInfo& distorted,
                                                        const VmafOptions& options,
                                                        const std::optional<CropBox>& source_crop,
                                                        const std::optional<CropBox>& distorted_crop,
                                                        const HardwareDecoding& decoding);

// The two FFmpeg commands of VMAF on the GPU, the test video's first: each
// decodes its video, runs its graph (build_stream_graphs) and writes it to
// its outputs (GpuAttempt::output_args).
std::vector<std::vector<std::string>> build_stream_commands(
    const std::string& ffmpeg, const std::filesystem::path& distorted, const std::filesystem::path& source,
    const std::pair<std::string, std::string>& graphs, const HardwareDecoding& decoding,
    const std::pair<std::vector<std::string>, std::vector<std::string>>& outputs, const InputOptions& inputs = {});

// The per-frame scores in libvmaf's JSON log, and XPSNR's from the xpsnr
// filter's stats file when there is one, timed at `fps`.
FrameScores parse_libvmaf_log(const std::filesystem::path& log, double fps,
                              const std::optional<std::filesystem::path>& xpsnr_log);

// The child's side of a run's scoring in a process of its own (isolated.h):
// VMAF, NEG and VMAF v1 on the GPU, or PSNR, SSIM and XPSNR in the app, as
// the plan in `request` says. Scores the run `request` describes, sending
// its progress and statuses, leaves the scores in a file in `folder`, and
// returns what the parent needs to read them.
nlohmann::json scoring_task(const nlohmann::json& request, const std::filesystem::path& folder);

// Compares two videos and returns the result, with every requested metric's
// scores: VMAF, NEG and VMAF v1 on the GPU when they are the run's only
// metrics and the GPU can score them (scores_on_gpu); PSNR, SSIM and XPSNR
// in the app when they are its only metrics; with FFmpeg's metric filters
// otherwise or when that fails. The GPU's and the app's parts run in a process of their
// own where a program to run them is set (isolated.h). A video whose
// picture size changes partway through is scored at one size: the run is
// made again with it marked when FFmpeg rebuilds its filters
// (size_changes.h). Throws VmafRunError when the run fails, and Cancelled.
// `result_distorted_path` is the path the result carries for the test video
// (by default, its own).
ComparisonResult run_vmaf(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                          VmafOptions options, const ProgressCallback& on_progress, const StatusCallback& on_status,
                          const CancelToken& cancel,
                          const std::optional<std::filesystem::path>& result_distorted_path = std::nullopt);

// A made-up path that stands for a resolution test's test video, which has
// no file of its own (the Python app's synthetic_resample_distorted_path):
// the target is in the name, so that two targets of one source are not
// taken for the same comparison.
std::filesystem::path synthetic_resample_distorted_path(const std::filesystem::path& source,
                                                        const ResampleTarget& target);

// Runs a resolution test (VmafOptions::resample_test): the source scaled
// down to the target width and back up to its own size, compared with the
// untouched source, with FFmpeg's metric filters. It needs only the source.
// Throws VmafRunError when the run fails, and Cancelled.
ComparisonResult run_resample_test(const RunEnvironment& environment, const VideoInfo& source, VmafOptions options,
                                   const ProgressCallback& on_progress, const StatusCallback& on_status,
                                   const CancelToken& cancel);

}  // namespace vml
