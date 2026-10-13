// Comparing two videos with FFmpeg's own metric filters: libvmaf (VMAF
// v0.6.1, NEG and v1, PSNR, SSIM) and xpsnr, all on the CPU. A port of the
// CPU path of vmaf_app/core/vmaf_runner.py: the same checks, the same
// filter graph and command, the same fallback from GPU decoding, and the
// same reading of the logs.
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
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

// What a run needs to know about this PC, found once.
struct RunEnvironment {
    std::string ffmpeg;          // the program
    std::string ffmpeg_version;  // "9.0.1", or "unknown"
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

// FFmpeg's command for a filter graph, the test video first.
std::vector<std::string> build_ffmpeg_command(const std::string& ffmpeg, const std::filesystem::path& distorted,
                                              const std::filesystem::path& source, const std::string& filtergraph,
                                              const HardwareDecoding& decoding, double duration_limit);

// The per-frame scores in libvmaf's JSON log, and XPSNR's from the xpsnr
// filter's stats file when there is one, timed at `fps`.
FrameScores parse_libvmaf_log(const std::filesystem::path& log, double fps,
                              const std::optional<std::filesystem::path>& xpsnr_log);

// Compares two videos with FFmpeg's metric filters and returns the result,
// with every requested metric's scores. Throws VmafRunError when the run
// fails, and Cancelled. `result_distorted_path` is the path the result
// carries for the test video (by default, its own).
ComparisonResult run_vmaf(const RunEnvironment& environment, const VideoInfo& source, const VideoInfo& distorted,
                          VmafOptions options, const ProgressCallback& on_progress, const CancelToken& cancel,
                          const std::optional<std::filesystem::path>& result_distorted_path = std::nullopt);

}  // namespace vml
