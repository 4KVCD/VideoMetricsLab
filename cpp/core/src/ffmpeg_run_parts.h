// What ffmpeg_run.cpp shares with the two files its work is split into:
// ffmpeg_commands.cpp, which makes FFmpeg's commands and filter graphs, and
// score_logs.cpp, which reads the scores FFmpeg's libvmaf and xpsnr write.
// Private to vml_core: in a namespace of its own, so that no name here meets
// another file's own helper of the same name.
#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vml/frame_range.h"
#include "vml/options.h"

namespace vml::run_parts {

// ---- ffmpeg_commands.cpp

// Whether the run calculates a metric of a VMAF model (VMAF, VMAF NEG, VMAF v1).
bool uses_vmaf_model(const VmafOptions& options);

// The VMAF v1 model file a run uses, once the run has resolved it.
std::optional<std::filesystem::path> v1_model_file(const VmafOptions& options);

// ---- score_logs.cpp

// One run's per-frame scores, as the Python app's FrameScores keeps them
// (32-bit floats; NaN where a frame has none), by metric key.
struct ParsedScores {
    std::vector<int> frame;
    std::vector<double> time;
    std::map<std::string, std::vector<float>> metrics;
    std::vector<std::string> order;  // the keys in the Python app's order

    bool has(const std::string& key) const { return metrics.contains(key); }
};

// The times of frames numbered `frames`, at `fps` (0 where it is unknown).
std::vector<double> times_of(const std::vector<int>& frames, double fps);

// A ranged run's scores as a run over the whole videos gives them: the
// frames it numbered from 0 at range.decode_start get their numbers in the
// whole videos, and only those in range are kept (those before them were
// decoded for the metrics that look back, the one after for VMAF's look
// ahead), timed at `fps`.
void keep_range(ParsedScores& scores, const FrameRange& range, double fps);

// The columns of the scores read, in the Python app's order.
enum Column { vmaf_column, psnr_column, ssim_column, xpsnr_column, neg_column, v1_column, column_count };
inline constexpr std::array<const char*, column_count> column_keys{
    "vmaf", "psnr", "ssim", "xpsnr", "vmaf_neg", "vmaf_v1"};

// XPSNR's luma score per frame from the xpsnr filter's stats file, by frame
// number counted from 0 (the file counts from 1).
std::map<int, double> parse_xpsnr_log(const std::filesystem::path& path);

// The scores in libvmaf's JSON log, with XPSNR's from its stats file.
ParsedScores parse_log(const std::filesystem::path& log_path, double fps,
                       const std::optional<std::filesystem::path>& xpsnr_log_path);

}  // namespace vml::run_parts
