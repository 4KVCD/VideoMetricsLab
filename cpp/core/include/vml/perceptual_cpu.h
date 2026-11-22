// SSIMULACRA2 and Butteraugli calculated on the CPU by the app's own
// engines, with the scores of libjxl 0.12's command-line tools to the last
// bit.
//
// These metrics compare still images, not videos. Each frame pair of a
// comparison is made as lossless 16-bit RGB pictures and scored as soon as
// it is made: two FFmpegs, one per video, write the pictures into pipes of
// the app's (rgb_stream_commands), and the engines (cpu_pair_scorer.h) score
// them in memory, in a process of the app's own, a few pairs at a time with
// Butteraugli's work split between threads (in_app_schedule).
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vml/cancel.h"
#include "vml/ffmpeg_run.h"
#include "vml/perceptual.h"

namespace vml {

// The display brightness Butteraugli assumes for SDR pictures, in nits. It
// is Vship's default, which the GPU is given too; the tool's own is 80. An
// HDR picture's brightness is its own, because PQ and HLG are absolute, so
// for HDR it is taken from the picture (butteraugli_intensity_of), as Vship
// does. Its scores
// then agree with Vship's: 2.892 against 2.881 on a PQ film, where 203 nits
// gave 0.815.
inline constexpr double butteraugli_intensity_nits = 203.0;

// How the app's engines (cpu_pair_scorer.h) share the CPU: how many frame
// pairs are scored at a time, and how many threads each pair's Butteraugli
// takes.
struct InAppSchedule {
    int workers = 1;
    std::size_t butteraugli_threads = 1;
};

// The schedule for a task of `metrics` on pairs of `width` x `height`,
// given the logical `cores` shared with the other tasks the CPU's queue runs
// at the same time (`concurrent_tasks`): with both metrics, a sixth of the
// task's cores pairs at a time, each Butteraugli on the rest of its share
// less one core for its SSIMULACRA2; Butteraugli alone, a twelfth, on its
// share; SSIMULACRA2 alone, half, on one thread each. Measured on 24 cores
// (8 fast, 16 slow), 4K pairs: Butteraugli alone 23.7 s for 2 pairs x 12
// threads, 25.3 s for 4 x 6, 29.4 s for 1 x 24 (the threads of one picture
// waiting on each other at the end of each band of rows); both metrics
// 27.7 s for 4 pairs, 36.7 s for 2 (one SSIMULACRA2 thread a pair).
// Workers are also no more than fit in 60% of `free_memory`, at a worker's
// memory (its pair, Butteraugli::memory_bytes, SSIMULACRA2's rows).
InAppSchedule in_app_schedule(int width, int height, const std::vector<std::string>& metrics, int concurrent_tasks,
                              unsigned cores, std::optional<unsigned long long> free_memory);

// The score in a tool's printed output (as the engines' SSIMULACRA2 score
// is printed, to round it as the tool does): its last number. Throws
// PerceptualRunError when there is none.
double parse_score(std::string_view metric, std::string_view output);

// The 3-norm of a Butteraugli distortion map held top row first, in rows of
// `width` values (as the app's engine makes it): the cube root of the mean
// of the cubes, as Vship reports it, worked out as numpy works it out on the
// tool's --rawdistmap PFM file, in that file's order (its bottom row first;
// the order sets the sum's last bits), so that it is the tool's map's to the
// last bit. The tool prints its own "3-norm", the mean of the 3-, 6- and
// 12-norms, which reads about 1.8 times Vship's on the same frames.
double butteraugli_norm3_top_row_first(std::span<const float> distortion_map, std::size_t width);

// The frames a CPU SSIMULACRA2/Butteraugli run of `options` compares and
// decodes (frame_range.h), every `step`-th: these metrics look at no frame
// but the one they score.
FrameRange cpu_range(const VmafOptions& options, const VideoInfo& source, int step);

// The filter graph that makes two lossless RGB48 picture streams of the same
// size, [distorted] and [reference], from the test video (FFmpeg's first
// input) and the source (its second). Their Nth pictures are a pair: each
// video's frame N, as every metric pairs them (frame_sync.h), every
// `step`-th, those of the range (cpu_range: counted from its decode_start).
// Each video is converted to RGB with the matrix and range Vship
// reads it with, after any crop and scaling, so that the engines see what
// Vship sees.
std::string image_filtergraph(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              int step);

// The two FFmpeg commands that write the pictures of image_filtergraph's
// two chains into pipes, one video each, as raw rgb48le with nothing
// between the frames: the test video's into `test_pipe`, then the source's
// into `source_pipe`. Each runs to its own input's end or the range's. A
// video whose pictures the app feeds (grain_feed.h) has its feed's input
// (`distorted_fed`, `source_fed`) in its file's place.
std::vector<std::vector<std::string>> rgb_stream_commands(
    const std::string& ffmpeg, const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
    const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop, int step,
    const std::string& test_pipe, const std::string& source_pipe,
    const std::optional<std::vector<std::string>>& distorted_fed = {},
    const std::optional<std::vector<std::string>>& source_fed = {});

// Calculates `keys` (SSIMULACRA2 and Butteraugli) on the CPU and returns
// each frame's scores (the Python app's run_perceptual_task): with the black
// bars `crops` already found, or else found here. Throws Cancelled,
// ComparisonCutShortError (a video ends long before its stated length),
// PerceptualRunError and CropDetectError.
PerceptualTaskOutput run_cpu_perceptual_task(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                             const std::optional<CropPair>& crops,
                                             const ProgressCallback& on_progress, const StatusCallback& on_status,
                                             const CancelToken& cancel);

// The isolated "cpu-perceptual" task's work (isolated.cpp): a task scored
// in the app, in this process, its scores' numbers left in `folder`
// (perceptual_output_json); a video cut short as {"cut_short": reason}, and
// a PerceptualRunError as {"failed": message, "stderr_tail": tail}.
nlohmann::json cpu_perceptual_task(const nlohmann::json& request, const std::filesystem::path& folder);

}  // namespace vml
