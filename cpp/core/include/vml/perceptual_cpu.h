// SSIMULACRA2 and Butteraugli calculated on the CPU with libjxl's
// command-line tools, as the Python app calculates them. A port of
// vmaf_app/core/perceptual_cpu.py.
//
// The tools compare still images, not videos. One FFmpeg writes each frame
// pair of a comparison as lossless 16-bit RGB PNG files, into a folder that
// exists only while the task runs, and each pair is scored as soon as it is
// written, several pairs at a time (scoring_workers), then deleted. FFmpeg
// is held a few pairs ahead of the scoring: it is suspended while that many
// wait, so that it does not write a whole film out first.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/cancel.h"
#include "vml/ffmpeg_run.h"
#include "vml/perceptual.h"

namespace vml {

// The program for `metric` ("ssimulacra2" or "butteraugli"): the one the
// environment variable SSIMULACRA2_PATH or BUTTERAUGLI_PATH names (nothing
// when that does not exist), else the bundled one (libjxl's official static
// tools, in tools/libjxl), else one on the PATH; nothing when there is none.
// The variables keep this choice out of Settings, and let tests use fake
// tools. Throws std::invalid_argument for another metric.
std::optional<std::filesystem::path> find_metric_executable(std::string_view metric);

// The tool's version, for the scores' provenance: the first line it prints
// for --version, -version or -V, or, for a bundled tool that prints none
// (libjxl's do not), the first line of the bundle's LIBJXL-VERSION.txt;
// else "unknown".
std::string implementation_version(const std::filesystem::path& executable);

// The display brightness Butteraugli assumes for SDR pictures, in nits. It
// is Vship's default, which the GPU is given too; the tool's own is 80. An
// HDR picture's brightness is its own, because PQ and HLG are absolute, so
// for HDR the tool takes it from the picture, as Vship does. Its scores
// then agree with Vship's: 2.892 against 2.881 on a PQ film, where 203 nits
// gave 0.815.
inline constexpr double butteraugli_intensity_nits = 203.0;

// How many frame pairs are scored at a time, at least 1.
//
// The tools score one still image on one core: 1.6 s (SSIMULACRA2) and
// 2.7 s (Butteraugli) for a 4K pair. One pair at a time, a 10-second 4K
// clip took 13 minutes with 1 core of 24 busy. Each pair's score is its
// own, so scoring several at once changes how soon the scores arrive, never
// what they are.
//
// The number is half the logical `cores`, shared with the other tasks the
// CPU's queue runs at the same time (`concurrent_tasks`, as libvmaf's
// threads are shared). On 24 cores (8 fast, 16 slow), 12 at once scored
// 6.5 times as many pairs a second as 1, and 24 only 7.4 times as many, for
// twice the memory: the cores share the memory's speed, and FFmpeg needs
// some of them to write the pairs. It is also no more than fits in 60% of
// `free_memory` (none: the cores alone decide), at the peak memory of the
// larger of `metrics`' tools for a pair of `pixels`, because a worker runs
// its tools one after the other.
int scoring_workers(long long pixels, const std::vector<std::string>& metrics, int concurrent_tasks, unsigned cores,
                    std::optional<unsigned long long> free_memory);

// The score a tool printed: the last number in its output. Throws
// PerceptualRunError when there is none.
double parse_score(std::string_view metric, std::string_view output);

// The 3-norm of a Butteraugli distortion map (the tool's --rawdistmap, a
// greyscale PFM file): the cube root of the mean of the cubes, as Vship
// reports it, worked out as numpy works it out in the Python app. The tool
// prints its own "3-norm", the mean of the 3-, 6- and 12-norms, which reads
// about 1.8 times Vship's on the same frames. Throws PerceptualRunError for
// a map that is not greyscale, and std::runtime_error for a file that is
// not a map at all.
double butteraugli_norm3(const std::filesystem::path& distortion_map);

// The filter graph that makes two lossless RGB48 picture streams of the same
// size, [distorted] and [reference], from the test video (FFmpeg's first
// input) and the source (its second). Their Nth pictures are a pair: each
// video's frame N, as every metric pairs them (frame_sync.h), every
// `step`-th; or, when frames are paired by timestamp, the test video's Nth
// frame and the source frame FFmpeg's frame sync pairs with it. Each video
// is converted to RGB with the matrix and range Vship reads it with, after
// any crop and scaling, so that the tools see what Vship sees.
std::string image_filtergraph(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              int step);

// The FFmpeg command that writes the pictures of image_filtergraph into
// `folder` as test-%08d.png and reference-%08d.png, each output to its own
// input's end or the duration limit, each picture under a temporary name
// until it is whole (-atomic_writing).
std::vector<std::string> png_pairs_command(const std::string& ffmpeg, const VideoInfo& source,
                                           const VideoInfo& distorted, const VmafOptions& options,
                                           const std::optional<CropBox>& source_crop,
                                           const std::optional<CropBox>& distorted_crop, int step,
                                           const std::filesystem::path& folder);

// Calculates `keys` (SSIMULACRA2 and Butteraugli) on the CPU and returns
// each frame's scores (the Python app's run_perceptual_task): with the black
// bars `crops` already found, or else found here. Throws Cancelled,
// ComparisonCutShortError (a video ends long before its stated length),
// PerceptualRunError and CropDetectError.
PerceptualTaskOutput run_cpu_perceptual_task(const PerceptualRequest& request, const std::vector<std::string>& keys,
                                             const std::optional<CropPair>& crops,
                                             const ProgressCallback& on_progress, const StatusCallback& on_status,
                                             const CancelToken& cancel);

}  // namespace vml
