// Video Compare's playback with FFmpeg: the commands that decode one video
// of a pair into frames for the window, the sizes they are decoded at, the
// test videos decoded beside the one shown, and the source's soundtrack. A
// port of vmaf_app/core/video_playback.py.
//
// FFmpeg decodes each video on its own (one process each), on the GPU's
// decoder where it can, and processes the pictures on the GPU (libplacebo:
// crop, scale, tone mapping to SDR) into 8-bit RGBA, which it writes to a
// pipe; the window pairs the two videos' frames by number.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vml/frame_extract.h"
#include "vml/geometry.h"
#include "vml/hardware_decoding.h"

namespace vml {

// How many test videos Video Compare keeps decoding, by default: the one
// shown and its two neighbours, so that the arrows switch to one already
// running (Settings: compare_decoded_videos).
inline constexpr int default_compare_decoded_videos = 3;

// The source as it is played: a comparison of the source with itself, at its
// own size, or with `native` false, scaled down (never up) to fit the test
// video's, its shape kept. Crops come off first.
FrameComparison source_playback_comparison(const FrameComparison& comparison, bool native = true);

// The `width` test videos to keep decoding, the one shown (`selected`)
// first, then its neighbours in the order the arrows reach them, around the
// ends: the next, the previous, the one after next, and so on. Never more
// than there are; nothing for a selection out of range.
std::vector<int> neighbour_indices(int count, int selected, int width = default_compare_decoded_videos);

// The largest even size the comparison's frames are decoded at within
// `maximum` (the view's size in device pixels), never scaled up; its own
// size, made even, without a maximum. Throws std::invalid_argument for a
// comparison with no size.
Size playback_dimensions(const FrameComparison& comparison, std::optional<Size> maximum = std::nullopt);

// How FFmpeg decodes and processes one video's frames, from the fastest:
// NVIDIA's decoder handing its pictures to the GPU processing on the GPU
// ("interop"), a hardware decoder's pictures through system memory
// ("transfer"), the CPU's decoder with the GPU processing ("software"), and
// the CPU for both ("cpu").
enum class PlaybackProcessing { interop, transfer, software, cpu };

// The order in which they are tried for a video the plan decodes with
// `hwaccel` (nothing: the CPU's decoder).
std::vector<PlaybackProcessing> playback_modes(const std::optional<std::string>& hwaccel);

// The command decoding `side` of `comparison` from `start_frame` on, at
// playback_dimensions(comparison, maximum), into 8-bit RGBA on standard
// output, a frame at a time at the comparison's rate. `plan` says which
// hardware decoder each side uses.
std::vector<std::string> build_playback_command(const std::string& ffmpeg, const FrameComparison& comparison,
                                                FrameSide side, long long start_frame,
                                                const PreviewColorSettings& colors, const HardwareDecoding& plan,
                                                std::optional<Size> maximum, PlaybackProcessing processing);

// The ffplay beside `ffmpeg`, or nothing if there is none.
std::optional<std::filesystem::path> ffplay_beside(const std::string& ffmpeg);

// The command playing the source's soundtrack from `start_frame` with
// `ffplay` (it has no picture): the comparison's one soundtrack, whichever
// test video is shown. Nothing without a frame rate.
std::optional<std::vector<std::string>> build_audio_command(const std::filesystem::path& ffplay,
                                                            const FrameComparison& comparison,
                                                            long long start_frame);

}  // namespace vml
