// SSIMULACRA2, Butteraugli and CVVDP on the GPU, through Vship's C API
// (Vship 5.1: Vship_InitHandler and Vship_ComputeHandler): the library, the
// GPU it runs on, and how a video's frames are laid out and described for
// it. A port of the first parts of vmaf_app/core/perceptual_vship.py; the
// passes that score are vship_run.h's.
//
// One build is bundled, in tools/vship/vulkan, on Windows and macOS: the
// Vulkan build of Vship-fast (github.com/4KVCD/Vship-fast), a faster fork of
// Vship whose metrics compute as libjxl's and pycvvdp's do (its README's
// measurements), on any GPU with a Vulkan driver. Vship converts each frame
// from the colour space it is described in (vship_colorspace), and takes a
// frame as three planes.
#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/frame_sync.h"
#include "vml/perceptual.h"
#include "vml/platform/library.h"
#include "vml/video.h"

namespace vml {

// Vship could not be loaded or run, so the metrics can safely be calculated
// on the CPU instead.
class VshipUnavailableError : public PerceptualRunError {
public:
    using PerceptualRunError::PerceptualRunError;
};

// Every metric's GPU pass failed. The message is the first error's text, and
// failures() holds each metric's own reason, so that no metric is reported
// with another's (such as CVVDP given SSIMULACRA2's out-of-memory error).
class VshipPassesFailedError : public VshipUnavailableError {
public:
    VshipPassesFailedError(const std::string& message, PerceptualFailures failures)
        : VshipUnavailableError(message), failures_(std::move(failures)) {}
    const PerceptualFailures& failures() const { return failures_; }

private:
    PerceptualFailures failures_;
};

// Vship 5.1's C API's structures, as its VshipAPI.h lays them out (and the
// Python app's binding declares them).
namespace vship_api {

struct Subsampling {
    int subw = 0;
    int subh = 0;
};

struct Crop {
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
};

// Vship_ColorSpace_t: how Vship reads a frame's samples. The enums are
// H.273's numbers; -1 targets keep the frame's size.
struct Colorspace {
    std::int64_t width = 0;
    std::int64_t height = 0;
    std::int64_t target_width = -1;
    std::int64_t target_height = -1;
    int sample = 0;
    int range = 0;
    Subsampling subsampling;
    int chromaLocation = 0;
    int colorFamily = 0;
    int YUVMatrix = 0;
    int transferFunction = 0;
    int primaries = 0;
    Crop crop;
};

struct Version {
    int major = 0;
    int minor = 0;
    int minorMinor = 0;
    int backend = 0;
};

// What Vship reports about a GPU. GetDeviceInfo writes the trailing feature
// matrix (added in Vship 5.0) even for backends that do not use it.
struct DeviceInfo {
    char name[256] = {};
    std::uint64_t VRAMSize = 0;
    int integrated = 0;
    int MultiProcessorCount = 0;
    int WarpSize = 0;
    bool vulkanFeatureMatrix[26] = {};
};

// Vship_StructType: which init or score structure a call's argument is.
inline constexpr int init_ssimulacra2 = 1, init_butteraugli = 2, init_cvvdp = 3;
inline constexpr int score_ssimulacra2 = 4, score_butteraugli = 5, score_cvvdp = 6;

struct InitSsimulacra2 {
    int structType = init_ssimulacra2;
    Colorspace src_colorspace;
    Colorspace dis_colorspace;
    int gpu_id = 0;
};

struct InitButteraugli {
    int structType = init_butteraugli;
    Colorspace src_colorspace;
    Colorspace dis_colorspace;
    int Qnorm = 2;
    float intensity_multiplier = 203.0f;
    int gpu_id = 0;
};

struct ScoreSsimulacra2 {
    int structType = score_ssimulacra2;
    double score = 0.0;
};

// The display is JSON text, with the key it is defined under (cvvdp.h's
// vship_display_json); the frame rate is part of the score, as CVVDP models
// how the eye integrates over time.
struct InitCvvdp {
    int structType = init_cvvdp;
    Colorspace src_colorspace;
    Colorspace dis_colorspace;
    float fps = 0.0f;
    bool resizeToDisplay = false;
    const char* model_key_cstr = nullptr;
    const char* model_config_json_cstr = nullptr;
    int gpu_id = 0;
};

// dstp nullptr: no distortion map.
struct ScoreButteraugli {
    int structType = score_butteraugli;
    double normQ = 0.0;
    double norm3 = 0.0;
    double norminf = 0.0;
    void* dstp = nullptr;
    std::int64_t dststride = 0;
};

// The JOD of the frames since the handler's score was last reset. dstp
// nullptr: no distortion map.
struct ScoreCvvdp {
    int structType = score_cvvdp;
    double score = 0.0;
    void* dstp = nullptr;
    std::int64_t dststride = 0;
};

}  // namespace vship_api

// A loaded Vship build and the functions of it the app calls: Vship 5.1's
// API. A library without it (4.x) is refused rather than used through old
// entry points.
struct VshipLibrary {
    // Throws VshipUnavailableError for a library older than 5.1.
    explicit VshipLibrary(platform::DynamicLibrary loaded);

    platform::DynamicLibrary library;
    vship_api::Version (*get_version)();
    int (*get_device_count)(int* count);
    int (*get_device_info)(vship_api::DeviceInfo* info, int gpu_id);
    int (*gpu_full_check)(int gpu_id);
    int (*get_error_message)(int code, char* out, int bytes);
    int (*get_detailed_last_error)(char* out, int bytes);
    // Pinned (page-locked) memory, allocated per GPU (Vulkan allocates per
    // device).
    int (*pinned_malloc)(void** address, std::uint64_t bytes, int gpu_id);
    int (*pinned_free)(void* address, int gpu_id);
    // One handler per metric, made from an init structure and scored into a
    // score structure: both start with their Vship_StructType.
    int (*init_handler)(void** handler, void* init);
    int (*free_handler)(void* handler);
    int (*compute_handler)(void* handler, void* score, const std::uint8_t* const* source_planes,
                           const std::uint8_t* const* distorted_planes, const std::int64_t* source_strides,
                           const std::int64_t* distorted_strides);
    int (*get_detailed_last_error_handler)(void* handler, char* out, int bytes);
    // Clears CVVDP's score accumulation only, not its temporal history.
    int (*reset_score)(void* handler);

    // Vship's text for the error `code`, or its last detailed error without
    // one.
    std::string message(std::optional<int> code) const;

    // Why a call on `handler` failed: the handler's own last error, or else
    // the text for `code`. The handler's own is used because other handlers,
    // on other threads, may have failed since.
    std::string handler_error(void* handler, int code) const;
};

// Where Vship's Vulkan build is: Vship-fast's, or with `vship51` stock Vship
// 5.1.2's, which the GPU metrics fall back to when Vship-fast fails
// (MetricRouting in vship_run.cpp). A package may bundle only Vship-fast.
std::filesystem::path vship_library_path(bool vship51 = false);

// Whether stock Vship 5.1.2 is bundled.
bool vship51_bundled();

// The Vulkan the app bundles on macOS (tools/vulkan): Khronos' loader, with
// MoltenVK as its driver, which Vship's and the Vulkan VMAF's builds there are
// linked to. The first call points the loader at MoltenVK (VK_DRIVER_FILES).
std::filesystem::path vulkan_library_path();

// The GPU Vship runs on.
struct VshipDevice {
    std::string backend;  // the build it runs on, "vulkan", as scores record it
    std::string name;
    int gpu_id = 0;
    std::string version;  // "5.1.2"
    // Who made the GPU, as the Vulkan loader says (GpuVendor::none for
    // another maker). Nothing when it could not be told.
    std::optional<GpuVendor> vendor;
    // Found by stock Vship 5.1.2's probe, where Vship-fast could use no GPU:
    // the metrics run on 5.1.2.
    bool vship51 = false;
    // Why Vship-fast could use no GPU, where `vship51`: for the run's note
    // that its metrics fell back to 5.1.2. Kept in the process that probed,
    // not in vship_device_json.
    std::string fast_failure{};
};

// The build `device` was found by, as the app shows it: "Vship-fast 5.1.2"
// or "Vship 5.1.2".
std::string vship_build_name(const VshipDevice& device);

// Vship's library, Vship-fast's or with `vship51` stock Vship 5.1.2's,
// loaded once in this process: a process loads only one of the two (the
// fallback runs in a process of its own). Throws VshipUnavailableError.
const VshipLibrary& vship_library(bool vship51 = false);

// Why `device` does not calculate `key` on frames of `size`, or nothing if
// it does: Vship 5.1.2's Vulkan CVVDP hung Intel's GPU above 1920x1080, and
// Vship-fast's has not been tried there (a hang resets the GPU).
// `computed` is the size the metric is calculated at, where that differs
// (CVVDP scaled to the display).
std::optional<std::string> size_refusal(const VshipDevice& device, const std::string& key, std::pair<int, int> size,
                                        std::optional<std::pair<int, int>> computed = std::nullopt);

// Loads Vship's build in this process, Vship-fast's or with `vship51` stock
// Vship 5.1.2's, and returns the GPU it would use, a discrete one first, or
// the reasons none was found.
std::pair<std::optional<VshipDevice>, std::string> probe_vship_device(bool vship51 = false);

// The probe's answer as the isolated "vship-probe" task sends it.
nlohmann::json vship_probe_result(const std::pair<std::optional<VshipDevice>, std::string>& probed);

// A device as text for a process of its own, and back (null is none).
nlohmann::json vship_device_json(const VshipDevice& device);
std::optional<VshipDevice> vship_device_from(const nlohmann::json& text);

// The GPU for Vship, checked once in a process of its own where a program to
// run it is set (loading Vship starts the GPU driver, and a crash there
// would end the run), and kept; or a reason the user can read for why there
// is none. Where Vship-fast can use none (it compiles every shader as it
// starts on a GPU), stock Vship 5.1.2 is tried in a process of its own too.
std::pair<std::optional<VshipDevice>, std::string> detect_vship_device();
// Whether detect_vship_device's answer of no GPU came from a check that
// crashed or failed with an error, rather than one that found no GPU Vship
// can use: the metrics then fall back to the CPU, and the run says so.
bool vship_check_failed();

// What the last probe to finish found, at once: never waiting for a probe in
// progress, which takes seconds (one starts with the app, and a run probes
// again after a failed one). Nothing before the first has finished. For the
// window's thread, which must not stop for a probe.
std::optional<std::pair<std::optional<VshipDevice>, std::string>> known_vship_device();
// Whether detect_vship_device would answer now, without probing or waiting
// for a probe.
bool vship_probe_ready();
// Calls `callback` each time a probe finishes, on the thread that ran it,
// with known_vship_device up to date; nothing for none. It is not called
// once this returns, so that its owner can go: it must be quick.
void on_vship_probe_finished(std::function<void()> callback);

// Forgets a probe that found no GPU a minute or more ago, so that the next
// run probes again: the failure may have been temporary (a driver being
// updated or restarted, a GPU briefly unavailable), and without this every
// GPU metric of a long run would go to the CPU.
void forget_failed_vship_probe();

// The planar layout FFmpeg sends a video's frames in (`pixel_format`), and
// how Vship reads it. `family` is 1 for RGB, whose planes are all full size
// (Vship itself tells RGB from YUV by the matrix).
struct VshipImageFormat {
    std::string pixel_format;
    int family = 0;
    int sample = 0;  // Vship_Sample_t
    int subw = 0;
    int subh = 0;
    bool full_range = false;  // yuvj: full range when the stream has no range tag
    // FFmpeg's planar RGB is ordered G, B, R; Vship reads R, G, B.
    std::array<int, 3> plane_order{0, 1, 2};

    // A frame's layout at this size.
    struct Layout {
        std::size_t frame_bytes = 0;
        std::array<std::size_t, 3> plane_bytes{};
        std::array<std::int64_t, 3> strides{};
    };
    Layout frame_layout(int width, int height) const;

    // 2 bytes a sample above 8 bits.
    bool wide() const;
};

// The layout FFmpeg sends a decoded video in, and how Vship reads it: a
// semi-planar or big-endian layout as the planar little-endian one, with
// alpha dropped and neutral chroma added to a monochrome video. Throws
// VshipUnavailableError for a layout it cannot read.
VshipImageFormat vship_image_format(const VideoInfo& info);

// The planar layout Vship is given and the layout FFmpeg sends, when a
// hardware-decoded 4:2:0 frame can cross the pipe exactly as it is
// downloaded (NV12, or limited-range P010 declared as 16-bit, which is the
// same picture to Vship); otherwise nothing. Only the chroma is then split
// into planes here.
std::optional<std::pair<VshipImageFormat, std::string>> passthrough_format(const VideoInfo& info,
                                                                           const std::optional<std::string>& hwaccel);

// The frame's colour space for Vship, from the stream's tags, read as
// colour.h's video_colour reads them. Throws VshipUnavailableError for tags
// Vship does not take.
vship_api::Colorspace vship_colorspace(const VideoInfo& info, const VshipImageFormat& image, int width, int height);

// The chroma siting each of two videos is read with: its own, except that a
// video that gives none (no tag, "unspecified" or "unknown") takes the other
// video's, so that two videos are not read half a pixel apart for want of a
// tag. FFmpeg writes "center" into a y4m file made from a video that gives
// no siting: such a reference, against a test video that gives none, scored
// SSIMULACRA2 -44.09 where the video it was made from scored -42.62, and
// Butteraugli 11.20 where it scored 10.48 (a 720p test clip, on Vulkan).
std::pair<std::string, std::string> shared_chroma_location(const std::string& source, const std::string& distorted);

// Whether shared_chroma_location changes what Vship reads of a pair: one
// video gives no siting, which Vship reads as left, and the other gives
// another.
bool chroma_location_shared(const std::string& source, const std::string& distorted);

// The sizes the source's and the test video's frames are scaled to: both
// the same, by the comparison's scaling direction.
std::pair<std::pair<int, int>, std::pair<int, int>> vship_scaled_sizes(const VideoInfo& source,
                                                                       const VideoInfo& distorted,
                                                                       ScaleDirection direction,
                                                                       const std::optional<CropBox>& source_crop,
                                                                       const std::optional<CropBox>& distorted_crop);

// The filters a video's frames go through on their way to Vship: numbered
// at the test video's frame rate `numbered_at` (frame_sync.h; nothing when
// pairing by timestamp, where each video's times start at 0 instead), every
// step-th kept, brought from the GPU (`hwaccel`), cropped, scaled and
// converted to `pixel_format`, in the full range where Vship is told the
// video is (`full_range`, vship_colorspace).
std::string vship_filter_chain(const VideoInfo& info, const std::optional<CropBox>& crop,
                               std::pair<int, int> target_size, const std::string& pixel_format, int step,
                               const std::string& algorithm, const std::optional<std::string>& hwaccel,
                               const std::optional<Rational>& numbered_at, bool full_range);

// FFmpeg's output arguments for sending frames of `pixel_format` down its
// standard output: YUV4MPEG where the muxer takes the layout (FFmpeg writes
// it straight from its frames), else raw video.
std::vector<std::string> vship_piped_as(const std::string& pixel_format);

}  // namespace vml
