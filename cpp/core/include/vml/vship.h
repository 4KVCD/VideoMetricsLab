// SSIMULACRA2, Butteraugli and CVVDP on the GPU, through Vship's C API
// (Vship 5.1: Vship_InitHandler and Vship_ComputeHandler): the library, the
// GPU it runs on, and how a video's frames are laid out and described for
// it. A port of the first parts of vmaf_app/core/perceptual_vship.py; the
// passes that score are vship_run.h's.
//
// Vship's builds are bundled in tools/vship/<folder>, for Windows only:
// CUDA runs on NVIDIA, HIP on AMD, and Vulkan on any GPU with a Vulkan
// driver. Vship converts each frame from the colour space it is described
// in (vship_colorspace), and takes a frame as three planes.
#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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

// dstp nullptr: no distortion map.
struct ScoreButteraugli {
    int structType = score_butteraugli;
    double normQ = 0.0;
    double norm3 = 0.0;
    double norminf = 0.0;
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

// Where Vship's `backend` build is ("vulkan", "cuda" or "hip").
std::filesystem::path vship_library_path(const std::string& backend);

// The GPU makers each build runs on: CUDA on NVIDIA, HIP on AMD; Vulkan on
// any (the Vulkan loader says whose a GPU is).
struct VshipDevice {
    std::string backend;  // "vulkan", "cuda" or "hip": the build it runs on
    std::string name;
    int gpu_id = 0;
    std::string version;  // "5.1.2"
    // Who made the GPU: NVIDIA for CUDA, AMD for HIP, and for Vulkan what
    // the Vulkan loader says (GpuVendor::none for another maker). Nothing
    // when it could not be told.
    std::optional<GpuVendor> vendor;
};

// The library `device` runs on, loaded once in this process. Throws
// VshipUnavailableError.
const VshipLibrary& vship_library(const VshipDevice& device);

// Settings > GPU metrics > GPU backend: "auto", or the build to try first.
// An unknown name means "auto". A probe made for another choice is
// forgotten.
void set_vship_backend(const std::string& backend);
std::string vship_backend();

// The name shown for a build: "Vulkan", "CUDA" or "HIP".
std::string backend_label(const std::string& backend);

// Whether `device`'s Vship build scores `key` correctly on its GPU. The
// Python app keeps a list of the (build, maker, metric) a bundled build
// scores wrongly; it is empty since Vship 5.1.2.
bool scores_correctly(const VshipDevice& device, const std::string& key);

// Why `device` does not calculate `key` on frames of `size`, or nothing if
// it does: Vship 5.1.2's Vulkan CVVDP hangs Intel's GPU above 1920x1080.
// `computed` is the size the metric is calculated at, where that differs
// (CVVDP scaled to the display).
std::optional<std::string> size_refusal(const VshipDevice& device, const std::string& key, std::pair<int, int> size,
                                        std::optional<std::pair<int, int>> computed = std::nullopt);

// Tries the builds in turn for the backend setting `choice` ("auto": CUDA,
// HIP, then Vulkan; a chosen build first, then the others), in this
// process, and returns the first usable GPU found, a discrete one first, or
// the reasons none was found.
std::pair<std::optional<VshipDevice>, std::string> probe_vship_device(const std::string& choice);

// The probe's answer as the isolated "vship-probe" task sends it.
nlohmann::json vship_probe_result(const std::pair<std::optional<VshipDevice>, std::string>& probed);

// A device as text for a process of its own, and back (null is none).
nlohmann::json vship_device_json(const VshipDevice& device);
std::optional<VshipDevice> vship_device_from(const nlohmann::json& text);

// The GPU for Vship, checked once in a process of its own where a program to
// run it is set (loading Vship starts the GPU driver, and a crash there
// would end the run), and kept; or a reason the user can read for why there
// is none.
std::pair<std::optional<VshipDevice>, std::string> detect_vship_device();

// Whether SSIMULACRA2 or Butteraugli set to the GPU is calculated on the
// GPU here: when Vship can use a GPU, and its build scores the metric
// correctly.
bool gpu_can_score(const std::string& key);

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

// The layout FFmpeg sends a decoded video in, and how Vship `version`
// ("5.1.2") reads it: a semi-planar or big-endian layout as the planar
// little-endian one, with alpha dropped and neutral chroma added to a
// monochrome video; 4:1:0 as it is to a Vship that reads it, else upsampled
// to 4:4:4. Throws VshipUnavailableError for a layout it cannot read.
VshipImageFormat vship_image_format(const VideoInfo& info, const std::optional<std::string>& version = std::nullopt);

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
// converted to `pixel_format`.
std::string vship_filter_chain(const VideoInfo& info, const std::optional<CropBox>& crop,
                               std::pair<int, int> target_size, const std::string& pixel_format, int step,
                               const std::string& algorithm, const std::optional<std::string>& hwaccel,
                               const std::optional<Rational>& numbered_at);

// FFmpeg's output arguments for sending frames of `pixel_format` down its
// standard output: YUV4MPEG where the muxer takes the layout (FFmpeg writes
// it straight from its frames), else raw video.
std::vector<std::string> vship_piped_as(const std::string& pixel_format);

}  // namespace vml
