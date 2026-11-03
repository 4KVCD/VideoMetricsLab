// A comparison's settings: which of FFmpeg's metrics to calculate, and how
// (the Python app's VmafOptions, a row's settings in the window).
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/range_seek.h"
#include "vml/video.h"

namespace vml {

// The two special model choices: "Auto" (the 4K model for a 4K comparison)
// and a custom model file.
inline constexpr std::string_view auto_model_choice = "__auto__";
inline constexpr std::string_view custom_model_choice = "__custom__";

// The scaling algorithms a resolution test may have, as FFmpeg's scale
// filter names them.
inline constexpr std::array<std::string_view, 4> resample_scalers{"bicubic", "lanczos", "bilinear", "spline"};

// A resolution test's scaling algorithm unless another is chosen, and the
// one written nowhere: a test with it has the identity, path and saved files
// tests had before they had a choice (the Options scaler, bicubic by default;
// a test saved with another is told apart by its saved scale_algorithm).
inline constexpr std::string_view default_resample_scaler = "bicubic";

// What a resolution test scales the reference down to (and back up from): a
// width, the height following from the reference's shape, and a label
// ("1080p"); and the scaling algorithm of both steps, which makes the test
// video, so it is part of what the test is (VmafOptions::scale_algorithm
// does not apply to it). A resolution test has no second file: both videos
// are made from the reference during the run, to see what a resolution
// alone costs.
struct ResampleTarget {
    int width = 0;
    std::string label;
    std::string scaler = std::string(default_resample_scaler);

    bool operator==(const ResampleTarget&) const = default;
};

// The resolution tests offered, largest first.
const std::vector<ResampleTarget>& resample_target_choices();

// A made-up path that stands for a second run of the same test video with
// the other scale direction (the Python app's
// synthetic_scale_direction_variant_path). Both directions can then be rows
// of their own, side by side, without one taking the other's saved scores or
// graph line.
std::filesystem::path synthetic_scale_direction_variant_path(const std::filesystem::path& distorted,
                                                             ScaleDirection direction);

// Which of the two bundled Vship builds calculates SSIMULACRA2, Butteraugli
// and CVVDP on the GPU: Vship-fast, which computes as libjxl and pycvvdp do,
// or stock Vship 5.1.2. Their scores differ slightly, and each build's are
// saved and reused apart.
enum class VshipBuild { fast, stock };

// A build as settings, run files and the command line name it: "vship-fast"
// or "vship"; and back, any other text being Vship-fast.
std::string_view vship_build_value(VshipBuild build);
VshipBuild vship_build_from(std::string_view value);
// A build's name for the log: "Vship-fast" or "Vship 5.1.2".
std::string_view vship_build_label(VshipBuild build);

// Where FFmpeg starts reading each video of a ranged run (ffmpeg_range_seek,
// gpu_frames.h); none: at the video's start.
struct RangeSeeks {
    std::optional<RangeSeek> source;
    std::optional<RangeSeek> distorted;

    bool operator==(const RangeSeeks&) const = default;
};

struct VmafOptions {
    std::string model = "version=vmaf_v0.6.1";  // the FFmpeg model= value a run uses
    // What the user chose, from which `model` is worked out.
    std::string model_choice = std::string(auto_model_choice);
    std::string custom_model_path;
    std::vector<std::string> extra_features;  // libvmaf features: "name=psnr", "name=float_ssim"
    int n_threads = 0;                        // 0: let the app decide (every core)
    int n_subsample = 1;
    std::string scale_algorithm = "bicubic";
    ScaleDirection scale_direction = ScaleDirection::source_to_distorted;
    bool compute_xpsnr = false;
    bool compute_vmaf = true;
    bool compute_vmaf_neg = false;
    // VMAF v1, with its own model: what the user chose, and what a run
    // worked it out to be ("path=<file>").
    bool compute_vmaf_v1 = false;
    std::string model_choice_v1 = std::string(auto_model_choice);
    std::string model_v1;
    // The range compared, in whole seconds of the reference counted as CVVDP
    // counts them (frame_range.h): from range_start to range_end; range_end
    // 0 compares to the videos' end. Both 0: the whole videos.
    long long range_start = 0;
    long long range_end = 0;
    // Both videos are decoded on the GPU where they can be (each falls back
    // to software on its own).
    bool gpu_decode = true;
    // A VVC video decoded on the GPU by vvdec-vulkan where the GPU metrics can take its pictures there
    // (Settings::vvc_decode_on_gpu; only with gpu_decode, vvdec_wanted).
    bool vvc_on_gpu = false;
    GpuVendor gpu_vendor = GpuVendor::automatic;
    // VMAF v0.6.1, NEG and v1 on the GPU when a GPU can score them.
    bool vmaf_on_gpu = true;
    VshipBuild vship_build = VshipBuild::fast;
    CropMode crop_mode = CropMode::automatic;
    // When set, a resolution test (ResampleTarget) instead of a comparison
    // of two files.
    std::optional<ResampleTarget> resample_test;
    // Not a choice of the user's: set where a run hands its two videos to FFmpeg.
    RangeSeeks range_seeks;

    // The keys of FFmpeg's metrics these options ask for, in the registry's
    // order.
    std::vector<std::string> requested_metrics() const;
    // Whether these options ask for one of FFmpeg's metrics.
    bool metric_enabled(std::string_view key) const;
    // Turns one of FFmpeg's metrics on or off. Throws std::invalid_argument
    // for a metric that is not FFmpeg's.
    void set_metric_enabled(std::string_view key, bool enabled);

    bool operator==(const VmafOptions&) const = default;
};

}  // namespace vml
