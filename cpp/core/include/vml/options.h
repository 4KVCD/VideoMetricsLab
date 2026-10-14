// A comparison's settings: which of FFmpeg's metrics to calculate, and how
// (the Python app's VmafOptions, a row's settings in the window).
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/video.h"

namespace vml {

// The two special model choices: "Auto" (the 4K model for a 4K comparison)
// and a custom model file.
inline constexpr std::string_view auto_model_choice = "__auto__";
inline constexpr std::string_view custom_model_choice = "__custom__";

// What a resolution test scales the reference down to (and back up from): a
// width, the height following from the reference's shape, and a label
// ("1080p"). A resolution test has no second file: both videos are made
// from the reference during the run, to see what a resolution alone costs.
struct ResampleTarget {
    int width = 0;
    std::string label;

    bool operator==(const ResampleTarget&) const = default;
};

// The resolution tests offered, largest first.
const std::vector<ResampleTarget>& resample_target_choices();

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
    double duration_limit = 0.0;  // seconds; 0 compares the whole videos
    // Both videos are decoded on the GPU where they can be (each falls back
    // to software on its own).
    bool gpu_decode = true;
    GpuVendor gpu_vendor = GpuVendor::automatic;
    // VMAF v0.6.1, NEG and v1 on the GPU when a GPU can score them.
    bool vmaf_on_gpu = true;
    CropMode crop_mode = CropMode::automatic;
    // When set, a resolution test (ResampleTarget) instead of a comparison
    // of two files.
    std::optional<ResampleTarget> resample_test;

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
