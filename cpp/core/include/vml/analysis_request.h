// What a comparison asks for, as the saved scores know it: the recipe that
// prepares the two videos' pictures, and each metric's request with the
// identity a saved score is filed under. A port of the Python app's
// comparison_recipe.py, analysis_request.py and ffmpeg_request.py.
#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vml/cvvdp.h"
#include "vml/options.h"
#include "vml/video.h"

namespace vml {

// The settings that prepare the pictures every metric compares.
struct ComparisonRecipe {
    CropMode crop_mode = CropMode::automatic;
    std::string scale_algorithm = "bicubic";
    ScaleDirection scale_direction = ScaleDirection::source_to_distorted;
    double duration_limit = 0.0;
    std::optional<ResampleTarget> resample_test;

    // The settings that decide what the comparison is (identity_dict): the
    // scaling algorithm is not one (the Python app counts a comparison
    // scaled another way as the same one), and the manual crops, which the
    // app no longer has, are null.
    nlohmann::json identity() const;
};

// Which source frames a frame metric's scores stand for: "full" (every
// frame) or "sampled" (every step-th).
struct FrameCoverage {
    std::string mode;
    int step = 1;
};

// One metric's request: what its scores mean, and so which saved scores can
// stand for them.
struct MetricRequestSpec {
    std::string key;
    std::string backend_id;  // "ffmpeg", "perceptual" or "cvvdp"
    nlohmann::json parameters = nlohmann::json::object();
    std::optional<FrameCoverage> coverage;
    std::string implementation_compatibility_id;

    // The identity a saved score is filed under (identity_dict).
    nlohmann::json identity() const;
};

// A comparison's request: its recipe, each metric's request in the order
// asked, and where SSIMULACRA2 and Butteraugli are set to be calculated
// ("gpu" or "cpu"), which decides which saved scores of theirs count.
struct AnalysisRequest {
    ComparisonRecipe recipe;
    std::vector<MetricRequestSpec> metrics;
    std::map<std::string, std::string> perceptual_backends;

    const MetricRequestSpec* metric(const std::string& key) const;
};

// The request for the metrics `metric_keys` of a comparison made with
// `options` (analysis_request_from_vmaf_options), with CVVDP modelling
// `cvvdp`'s display. Throws std::invalid_argument for a metric with no way
// to calculate it.
AnalysisRequest analysis_request(const VmafOptions& options, const std::vector<std::string>& metric_keys,
                                 const std::map<std::string, std::string>& perceptual_backends,
                                 const CvvdpSettings& cvvdp);

// The request of every saved score a row of the window can show for its
// comparison, its metric ticked or not (the Python app's
// displayable_metric_specs): FFmpeg's metrics as a run of all of them saves
// them, XPSNR as a run of XPSNR alone saves it (every frame, even when
// libvmaf's are subsampled), SSIMULACRA2 and Butteraugli at the row's frame
// coverage, and CVVDP for its display. Only for finding scores: a run still
// calculates the ticked metrics, and Recalculate clears those.
std::vector<MetricRequestSpec> displayable_metric_specs(const VmafOptions& options, const CvvdpSettings& cvvdp);

// A file's identity in the saved scores: its full path (resolved, as
// Python's Path.resolve() gives it), its size and its modification time in
// nanoseconds, "path:size:mtime"; -1 for both when it cannot be read.
std::string file_identity(const std::filesystem::path& path);

}  // namespace vml
