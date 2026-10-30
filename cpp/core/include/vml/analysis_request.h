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
    // The two videos' film grain as they are scored: {"source": ..,
    // "distorted": ..}, each film_grain_identity's text or null; null where
    // neither has one (film_grain_scoring.h).
    nlohmann::json film_grain = nullptr;
    // The app's own decoder of each video whose codec it decodes
    // (app_decoders.h): {"source": .., "distorted": ..}, its identity text or
    // null; null where neither video is of such a codec.
    nlohmann::json app_decoder = nullptr;

    // The settings that decide what the comparison is (identity_dict): the
    // scaling algorithm is not one (the Python app counts a comparison
    // scaled another way as the same one), and the manual crops, which the
    // app no longer has, are null. The film grain and the app's decoder are in
    // it only where they are not null, so that every other comparison keeps
    // the identity it had.
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

// Where SSIMULACRA2 and Butteraugli are calculated unless a run asks for
// the CPU (the command line's --cpu): "gpu", on the GPU where one can, and
// on the CPU without one. Their GPU and CPU scores differ (each converts
// the videos to RGB its own way), so a CPU score of theirs counts
// only where the GPU cannot calculate it (made_as_chosen in metric_cache.h).
inline const std::map<std::string, std::string> automatic_perceptual_backends{{"butteraugli", "gpu"},
                                                                              {"ssimulacra2", "gpu"}};

// A comparison's request: its recipe, each metric's request in the order
// asked, and where SSIMULACRA2 and Butteraugli are calculated ("gpu" or
// "cpu"), which decides which saved scores of theirs count.
struct AnalysisRequest {
    ComparisonRecipe recipe;
    std::vector<MetricRequestSpec> metrics;
    std::map<std::string, std::string> perceptual_backends;

    const MetricRequestSpec* metric(const std::string& key) const;
};

// The request for the metrics `metric_keys` of a comparison of `distorted`
// with `source` made with `options` (analysis_request_from_vmaf_options),
// with CVVDP modelling `cvvdp`'s display. Its Vship metrics ask for the
// scores of the build that calculates here (vship_build_in_use), which can
// depend on Vship-fast's finished probe. Throws std::invalid_argument for a
// metric with no way to calculate it.
AnalysisRequest analysis_request(const VmafOptions& options, const std::vector<std::string>& metric_keys,
                                 const std::map<std::string, std::string>& perceptual_backends,
                                 const CvvdpSettings& cvvdp, const VideoInfo& source, const VideoInfo& distorted);

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
