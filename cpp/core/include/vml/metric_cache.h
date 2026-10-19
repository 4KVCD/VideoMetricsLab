// The saved scores: scores kept on disk so that a run can reuse those
// calculated before instead of calculating them again. A port of the Python
// app's metric_cache.py (and the parts of result_cache.py the command line
// uses), on the same files, so that either app finds what the other saved.
//
// Each comparison has a folder of its own, <base>/v2/<hash>, named by a hash
// of the two files' identities (file_identity) and the comparison recipe. It
// holds context.json, which describes the comparison, and one .npz file per
// metric score, named by the metric and a hash of its request.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/analysis_request.h"
#include "vml/results.h"

namespace vml {

// The format of the saved scores this app reads and writes.
inline constexpr int metric_cache_format_version = 2;

// The folder the scores are saved in unless Settings name another:
// .videometricslab/results_cache in the user's home.
std::filesystem::path default_cache_dir();

// The hash that names a comparison's folder: of both files' identities and
// the recipe's.
std::string recipe_hash(const std::filesystem::path& source, const std::filesystem::path& distorted,
                        const ComparisonRecipe& recipe);

// The hash of a metric's request, which names its score's file.
std::string metric_identity_hash(const MetricRequestSpec& spec);

// The comparison's folder under `base`. Folders saved under the older
// identity, one per scaling algorithm, are merged into it the first time it
// is asked for; if the first of those cannot be renamed (it is open
// elsewhere), it is used where it is.
std::filesystem::path recipe_directory(const std::filesystem::path& base, const std::filesystem::path& source,
                                       const std::filesystem::path& distorted, const ComparisonRecipe& recipe);

// What the rule against known-wrong Vship scores reads of a video: its
// pixel format, transfer tag, chroma siting, colour matrix and height
// (video_infos).
struct ColourTags {
    std::string pix_fmt;
    std::string color_transfer;
    std::string chroma_location;
    std::string color_space;
    int height = 0;
};
std::vector<ColourTags> colour_tags(const VideoInfo& source, const VideoInfo& distorted);

// Whether a score made with `provenance` answers `spec`: made by a
// compatible implementation, and not known to be wrong for these videos.
bool answers(const MetricRequestSpec& spec, const MetricProvenance& provenance, const std::vector<ColourTags>& infos);

// Whether a score of `key` made on `produced` ("gpu" or "cpu") counts for
// the metric set to `choice` ("gpu" or "cpu"; nothing for a metric with no
// such choice, where any counts). SSIMULACRA2 and Butteraugli give
// different scores on the GPU and on the CPU, so set to CPU, only a CPU
// score counts, and set to GPU, a CPU score counts only where the GPU
// cannot calculate the metric (`gpu_can_score`).
bool made_as_chosen(const std::string& key, const std::optional<std::string>& choice, const std::string& produced,
                    const std::function<bool(const std::string&)>& gpu_can_score);

// The scores of `result` that a run asking for `specs` keeps instead of
// calculating them again: those that exist, were made where their metric
// is set to be calculated (`backends`), and answer the request.
MetricResultSet reusable(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                         const std::map<std::string, std::string>& backends,
                         const std::function<bool(const std::string&)>& gpu_can_score,
                         const std::vector<ColourTags>& infos);

// The keys of the scores reusable() keeps, in the order of `specs`, without
// copying the scores: for the window, which asks for each row whenever the
// row is drawn.
std::vector<std::string> reusable_keys(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                                       const std::map<std::string, std::string>& backends,
                                       const std::function<bool(const std::string&)>& gpu_can_score,
                                       const std::vector<ColourTags>& infos);

// Saves a run's result in `base`: the comparison's context.json and the
// score of each request in `specs` the result has. Returns the comparison's
// folder. Throws std::filesystem::filesystem_error (or std::runtime_error)
// when a file cannot be written.
std::filesystem::path store_result(const std::filesystem::path& base, const std::filesystem::path& source,
                                   const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                                   const ComparisonResult& result, const std::string& label,
                                   const std::vector<MetricRequestSpec>& specs);

// The saved scores of `specs` for a comparison, with the comparison's
// details and its label; nothing when none are saved. `backends` says where
// SSIMULACRA2 and Butteraugli are set to be calculated: set to CPU, a Vship
// GPU score never answers. The saved scores of `supplemental` come too, for
// the metrics `specs` do not ask for: a metric asked for has its own saved
// score or none, or a score made another way (XPSNR of every frame for one
// of every n-th) would stand for it and never be calculated as asked.
std::optional<std::pair<ComparisonResult, std::string>> load_result(
    const std::filesystem::path& base, const std::filesystem::path& source, const std::filesystem::path& distorted,
    const ComparisonRecipe& recipe, const std::vector<MetricRequestSpec>& specs,
    const std::map<std::string, std::string>& backends, const std::vector<MetricRequestSpec>& supplemental = {});

// CVVDP's scores saved for a comparison with other displays than the one
// its request in `specs` or `supplemental` models, each with the CVVDP
// parameters it was made with (the Python app's other_cvvdp_scores), and
// that request's parameters; nothing when neither asks for CVVDP. Never an
// answer for the request: a CVVDP score holds for its own display only.
struct OtherCvvdpScores {
    nlohmann::json parameters;                             // the request's
    std::vector<std::pair<CvvdpSettings, double>> scores;  // made for other displays
};
std::optional<OtherCvvdpScores> other_cvvdp_scores(const std::filesystem::path& base,
                                                   const std::filesystem::path& source,
                                                   const std::filesystem::path& distorted,
                                                   const ComparisonRecipe& recipe,
                                                   const std::vector<MetricRequestSpec>& specs,
                                                   const std::vector<MetricRequestSpec>& supplemental = {});

// Deletes a comparison's saved scores of `specs`, and those that would
// answer the same requests (another VMAF score made with the same model, a
// VMAF v1 score saved under VMAF's key, either implementation's SSIMULACRA2
// or Butteraugli), so that none comes back after a Recalculate; the folder
// goes with its last score. Returns how many files were deleted.
int clear_metrics(const std::filesystem::path& base, const std::filesystem::path& source,
                  const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                  const std::vector<MetricRequestSpec>& specs);

// A token for a row's request to the saved scores: of the two files'
// identities, the recipe and each metric's request. The window refuses a
// saved result read under a token that is no longer the row's: a file or
// the row's settings changed while it was being read.
std::string lookup_token(const std::filesystem::path& source, const std::filesystem::path& distorted,
                         const AnalysisRequest& request);

// How many comparisons are saved under `base`, and the bytes their files
// take, for Settings.
struct CacheSummary {
    int comparisons = 0;
    std::uintmax_t bytes = 0;
};
CacheSummary cache_summary(const std::filesystem::path& base);

// Deletes every comparison saved under `base` (its v2 folder; nothing else
// there) and returns how many there were: 0 when they could not all be
// deleted, as the Python app counts it.
int clear_all(const std::filesystem::path& base);

}  // namespace vml
