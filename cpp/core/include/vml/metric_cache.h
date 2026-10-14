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
// pixel format and transfer tag (video_infos).
struct ColourTags {
    std::string pix_fmt;
    std::string color_transfer;
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
// GPU score never answers.
std::optional<std::pair<ComparisonResult, std::string>> load_result(
    const std::filesystem::path& base, const std::filesystem::path& source, const std::filesystem::path& distorted,
    const ComparisonRecipe& recipe, const std::vector<MetricRequestSpec>& specs,
    const std::map<std::string, std::string>& backends);

}  // namespace vml
