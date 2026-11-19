// What metric_cache.cpp, which keeps the saved scores' files, shares with
// metric_cache_load.cpp, which loads a request's saved scores back. Private
// to vml_core: in a namespace of its own, so that no name here meets another
// file's own helper of the same name.
#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "vml/metric_cache.h"
#include "vml/results.h"

namespace vml::cache_parts {

// The folder of this format's comparisons under the cache's base.
inline constexpr const char* format_folder = "v2";

// The scaling algorithms a comparison can be made with (the Options panel's
// and the command line's --scaler), each with a folder of its own: the
// recipe's identity holds it, and the older identity had one folder per
// algorithm even for a resolution test (recipe_directory).
inline constexpr std::array<std::string_view, 4> scale_algorithms{"bicubic", "lanczos", "bilinear", "spline"};

// A comparison's folder name: the hash of its two files' identities
// (file_identity) and its recipe's identity.
std::string hash_of(const std::string& source_identity, const std::string& distorted_identity,
                    const nlohmann::json& recipe_identity);

using LoadedMetric = std::variant<FrameMetricResult, SequenceMetricResult>;

// A provenance parameter's value: the JSON its text holds.
std::optional<nlohmann::json> parameter(const MetricProvenance& provenance, const std::string& name);

// The model and model choice of VMAF or VMAF v1 (`key`) in a result of saved
// scores (the Python app's _loaded_model). When it was loaded: the model its
// score was made with, as its file records it, and the choice of the request
// it answers, as calculating that request would give them. Otherwise `model`
// and `choice` stay as they are, context.json's: those of the result saved
// last, which can be another display's CVVDP alone (model "") or VMAF with
// another model chosen.
void loaded_model(const MetricResultSet& results, const std::vector<MetricRequestSpec>& specs, const std::string& key,
                  std::string& model, std::optional<std::string>& choice);

// The provenance a loaded score was saved with.
const MetricProvenance& provenance_of(const LoadedMetric& metric);

// Whether a SSIMULACRA2 or Butteraugli score's compatibility id is a Vship
// build's (the GPU's), which comes before the CPU's.
bool is_vship_id(std::string_view id);

// Whether a request is a SSIMULACRA2 or Butteraugli one that does not fix
// where the metric is calculated: on the GPU by its Vship build, or on the
// CPU (score_identity.h).
bool is_auto_perceptual(const MetricRequestSpec& spec);

// Writes a comparison folder's identity.json: each file's path identity and
// sampled_file_hash, the reference's video_fingerprint (where a
// fingerprinter is set), the frame rate the test video was read at
// (`recipe`'s) and its stream_fingerprint, of its first packets and of all
// of them, where those were read already (record_test_stream reads them),
// for recipe_directory to recognise them by when their paths change or the
// test stream is remuxed. With `only_if_missing`, a folder that has one
// keeps it.
void record_identity(const std::filesystem::path& directory, const std::filesystem::path& source,
                     const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                     bool only_if_missing);

// A JSON file's data, or nothing for a file that is missing or not JSON.
std::optional<nlohmann::json> read_json(const std::filesystem::path& path);

// Removes the scores saved in `directory` if its context cut other black bars
// than `context`, which replaces it: they would answer as this picture's.
void forget_scores_of_another_picture(const std::filesystem::path& directory, const nlohmann::json& context);

// Whether a score file's metadata is of the format this app writes.
bool current_format(const nlohmann::json& metadata);

// The whole-video scores saved for `spec`'s metric with other parameters but
// otherwise the same request, each with those parameters
// (load_other_parameters): CVVDP's for other displays, by the same Vship
// build. In a range (`ranged`), only a ranged run's: a duration limit's,
// saved before ranges, pooled more frames than the range has. `recipe`
// gives the videos' profiles where the context has none
// (with_profiles_read_now).
std::vector<std::pair<nlohmann::json, double>> other_parameters(const std::filesystem::path& directory,
                                                                const MetricRequestSpec& spec, bool ranged,
                                                                const ComparisonRecipe& recipe);

// Fills in a context.json's missing video profiles from `recipe`'s, as the
// videos are read now (in memory only). A comparison saved before 2.3 has
// none, and its H.264 scores would otherwise count as made from a maybe
// lossless 4:4:4 video (decoding_marks). Its folder is the same files' (by
// their identity in recipe_hash), so the profiles read now are theirs.
void with_profiles_read_now(nlohmann::json& context, const ComparisonRecipe& recipe);

// One saved metric score (load_metric), or nothing. `backend` is where the
// user set a perceptual metric to be calculated: "cpu" accepts only a CPU
// score; "gpu" prefers the request's Vship build's score and also accepts a
// CPU one, which is what that choice makes on a PC without a suitable GPU. A
// score known to be wrong is passed over. `context` is the comparison's
// context.json.
std::optional<LoadedMetric> load_metric(const std::filesystem::path& directory, const MetricRequestSpec& spec,
                                        const std::string& backend, const nlohmann::json& context);

// A result's FramesShort as context.json keeps it (null without one), and
// read back (nothing from null, a missing key or anything malformed): a run
// that loads the scores then need not count the frames again.
nlohmann::json frames_short_json(const std::optional<FramesShort>& short_of);
std::optional<FramesShort> frames_short_from_json(const nlohmann::json& saved);

}  // namespace vml::cache_parts
