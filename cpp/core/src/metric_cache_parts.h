// What metric_cache.cpp, which keeps the saved scores' files, shares with
// metric_cache_load.cpp, which loads a request's saved scores back. Private
// to vml_core: in a namespace of its own, so that no name here meets another
// file's own helper of the same name.
#pragma once

#include <nlohmann/json.hpp>

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

// A JSON file's data, or nothing for a file that is missing or not JSON.
std::optional<nlohmann::json> read_json(const std::filesystem::path& path);

// Whether a score file's metadata is of the format this app writes.
bool current_format(const nlohmann::json& metadata);

// The whole-video scores saved for `spec`'s metric with other parameters but
// otherwise the same request, each with those parameters
// (load_other_parameters): CVVDP's for other displays, by the same Vship
// build. In a range (`ranged`), only a ranged run's: a duration limit's,
// saved before ranges, pooled more frames than the range has.
std::vector<std::pair<nlohmann::json, double>> other_parameters(const std::filesystem::path& directory,
                                                                const MetricRequestSpec& spec, bool ranged);

// One saved metric score (load_metric), or nothing. `backend` is where the
// user set a perceptual metric to be calculated: "cpu" accepts only a CPU
// score; "gpu" prefers the request's Vship build's score and also accepts a
// CPU one, which is what that choice makes on a PC without a suitable GPU. A
// score known to be wrong is passed over. `context` is the comparison's
// context.json.
std::optional<LoadedMetric> load_metric(const std::filesystem::path& directory, const MetricRequestSpec& spec,
                                        const std::string& backend, const nlohmann::json& context);

}  // namespace vml::cache_parts
