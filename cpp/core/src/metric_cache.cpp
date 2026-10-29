#include "vml/metric_cache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <random>
#include <regex>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <variant>

#include "vml/app_log.h"
#include "vml/colour.h"
#include "vml/geometry.h"
#include "vml/hardware_decoding.h"
#include "vml/model_select.h"
#include "vml/npz.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/result_json.h"
#include "vml/score_identity.h"
#include "vml/sha256.h"
#include "vml/vship.h"

namespace vml {

namespace {

// The folder of this format's comparisons under the cache's base.
constexpr const char* format_folder = "v2";

// The id of a score saved before scores recorded their provenance.
constexpr const char* unversioned_compatibility = "unversioned";

// What a score's metadata carries when its pair has a video of a codec the
// GPUs decode to other pictures than FFmpeg's (gpu_decodes_exactly), which
// is decoded in software now: the version of that list. Such a pair's scores
// saved without it, or with an older one, may have been made from a GPU's
// pictures, and are calculated again. Raise it when a codec joins the list.
constexpr const char* exact_decoding = "exact_decoding";
constexpr int exact_decoding_version = 1;

// The scaling algorithms of the older identity, which had one folder per
// algorithm (recipe_directory).
constexpr std::array<std::string_view, 4> legacy_scale_algorithms{"bicubic", "lanczos", "bilinear", "spline"};

// The RGB pixel formats Vship scored wrongly without a transfer tag before
// app version 1.2.2, and the transfer tags that count as none.
const std::regex& old_vship_rgb() {
    static const std::regex pattern(
        "gbrp(?:(?:9|10|12|14|16)(?:le|be)?)?|rgb24|bgr24|rgba|bgra|argb|abgr|rgb0|bgr0|rgb48(?:le|be)|rgba64(?:le|be)");
    return pattern;
}

using LoadedMetric = std::variant<FrameMetricResult, SequenceMetricResult>;

std::string sha256_hex(const std::string& text) {
    Sha256 hash;
    hash.update(text.data(), text.size());
    return hash.hexdigest();
}

std::string hash_of(const std::string& source_identity, const std::string& distorted_identity,
                    const nlohmann::json& recipe_identity) {
    return sha256_hex(python_json_compact(
        {{"source", source_identity}, {"distorted", distorted_identity}, {"recipe", recipe_identity}}));
}

// A provenance parameter's value: the JSON its text holds.
std::optional<nlohmann::json> parameter(const MetricProvenance& provenance, const std::string& name) {
    const auto found = provenance.parameters.find(name);
    if (found == provenance.parameters.end()) {
        return std::nullopt;
    }
    return nlohmann::json::parse(found->second, nullptr, false);
}

// The model and model choice of VMAF or VMAF v1 (`key`) in a result of saved
// scores (the Python app's _loaded_model). When it was loaded: the model its
// score was made with, as its file records it, and the choice of the request
// it answers, as calculating that request would give them. Otherwise `model`
// and `choice` stay as they are, context.json's: those of the result saved
// last, which can be another display's CVVDP alone (model "") or VMAF with
// another model chosen.
void loaded_model(const MetricResultSet& results, const std::vector<MetricRequestSpec>& specs, const std::string& key,
                  std::string& model, std::optional<std::string>& choice) {
    const FrameMetricResult* metric = results.frame(key);
    const auto spec =
        std::find_if(specs.begin(), specs.end(), [&key](const MetricRequestSpec& asked) { return asked.key == key; });
    if (metric == nullptr || spec == specs.end()) {
        return;
    }
    if (const std::optional<nlohmann::json> ran = parameter(metric->provenance, "model");
        ran && ran->is_string() && !ran->get<std::string>().empty()) {
        model = ran->get<std::string>();
    }
    if (const auto chosen = spec->parameters.find("model_choice"); chosen != spec->parameters.end()) {
        if (chosen->is_string()) {
            choice = chosen->get<std::string>();
        } else {
            choice.reset();
        }
    }
}

// A provenance read back (provenance_from_dict); nothing for one of another
// shape.
std::optional<MetricProvenance> provenance_from(const nlohmann::json& data) {
    if (!data.is_object()) {
        return std::nullopt;
    }
    try {
        return provenance_from_dict(data);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

const MetricProvenance& provenance_of(const LoadedMetric& metric) {
    return std::visit([](const auto& result) -> const MetricProvenance& { return result.provenance; }, metric);
}

// Whether the pair's videos are read with another colour matrix now than
// each video's own tag or guess gave (matrix_shared in colour.h); not where
// the two videos' tags are not known.
bool matrix_read_otherwise(const std::vector<ColourTags>& infos) {
    if (infos.size() != 2) {
        return false;
    }
    const auto video = [](const ColourTags& tags) {
        VideoInfo info;
        info.pix_fmt = tags.pix_fmt;
        info.color_transfer = tags.color_transfer;
        info.color_space = tags.color_space;
        info.height = tags.height;
        return info;
    };
    try {
        return matrix_shared(video(infos[0]), video(infos[1]));
    } catch (const std::exception&) {
        return false;
    }
}

// Whether a CPU SSIMULACRA2 or Butteraugli score predates the colour
// handling the CPU's tools have now: one made before 8-bit videos were
// converted to RGB precisely, for a pair with an 8-bit video (or a format
// not known: FFmpeg's 8-bit 4:2:0 and 4:2:2 ones changed); one made before
// the pair was read with one colour matrix only where that changes what is
// read.
bool stale_cpu_score(const MetricProvenance& provenance, const std::vector<ColourTags>& infos) {
    if (provenance.compute_backend != "cpu" ||
        (provenance.implementation != "ssimulacra2" && provenance.implementation != "butteraugli")) {
        return false;
    }
    const nlohmann::json mapping = parameter(provenance, "color_tags");
    if (mapping == nlohmann::json(score_identity::cpu_colour_tags)) {
        return false;
    }
    const bool eight_bit = infos.size() != 2 || std::any_of(infos.begin(), infos.end(), [](const ColourTags& info) {
                               return bit_depth(info.pix_fmt) <= 8;
                           });
    if (eight_bit) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::cpu_colour_tags_8_bit_rgb)) {
        return false;
    }
    return mapping != nlohmann::json(score_identity::cpu_colour_tags_unshared_matrix) || matrix_read_otherwise(infos);
}

// Whether a video is gray tagged full range, which Vship 5.1 was sent
// squeezed into the limited range.
bool full_range_gray(const ColourTags& info) {
    static const std::regex gray("gray(?:(?:9|10|12|14|16)(?:le|be)?)?");
    const std::string range = ascii_lower(python_strip(info.color_range));
    return std::regex_match(ascii_lower(python_strip(info.pix_fmt)), gray) &&
           (range == "pc" || range == "jpeg" || range == "full");
}

// Whether a saved score is known to be wrong and must be calculated again:
// a stale CPU score, or a Vship 5.1 GPU score of a pair a later fix reads
// otherwise. That is a pair with a full-range gray video (full_range_gray);
// and, for a score that predates the fix, one whose videos are read with
// one colour matrix where that changes what Vship reads
// (matrix_read_otherwise), one with a video without chroma siting read with
// the other's where that changes what it reads (chroma_location_shared), or,
// from app version 1.2 or 1.2.1, one with an RGB video without a transfer
// tag. Vship-fast's scores are never stale.
bool stale_score(const MetricProvenance& provenance, const std::vector<ColourTags>& infos) {
    if (stale_cpu_score(provenance, infos)) {
        return true;
    }
    if (provenance.compute_backend != "gpu" ||
        !score_identity::is_vship51_id(provenance.implementation_compatibility_id)) {
        return false;
    }
    if (std::any_of(infos.begin(), infos.end(), full_range_gray)) {
        return true;
    }
    const nlohmann::json mapping = parameter(provenance, "color_tags");
    if (mapping == nlohmann::json(score_identity::vship_colour_tags)) {
        return false;
    }
    if (matrix_read_otherwise(infos)) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::vship_colour_tags_unshared_matrix)) {
        return false;
    }
    if (infos.size() == 2 && chroma_location_shared(infos[0].chroma_location, infos[1].chroma_location)) {
        return true;
    }
    if (mapping == nlohmann::json(score_identity::vship_colour_tags_unshared_siting)) {
        return false;
    }
    return std::any_of(infos.begin(), infos.end(), [](const ColourTags& info) {
        const std::string transfer = ascii_lower(python_strip(info.color_transfer));
        return std::regex_match(ascii_lower(python_strip(info.pix_fmt)), old_vship_rgb()) &&
               is_untagged_tag(transfer);
    });
}

// Whether a request is a SSIMULACRA2 or Butteraugli one that does not fix
// where the metric is calculated: on the GPU by its Vship build, or on the
// CPU (score_identity.h).
bool is_auto_perceptual(const MetricRequestSpec& spec) {
    const std::string& id = spec.implementation_compatibility_id;
    return spec.backend_id == "perceptual" &&
           (id.ends_with(score_identity::gpu_or_cpu_suffix) || id.ends_with(score_identity::vship51_gpu_or_cpu_suffix));
}

// Whether a saved perceptual score's compatibility id answers such a
// request: the request's Vship build's on the GPU, or libjxl's on the CPU,
// whose id may have a version inside.
bool auto_perceptual_compatible(const MetricRequestSpec& spec, const std::string& value) {
    const std::string& key = spec.key;
    const bool vship51 = spec.implementation_compatibility_id.ends_with(score_identity::vship51_gpu_or_cpu_suffix);
    return (vship51 ? score_identity::is_vship51_gpu(key, value) : value == score_identity::vship_gpu(key)) ||
           value == score_identity::libjxl_cpu(key) ||
           (value.starts_with(key + "-libjxl-") && value.ends_with("-cpu-v1"));
}

bool compatible(const MetricRequestSpec& spec, const MetricProvenance& provenance) {
    const std::string& compatibility = provenance.implementation_compatibility_id;
    if (is_auto_perceptual(spec)) {
        return auto_perceptual_compatible(spec, compatibility);
    }
    if (compatibility == unversioned_compatibility) {
        return spec.implementation_compatibility_id == score_identity::libvmaf;
    }
    return compatibility == spec.implementation_compatibility_id;
}

std::filesystem::path metric_path(const std::filesystem::path& directory, const MetricRequestSpec& spec) {
    return directory / platform::path_from_text(spec.key + "_" + metric_identity_hash(spec) + ".npz");
}

// Writes `bytes` to `path` through a temporary file that is then renamed
// over it, so that a crash never leaves half a file.
void write_atomically(const std::filesystem::path& path, const std::string& bytes, const std::string& suffix) {
    std::filesystem::create_directories(path.parent_path());
    std::random_device random;
    const std::filesystem::path temp =
        path.parent_path() /
        platform::path_from_text("." + platform::path_text(path.filename()) + "." + std::to_string(random()) + suffix);
    {
        std::ofstream file(temp, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        file.close();
        if (!file) {
            std::error_code ignored;
            std::filesystem::remove(temp, ignored);
            throw std::runtime_error("Could not write " + platform::path_text(path));
        }
    }
    std::error_code error;
    std::filesystem::rename(temp, path, error);
    if (error) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        throw std::filesystem::filesystem_error("Could not save the scores", path, error);
    }
}

// The metadata saved with a score: the cache format, the kind of score, the
// request it answers and the score's provenance (with the version that made
// it, which the results name; no lookup reads it), and for a pair with an
// inexact codec, exact_decoding.
std::string metadata(const std::string& key, const char* kind, const MetricRequestSpec& spec,
                     const MetricProvenance& provenance, bool inexact_codec) {
    nlohmann::json data{{"format_version", metric_cache_format_version},
                        {"kind", kind},
                        {"key", key},
                        {"request", spec.identity()},
                        {"provenance", provenance_to_dict(provenance)}};
    if (inexact_codec) {
        data[exact_decoding] = exact_decoding_version;
    }
    return python_json_compact(data);
}

// The request a perceptual score is saved under: the implementation that
// made it, so that a CPU score never passes for a Vship GPU one, nor one
// Vship build's for the other's. A score the run's other build made (Vship
// 5.1.2 standing in for Vship-fast) is so kept for that build's requests.
MetricRequestSpec stored_spec(const MetricRequestSpec& spec, const MetricProvenance& provenance) {
    MetricRequestSpec stored = spec;
    if (spec.backend_id == "perceptual") {
        stored.implementation_compatibility_id = provenance.implementation_compatibility_id;
    }
    return stored;
}

void store_metric(const std::filesystem::path& directory, const FrameMetricResult& result,
                  const MetricRequestSpec& spec, bool inexact_codec) {
    const MetricRequestSpec stored = stored_spec(spec, result.provenance);
    const std::string described = metadata(result.key, "frame", stored, result.provenance, inexact_codec);
    write_atomically(metric_path(directory, stored),
                     npz_bytes({{"metadata", npy_text(described)},
                                {"frame", npy_int32(result.frame)},
                                {"time", npy_float64(result.time)},
                                {"values", npy_float32(result.values)}}),
                     ".npz");
}

void store_metric(const std::filesystem::path& directory, const SequenceMetricResult& result,
                  const MetricRequestSpec& spec, bool inexact_codec) {
    const MetricRequestSpec stored = stored_spec(spec, result.provenance);
    std::vector<std::pair<std::string, NpyArray>> arrays{
        {"metadata", npy_text(metadata(result.key, "sequence", stored, result.provenance, inexact_codec))},
        {"score", npy_float64_scalar(result.score)}};
    if (result.has_timeline()) {
        arrays.emplace_back("frame", npy_int32(result.frame));
        arrays.emplace_back("time", npy_float64(result.time));
        arrays.emplace_back("values", npy_float32(result.values));
    }
    write_atomically(metric_path(directory, stored), npz_bytes(arrays), ".npz");
}

nlohmann::json optional_text(const std::optional<std::string>& text) {
    return text ? nlohmann::json(*text) : nlohmann::json(nullptr);
}

// The comparison's context.json: its files, crops, frame rate and recipe.
nlohmann::json context_json(const ComparisonResult& result, const std::string& label, const ComparisonRecipe& recipe) {
    return {{"format_version", metric_cache_format_version},
            {"label", label},
            {"source", python_path_text(result.source)},
            {"distorted", python_path_text(result.distorted)},
            {"fps", result.fps},
            {"source_crop", nlohmann::json(crop_to_dict(result.source_crop))},
            {"distorted_crop", nlohmann::json(crop_to_dict(result.distorted_crop))},
            {"source_info", nlohmann::json(video_info_to_dict(result.source_info))},
            {"distorted_info", nlohmann::json(video_info_to_dict(result.distorted_info))},
            {"scale_direction", scale_direction_value(result.scale_direction)},
            {"scale_algorithm", result.scale_algorithm},
            {"resample_target", nlohmann::json(resample_target_to_dict(result.resample_target))},
            {"compared_frame_count", result.compared_frame_count},
            {"model", result.model},
            {"model_choice", optional_text(result.model_choice)},
            {"model_v1", result.model_v1},
            {"model_choice_v1", optional_text(result.model_choice_v1)},
            {"recipe", recipe.identity()}};
}

std::optional<nlohmann::json> read_json(const std::filesystem::path& path) {
    const std::optional<std::string> text = platform::read_file(path);
    if (!text) {
        return std::nullopt;
    }
    nlohmann::json data = nlohmann::json::parse(*text, nullptr, false);
    if (data.is_discarded()) {
        return std::nullopt;
    }
    return data;
}

// The two videos' colour tags as a comparison's context.json has them.
std::vector<ColourTags> context_colour_tags(const nlohmann::json& context) {
    std::vector<ColourTags> infos;
    if (!context.is_object()) {
        return infos;
    }
    for (const char* key : {"source_info", "distorted_info"}) {
        const nlohmann::json info = context.value(key, nlohmann::json());
        if (info.is_object()) {
            const auto text = [&info](const char* name) {
                const nlohmann::json value = info.value(name, nlohmann::json());
                return value.is_null() ? std::string() : python_str(value);
            };
            const nlohmann::json height = info.value("height", nlohmann::json());
            infos.push_back({text("pix_fmt"), text("color_transfer"), text("chroma_location"), text("color_space"),
                             text("color_range"), height.is_number_integer() ? height.get<int>() : 0});
        }
    }
    return infos;
}

// Whether a comparison's context.json gives a video of a codec the GPUs
// decode inexactly, whose saved scores count only with exact_decoding.
bool has_inexact_codec(const nlohmann::json& context) {
    if (!context.is_object()) {
        return false;
    }
    for (const char* key : {"source_info", "distorted_info"}) {
        const nlohmann::json info = context.value(key, nlohmann::json());
        const nlohmann::json codec = info.is_object() ? info.value("codec_name", nlohmann::json()) : nlohmann::json();
        if (codec.is_string() && !gpu_decodes_exactly(codec.get<std::string>())) {
            return true;
        }
    }
    return false;
}

// Whether a saved score was made from the pictures its pair is decoded to
// now: for a pair with an inexact codec, one with this exact_decoding.
bool decoded_as_now(const nlohmann::json& metadata, bool inexact_codec) {
    return !inexact_codec || metadata.value(exact_decoding, nlohmann::json()) == exact_decoding_version;
}

// A saved score's metadata, or nothing if the file cannot be read.
std::optional<std::pair<nlohmann::json, std::map<std::string, NpyArray>>> read_score(const std::filesystem::path& path) {
    std::optional<std::map<std::string, NpyArray>> arrays = read_npz(path);
    if (!arrays || !arrays->contains("metadata")) {
        return std::nullopt;
    }
    const std::optional<std::string> text = npy_text_value(arrays->at("metadata"));
    if (!text) {
        return std::nullopt;
    }
    nlohmann::json metadata = nlohmann::json::parse(*text, nullptr, false);
    if (!metadata.is_object()) {
        return std::nullopt;
    }
    return std::pair{std::move(metadata), std::move(*arrays)};
}

bool current_format(const nlohmann::json& metadata) {
    const nlohmann::json version = metadata.value("format_version", nlohmann::json());
    return version.is_number() && version.get<double>() == metric_cache_format_version;
}

// `object` without the keys `names`.
nlohmann::json without(nlohmann::json object, std::initializer_list<const char*> names) {
    if (object.is_object()) {
        for (const char* name : names) {
            object.erase(name);
        }
    }
    return object;
}

// A frame metric with only every `step`-th frame (sampled_every): the
// frames a subsampled request covers.
FrameMetricResult sampled_every(FrameMetricResult result, int step) {
    if (step <= 1 || std::none_of(result.frame.begin(), result.frame.end(), [step](int f) { return f % step != 0; })) {
        return result;
    }
    FrameMetricResult kept{result.key, {}, {}, {}, result.provenance};
    for (std::size_t i = 0; i < result.frame.size(); ++i) {
        if (result.frame[i] % step == 0) {
            kept.frame.push_back(result.frame[i]);
            kept.time.push_back(result.time[i]);
            kept.values.push_back(result.values[i]);
        }
    }
    return kept;
}

// A saved timeline's arrays, or nothing if one is missing or their lengths
// differ.
std::optional<std::tuple<std::vector<int>, std::vector<double>, std::vector<float>>> timeline(
    const std::map<std::string, NpyArray>& arrays) {
    if (!arrays.contains("frame") || !arrays.contains("time") || !arrays.contains("values")) {
        return std::nullopt;
    }
    const auto frames = npy_numbers(arrays.at("frame"));
    const auto times = npy_numbers(arrays.at("time"));
    const auto values = npy_numbers(arrays.at("values"));
    if (!frames || !times || !values || frames->size() != times->size() || frames->size() != values->size()) {
        return std::nullopt;
    }
    std::tuple<std::vector<int>, std::vector<double>, std::vector<float>> converted;
    for (const double frame : *frames) {
        std::get<0>(converted).push_back(static_cast<int>(frame));
    }
    std::get<1>(converted) = *times;
    for (const double value : *values) {
        std::get<2>(converted).push_back(static_cast<float>(value));
    }
    return converted;
}

// The score saved in `path` if it was saved for exactly `spec`
// (_load_metric_file), from the pictures the pair is decoded to now
// (`inexact_codec`: has_inexact_codec); nothing otherwise, or if the file
// cannot be read.
std::optional<LoadedMetric> load_metric_file(const std::filesystem::path& path, const MetricRequestSpec& spec,
                                             bool inexact_codec) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        return std::nullopt;
    }
    try {
        const auto score = read_score(path);
        if (!score) {
            return std::nullopt;
        }
        const auto& [metadata, arrays] = *score;
        if (!current_format(metadata) || !decoded_as_now(metadata, inexact_codec) ||
            metadata.value("key", nlohmann::json()) != spec.key ||
            metadata.value("request", nlohmann::json()) != spec.identity()) {
            return std::nullopt;
        }
        const std::optional<MetricProvenance> provenance = provenance_from(metadata.at("provenance"));
        if (!provenance) {
            return std::nullopt;
        }
        const nlohmann::json kind = metadata.at("kind");
        if (kind == "frame") {
            auto frames = timeline(arrays);
            if (!frames) {
                return std::nullopt;
            }
            auto& [frame, time, values] = *frames;
            return sampled_every(FrameMetricResult{spec.key, std::move(frame), std::move(time), std::move(values), *provenance},
                                 spec.coverage ? spec.coverage->step : 1);
        }
        if (kind == "sequence") {
            std::optional<std::vector<double>> score_value;
            if (arrays.contains("score")) {
                score_value = npy_numbers(arrays.at("score"));
            }
            if (!score_value || score_value->size() != 1) {
                return std::nullopt;
            }
            SequenceMetricResult result{spec.key, score_value->front(), *provenance, {}, {}, {}};
            if (arrays.contains("values")) {
                auto frames = timeline(arrays);
                if (!frames) {
                    return std::nullopt;
                }
                std::tie(result.frame, result.time, result.values) = std::move(*frames);
            }
            return result;
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
    return std::nullopt;
}

// The comparison's score files of one metric, by name: <key>_<hash>.npz.
std::vector<std::filesystem::path> score_files(const std::filesystem::path& directory, const std::string& key) {
    std::vector<std::filesystem::path> paths;
    std::error_code error;
    std::filesystem::directory_iterator entry(directory, error);
    for (; !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        const std::string name = platform::path_text(entry->path().filename());
        if (name.starts_with(key + "_") && name.ends_with(".npz")) {
            paths.push_back(entry->path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// The saved scores that answer a perceptual request which does not fix
// where the metric is calculated, each with the request carrying its
// implementation: the request's Vship build's first, then the CPU's.
std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> auto_perceptual_candidates(
    const std::filesystem::path& directory, const MetricRequestSpec& spec) {
    std::vector<std::pair<int, std::pair<std::filesystem::path, MetricRequestSpec>>> ranked;
    for (const std::filesystem::path& path : score_files(directory, spec.key)) {
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const nlohmann::json& metadata = score->first;
            const nlohmann::json request = metadata.value("request", nlohmann::json::object());
            const nlohmann::json compatibility = request.value("implementation_compatibility_id", nlohmann::json());
            const nlohmann::json provenance = metadata.value("provenance", nlohmann::json::object());
            const std::string value = compatibility.is_string() ? compatibility.get<std::string>() : std::string();
            if (!current_format(metadata) || metadata.value("key", nlohmann::json()) != spec.key ||
                !auto_perceptual_compatible(spec, value) || !provenance.is_object() ||
                provenance.value("implementation_compatibility_id", nlohmann::json()) != compatibility) {
                continue;
            }
            MetricRequestSpec concrete = spec;
            concrete.implementation_compatibility_id = value;
            if (request != concrete.identity()) {
                continue;
            }
            const int rank = value.find("-vship-") != std::string::npos ? 0 : 1;
            ranked.push_back({rank, {path, concrete}});
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> found;
    for (auto& [rank, candidate] : ranked) {
        found.push_back(std::move(candidate));
    }
    return found;
}

// The size this comparison's frames are compared at, worked out from the
// sizes and black bars in its context.json, as the run works it out. Auto
// picks VMAF's model by it.
std::optional<Size> compared_size(const std::filesystem::path& directory) {
    try {
        const std::optional<nlohmann::json> context = read_json(directory / "context.json");
        if (!context || !context->is_object()) {
            return std::nullopt;
        }
        const VideoInfo source = video_info_from_dict(context->at("source_info"));
        if (python_truthy(context->value("resample_target", nlohmann::json()))) {
            // A resolution test compares at the source's own (cropped) size:
            // the downscale is undone before the comparison.
            return resample_analysis_dimensions(source,
                                                crop_from_dict(context->value("source_crop", nlohmann::json())));
        }
        const VideoInfo distorted = video_info_from_dict(context->at("distorted_info"));
        return compared_dimensions(source, distorted,
                                   scale_direction_from(context->at("scale_direction").get<std::string>()),
                                   crop_from_dict(context->value("source_crop", nlohmann::json())),
                                   crop_from_dict(context->value("distorted_crop", nlohmann::json())));
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

// An object's member, or {} when it is missing, null or not an object (the
// Python app's `x.get(name) or {}`).
nlohmann::json member_object(const nlohmann::json& object, const char* name) {
    const nlohmann::json value = object.is_object() ? object.value(name, nlohmann::json()) : nlohmann::json();
    return value.is_object() ? value : nlohmann::json::object();
}

// The model a saved VMAF score was calculated with, or nothing when that is
// not certain (_vmaf_model_ran): an explicitly chosen (or bundled) model is
// the model; for Auto, the model the score's provenance records, or the one
// a score migrated from the first cache format was saved under.
std::optional<std::string> vmaf_model_ran(const nlohmann::json& request, const nlohmann::json& provenance) {
    const nlohmann::json parameters = member_object(request, "parameters");
    const nlohmann::json choice = parameters.value("model_choice", nlohmann::json());
    if (choice.is_string() && choice != "" && choice != std::string(auto_model_choice) &&
        choice != std::string(custom_model_choice)) {
        return choice.get<std::string>();
    }
    if (choice != std::string(auto_model_choice)) {
        return std::nullopt;
    }
    const nlohmann::json ran = member_object(provenance, "parameters").value("model", nlohmann::json());
    if (ran.is_string() && ran.get<std::string>().starts_with("version=")) {
        return ran.get<std::string>();
    }
    const nlohmann::json model = parameters.value("model", nlohmann::json());
    if (provenance.is_object() && provenance.value("implementation", nlohmann::json()) == "legacy-v1-cache" &&
        model.is_string() && model.get<std::string>().starts_with("version=")) {
        return model.get<std::string>();
    }
    return std::nullopt;
}

// Saved VMAF scores of the same comparison under another request that were
// calculated with the same model, best match first: for a request on Auto,
// any other Auto score, or a score of the model Auto picks here; for an
// explicit model, an Auto score that certainly ran it.
std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> vmaf_equivalents(const std::filesystem::path& directory,
                                                                                  const MetricRequestSpec& spec) {
    const nlohmann::json wanted = spec.identity();
    const nlohmann::json& parameters = wanted["parameters"];
    const nlohmann::json choice = parameters.value("model_choice", nlohmann::json());
    const nlohmann::json rest = without(wanted, {"parameters"});
    const std::filesystem::path exact = metric_path(directory, spec);
    std::optional<std::optional<std::string>> auto_model;  // worked out once, only if needed
    std::vector<std::pair<int, std::pair<std::filesystem::path, MetricRequestSpec>>> ranked;
    for (const std::filesystem::path& path : score_files(directory, "vmaf")) {
        if (path == exact) {
            continue;
        }
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const nlohmann::json& metadata = score->first;
            const nlohmann::json request = member_object(metadata, "request");
            const nlohmann::json stored = member_object(request, "parameters");
            if (!current_format(metadata) || metadata.value("key", nlohmann::json()) != "vmaf" ||
                without(request, {"parameters"}) != rest ||
                stored.value("custom_model", nlohmann::json("")) != parameters.value("custom_model", nlohmann::json(""))) {
                continue;
            }
            int rank = 1;
            if (stored.value("model_choice", nlohmann::json()) != choice) {
                const std::optional<std::string> ran = vmaf_model_ran(request, member_object(metadata, "provenance"));
                if (!ran) {
                    continue;
                }
                if (choice == std::string(auto_model_choice)) {
                    if (!auto_model) {
                        auto_model.emplace();
                        if (const std::optional<Size> size = compared_size(directory)) {
                            *auto_model = model_for_resolution(size->first, size->second);
                        }
                    }
                    if (*auto_model != *ran) {
                        continue;
                    }
                } else if (!(choice.is_string() && choice.get<std::string>().starts_with("version=") &&
                             stored.value("model_choice", nlohmann::json()) == std::string(auto_model_choice) &&
                             *ran == choice.get<std::string>())) {
                    continue;
                }
                rank = 2;
            }
            MetricRequestSpec concrete = spec;
            concrete.parameters = stored;
            ranked.push_back({rank, {path, concrete}});
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> found;
    for (auto& [rank, candidate] : ranked) {
        found.push_back(std::move(candidate));
    }
    return found;
}

bool is_v1_choice(const nlohmann::json& choice) {
    return choice.is_string() && !choice.get<std::string>().empty() && is_builtin_model_choice(choice.get<std::string>());
}

// VMAF v1 scores saved before VMAF v1 had a column of its own: under the key
// "vmaf", with a bundled v1 model as the model choice. They answer a VMAF v1
// request for that model, or one on Auto when it is the model Auto picks.
std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> old_v1_scores(const std::filesystem::path& directory,
                                                                               const MetricRequestSpec& spec) {
    const nlohmann::json wanted = spec.identity();
    const nlohmann::json choice = wanted["parameters"].value("model_choice", nlohmann::json());
    const nlohmann::json rest = without(wanted, {"key", "parameters"});
    std::optional<std::optional<std::string>> auto_model;
    std::vector<std::pair<std::filesystem::path, MetricRequestSpec>> found;
    for (const std::filesystem::path& path : score_files(directory, "vmaf")) {
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const nlohmann::json& metadata = score->first;
            const nlohmann::json request = member_object(metadata, "request");
            const nlohmann::json stored = member_object(request, "parameters");
            const nlohmann::json stored_choice = stored.value("model_choice", nlohmann::json());
            if (!current_format(metadata) || metadata.value("key", nlohmann::json()) != "vmaf" ||
                !is_v1_choice(stored_choice) || without(request, {"key", "parameters"}) != rest) {
                continue;
            }
            if (choice == std::string(auto_model_choice)) {
                if (!auto_model) {
                    auto_model.emplace();
                    if (const std::optional<Size> size = compared_size(directory)) {
                        *auto_model = v1_model_for_resolution(size->first, size->second);
                    }
                }
                if (!*auto_model || stored_choice != **auto_model) {
                    continue;
                }
            } else if (stored_choice != choice) {
                continue;
            }
            MetricRequestSpec concrete = spec;
            concrete.key = "vmaf";
            concrete.parameters = stored;
            found.emplace_back(path, concrete);
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    return found;
}

// The whole-video scores saved for `spec`'s metric with other parameters but
// otherwise the same request, each with those parameters
// (load_other_parameters): CVVDP's for other displays, by the same Vship
// build.
std::vector<std::pair<nlohmann::json, double>> other_parameters(const std::filesystem::path& directory,
                                                                const MetricRequestSpec& spec) {
    const nlohmann::json wanted = spec.identity();
    const nlohmann::json rest = without(wanted, {"parameters"});
    const std::vector<std::filesystem::path> paths = score_files(directory, spec.key);
    const nlohmann::json context =
        paths.empty() ? nlohmann::json() : read_json(directory / "context.json").value_or(nlohmann::json());
    const std::vector<ColourTags> infos = context_colour_tags(context);
    const bool inexact_codec = has_inexact_codec(context);
    std::vector<std::pair<nlohmann::json, double>> found;
    for (const std::filesystem::path& path : paths) {
        try {
            const auto score = read_score(path);
            if (!score) {
                continue;
            }
            const auto& [metadata, arrays] = *score;
            const nlohmann::json request = metadata.value("request", nlohmann::json::object());
            const nlohmann::json request_rest = without(request, {"parameters"});
            if (!current_format(metadata) || !decoded_as_now(metadata, inexact_codec) ||
                metadata.value("key", nlohmann::json()) != spec.key ||
                metadata.value("kind", nlohmann::json()) != "sequence" ||
                request_rest != rest ||
                request.value("parameters", nlohmann::json()) == wanted.at("parameters")) {
                continue;
            }
            if (const nlohmann::json made = metadata.value("provenance", nlohmann::json()); made.is_object()) {
                const std::optional<MetricProvenance> provenance = provenance_from(made);
                if (!provenance || stale_score(*provenance, infos)) {
                    continue;
                }
            }
            const nlohmann::json parameters = request.at("parameters");
            const std::optional<std::vector<double>> value =
                arrays.contains("score") ? npy_numbers(arrays.at("score")) : std::nullopt;
            if (!parameters.is_object() || !value || value->size() != 1) {
                continue;
            }
            if (std::isfinite(value->front())) {
                found.emplace_back(parameters, value->front());
            }
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    return found;
}

// One saved metric score (load_metric), or nothing. `backend` is where the
// user set a perceptual metric to be calculated: "cpu" accepts only a CPU
// score; "gpu" prefers the request's Vship build's score and also accepts a
// CPU one, which is what that choice makes on a PC without a suitable GPU. A
// score known to be wrong is passed over. `context` is the comparison's
// context.json.
std::optional<LoadedMetric> load_metric(const std::filesystem::path& directory, const MetricRequestSpec& spec,
                                        const std::string& backend, const nlohmann::json& context) {
    const std::vector<ColourTags> infos = context_colour_tags(context);
    const bool inexact_codec = has_inexact_codec(context);
    const auto stale = [&infos](const LoadedMetric& metric) { return stale_score(provenance_of(metric), infos); };
    if (is_auto_perceptual(spec)) {
        for (const auto& [path, concrete] : auto_perceptual_candidates(directory, spec)) {
            if (backend == "cpu" && concrete.implementation_compatibility_id.find("-vship-") != std::string::npos) {
                continue;
            }
            std::optional<LoadedMetric> result = load_metric_file(path, concrete, inexact_codec);
            if (result && !stale(*result)) {
                return result;
            }
        }
        return std::nullopt;
    }
    std::optional<LoadedMetric> result = load_metric_file(metric_path(directory, spec), spec, inexact_codec);
    if (result && stale(*result)) {
        return std::nullopt;
    }
    if (!result && spec.key == "vmaf" && spec.backend_id == "ffmpeg") {
        for (const auto& [path, concrete] : vmaf_equivalents(directory, spec)) {
            result = load_metric_file(path, concrete, inexact_codec);
            if (result) {
                break;
            }
        }
    }
    if (!result && spec.key == "vmaf_v1" && spec.backend_id == "ffmpeg") {
        for (const auto& [path, concrete] : old_v1_scores(directory, spec)) {
            std::optional<LoadedMetric> old = load_metric_file(path, concrete, inexact_codec);
            if (old && std::holds_alternative<FrameMetricResult>(*old)) {
                FrameMetricResult v1 = std::get<FrameMetricResult>(std::move(*old));
                v1.key = "vmaf_v1";
                return v1;
            }
        }
    }
    return result;
}

void merge_legacy(const std::filesystem::path& legacy, const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::directory_iterator item(legacy, error);
    for (; !error && item != std::filesystem::directory_iterator(); item.increment(error)) {
        const std::filesystem::path target = directory / item->path().filename();
        std::error_code ignored;
        if (std::filesystem::exists(target, ignored)) {
            if (item->is_regular_file(ignored)) {
                std::filesystem::remove(item->path(), ignored);
            }
        } else {
            std::filesystem::rename(item->path(), target, ignored);
        }
    }
    std::error_code ignored;
    std::filesystem::remove(legacy, ignored);  // only if it is empty now
}

}  // namespace

std::filesystem::path default_cache_dir() {
    return platform::home_directory() / ".videometricslab" / "results_cache";
}

std::string recipe_hash(const std::filesystem::path& source, const std::filesystem::path& distorted,
                        const ComparisonRecipe& recipe) {
    return hash_of(file_identity(source), file_identity(distorted), recipe.identity());
}

std::string metric_identity_hash(const MetricRequestSpec& spec) {
    return sha256_hex(python_json_compact(spec.identity()));
}

std::filesystem::path recipe_directory(const std::filesystem::path& base, const std::filesystem::path& source,
                                       const std::filesystem::path& distorted, const ComparisonRecipe& recipe) {
    const std::filesystem::path root = base / format_folder;
    const std::string source_identity = file_identity(source);
    const std::string distorted_identity = file_identity(distorted);
    const nlohmann::json identity = recipe.identity();
    const std::filesystem::path directory = root / hash_of(source_identity, distorted_identity, identity);
    std::error_code error;
    if (std::filesystem::exists(directory, error) || !std::filesystem::is_directory(root, error)) {
        return directory;
    }
    std::vector<std::string> algorithms{recipe.scale_algorithm};
    for (const std::string_view algorithm : legacy_scale_algorithms) {
        if (algorithm != recipe.scale_algorithm) {
            algorithms.emplace_back(algorithm);
        }
    }
    std::vector<std::filesystem::path> legacy;
    for (const std::string& algorithm : algorithms) {
        nlohmann::json old_identity = identity;
        old_identity["scale_algorithm"] = algorithm;
        const std::filesystem::path path = root / hash_of(source_identity, distorted_identity, old_identity);
        if (std::filesystem::is_directory(path, error)) {
            legacy.push_back(path);
        }
    }
    if (legacy.empty()) {
        return directory;
    }
    std::filesystem::rename(legacy.front(), directory, error);
    if (error) {
        return legacy.front();
    }
    for (std::size_t i = 1; i < legacy.size(); ++i) {
        merge_legacy(legacy[i], directory);
    }
    return directory;
}

std::vector<ColourTags> colour_tags(const VideoInfo& source, const VideoInfo& distorted) {
    return {{source.pix_fmt, source.color_transfer, source.chroma_location, source.color_space, source.color_range,
             source.height},
            {distorted.pix_fmt, distorted.color_transfer, distorted.chroma_location, distorted.color_space,
             distorted.color_range, distorted.height}};
}

bool answers(const MetricRequestSpec& spec, const MetricProvenance& provenance, const std::vector<ColourTags>& infos) {
    return compatible(spec, provenance) && !stale_score(provenance, infos);
}

bool made_as_chosen(const std::string& key, const std::optional<std::string>& choice, const std::string& produced,
                    const std::function<bool(const std::string&)>& gpu_can_score) {
    if (choice == "cpu") {
        return produced == "cpu";
    }
    if (choice == "gpu" && produced == "cpu") {
        return !gpu_can_score(key);
    }
    return true;
}

std::vector<std::string> reusable_keys(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                                       const std::map<std::string, std::string>& backends,
                                       const std::function<bool(const std::string&)>& gpu_can_score,
                                       const std::vector<ColourTags>& infos) {
    std::vector<std::string> kept;
    for (const MetricRequestSpec& spec : specs) {
        if (const SequenceMetricResult* sequence = result.metric_results.sequence(spec.key)) {
            // One score for the video (CVVDP).
            if (std::isfinite(sequence->score) && answers(spec, sequence->provenance, infos)) {
                kept.push_back(spec.key);
            }
            continue;
        }
        const FrameMetricResult* metric = result.metric_results.frame(spec.key);
        if (metric == nullptr ||
            std::all_of(metric->values.begin(), metric->values.end(), [](float value) { return std::isnan(value); })) {
            continue;
        }
        std::optional<std::string> choice;
        if (const auto found = backends.find(spec.key); found != backends.end()) {
            choice = found->second;
        }
        if (made_as_chosen(spec.key, choice, metric->provenance.compute_backend, gpu_can_score) &&
            answers(spec, metric->provenance, infos)) {
            kept.push_back(spec.key);
        }
    }
    return kept;
}

MetricResultSet reusable(const ComparisonResult& result, const std::vector<MetricRequestSpec>& specs,
                         const std::map<std::string, std::string>& backends,
                         const std::function<bool(const std::string&)>& gpu_can_score,
                         const std::vector<ColourTags>& infos) {
    MetricResultSet kept;
    for (const std::string& key : reusable_keys(result, specs, backends, gpu_can_score, infos)) {
        kept.copy_from(result.metric_results, key);
    }
    return kept;
}

std::filesystem::path store_result(const std::filesystem::path& base, const std::filesystem::path& source,
                                   const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                                   const ComparisonResult& result, const std::string& label,
                                   const std::vector<MetricRequestSpec>& specs) {
    const std::filesystem::path directory = recipe_directory(base, source, distorted, recipe);
    write_atomically(directory / "context.json", python_json_compact(context_json(result, label, recipe)), ".tmp");
    // A score saved under a request it does not answer would answer it from
    // then on: loading checks the saved request, not the score. A perceptual
    // score is saved under its own implementation's id (stored_spec), which
    // answers only requests that take that implementation's scores.
    const auto answers_request = [](const MetricRequestSpec& spec, const MetricProvenance& provenance) {
        if (spec.backend_id == "perceptual" || compatible(spec, provenance)) {
            return true;
        }
        app_log::write(app_log::Level::warning, "vml.metric_cache",
                       spec.key + " not saved: its score is " + provenance.implementation_compatibility_id +
                           "'s, the request asks for " + spec.implementation_compatibility_id + "'s");
        return false;
    };
    const bool inexact_codec =
        !gpu_decodes_exactly(result.source_info.codec_name) || !gpu_decodes_exactly(result.distorted_info.codec_name);
    for (const MetricRequestSpec& spec : specs) {
        if (const FrameMetricResult* metric = result.metric_results.frame(spec.key)) {
            if (answers_request(spec, metric->provenance)) {
                store_metric(directory, *metric, spec, inexact_codec);
            }
        } else if (const SequenceMetricResult* sequence = result.metric_results.sequence(spec.key)) {
            if (answers_request(spec, sequence->provenance)) {
                store_metric(directory, *sequence, spec, inexact_codec);
            }
        }
    }
    return directory;
}

std::optional<std::pair<ComparisonResult, std::string>> load_result(
    const std::filesystem::path& base, const std::filesystem::path& source, const std::filesystem::path& distorted,
    const ComparisonRecipe& recipe, const std::vector<MetricRequestSpec>& specs,
    const std::map<std::string, std::string>& backends, const std::vector<MetricRequestSpec>& supplemental) {
    const std::filesystem::path directory = recipe_directory(base, source, distorted, recipe);
    const std::optional<nlohmann::json> context = read_json(directory / "context.json");
    if (!context || !context->is_object() || !current_format(*context)) {
        return std::nullopt;
    }
    MetricResultSet results;
    const auto load = [&](const MetricRequestSpec& spec) {
        const auto backend = backends.find(spec.key);
        if (std::optional<LoadedMetric> metric =
                load_metric(directory, spec, backend == backends.end() ? "gpu" : backend->second, *context)) {
            std::visit([&results](auto&& loaded) { results.add(std::move(loaded)); }, std::move(*metric));
        }
    };
    for (const MetricRequestSpec& spec : specs) {
        load(spec);
    }
    for (const MetricRequestSpec& spec : supplemental) {
        const bool asked = std::any_of(specs.begin(), specs.end(),
                                       [&spec](const MetricRequestSpec& other) { return other.key == spec.key; });
        if (!asked && !results.has(spec.key)) {
            load(spec);
        }
    }
    if (results.empty()) {
        return std::nullopt;
    }
    std::vector<MetricRequestSpec> loaded_for = specs;  // the requests a loaded score answers
    loaded_for.insert(loaded_for.end(), supplemental.begin(), supplemental.end());
    try {
        // A context saved before some of these fields were recorded lacks
        // them: the recipe's values stand in, the comparison found under the
        // recipe being the same one.
        ComparisonResult result;
        result.source = platform::path_from_text(context->at("source").get<std::string>());
        result.distorted = platform::path_from_text(context->at("distorted").get<std::string>());
        result.fps = context->at("fps").get<double>();
        const nlohmann::json model = context->value("model", nlohmann::json(""));
        result.model = model.is_string() ? model.get<std::string>() : std::string();
        result.source_crop = crop_from_dict(context->value("source_crop", nlohmann::json()));
        result.distorted_crop = crop_from_dict(context->value("distorted_crop", nlohmann::json()));
        result.source_info = video_info_from_dict(context->at("source_info"));
        result.distorted_info = video_info_from_dict(context->at("distorted_info"));
        result.scale_direction = scale_direction_from(
            context->value("scale_direction", std::string(scale_direction_value(recipe.scale_direction))));
        const nlohmann::json algorithm = context->value("scale_algorithm", nlohmann::json());
        result.scale_algorithm = algorithm.is_string() && !algorithm.get<std::string>().empty()
                                     ? algorithm.get<std::string>()
                                     : recipe.scale_algorithm;
        result.resample_target = recipe.resample_test;
        if (const nlohmann::json target = context->value("resample_target", nlohmann::json()); python_truthy(target)) {
            // As Python's ResampleTarget(**target): its two fields, no more.
            if (!target.is_object() || target.size() != 2) {
                return std::nullopt;
            }
            result.resample_target = ResampleTarget{target.at("width").get<int>(), target.at("label").get<std::string>()};
        }
        result.compared_frame_count = context->value("compared_frame_count", 0LL);
        const auto optional_string = [&context](const char* name) -> std::optional<std::string> {
            const nlohmann::json value = context->value(name, nlohmann::json());
            if (!value.is_string()) {
                return std::nullopt;
            }
            return value.get<std::string>();
        };
        result.model_choice = optional_string("model_choice");
        result.model_v1 = optional_string("model_v1").value_or("");
        result.model_choice_v1 = optional_string("model_choice_v1");
        result.metric_results = std::move(results);
        loaded_model(result.metric_results, loaded_for, "vmaf", result.model, result.model_choice);
        loaded_model(result.metric_results, loaded_for, "vmaf_v1", result.model_v1, result.model_choice_v1);
        std::optional<std::string> label = optional_string("label");
        if (!label || label->empty()) {
            label = platform::path_text(result.distorted.stem());
        }
        return std::pair{std::move(result), std::move(*label)};
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

std::optional<OtherCvvdpScores> other_cvvdp_scores(const std::filesystem::path& base,
                                                   const std::filesystem::path& source,
                                                   const std::filesystem::path& distorted,
                                                   const ComparisonRecipe& recipe,
                                                   const std::vector<MetricRequestSpec>& specs,
                                                   const std::vector<MetricRequestSpec>& supplemental) {
    const MetricRequestSpec* spec = nullptr;
    for (const std::vector<MetricRequestSpec>* list : {&specs, &supplemental}) {
        for (const MetricRequestSpec& asked : *list) {
            if (spec == nullptr && asked.key == "cvvdp") {
                spec = &asked;
            }
        }
    }
    if (spec == nullptr) {
        return std::nullopt;
    }
    OtherCvvdpScores found{spec->parameters, {}};
    for (const auto& [parameters, score] : other_parameters(recipe_directory(base, source, distorted, recipe), *spec)) {
        try {
            found.scores.emplace_back(CvvdpSettings::from_parameters(parameters), score);
        } catch (const std::invalid_argument&) {
            continue;  // a display CVVDP cannot model, as a hand-edited file can say
        } catch (const nlohmann::json::exception&) {
            continue;
        }
    }
    return found;
}

int clear_metrics(const std::filesystem::path& base, const std::filesystem::path& source,
                  const std::filesystem::path& distorted, const ComparisonRecipe& recipe,
                  const std::vector<MetricRequestSpec>& specs) {
    const std::filesystem::path directory = recipe_directory(base, source, distorted, recipe);
    int removed = 0;
    const auto remove = [&removed](const std::filesystem::path& path) {
        std::error_code error;
        if (std::filesystem::remove(path, error)) {
            ++removed;
        }
    };
    for (const MetricRequestSpec& spec : specs) {
        if (is_auto_perceptual(spec)) {
            for (const auto& [path, concrete] : auto_perceptual_candidates(directory, spec)) {
                remove(path);
            }
            continue;
        }
        if (spec.key == "vmaf" && spec.backend_id == "ffmpeg") {
            for (const auto& [path, concrete] : vmaf_equivalents(directory, spec)) {
                remove(path);
            }
        } else if (spec.key == "vmaf_v1" && spec.backend_id == "ffmpeg") {
            for (const auto& [path, concrete] : old_v1_scores(directory, spec)) {
                remove(path);
            }
        }
        remove(metric_path(directory, spec));
    }
    // The folder goes with its last score.
    std::error_code error;
    bool scores_left = false;
    for (std::filesystem::directory_iterator entry(directory, error);
         !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        scores_left = scores_left || entry->path().extension() == ".npz";
    }
    if (!error && !scores_left) {
        std::filesystem::remove_all(directory, error);
    }
    return removed;
}

std::string lookup_token(const std::filesystem::path& source, const std::filesystem::path& distorted,
                         const AnalysisRequest& request) {
    nlohmann::json metrics = nlohmann::json::array();
    for (const MetricRequestSpec& spec : request.metrics) {
        metrics.push_back(spec.identity());
    }
    return sha256_hex(
        nlohmann::json{{"recipe", recipe_hash(source, distorted, request.recipe)}, {"metrics", metrics}}.dump());
}

CacheSummary cache_summary(const std::filesystem::path& base) {
    // Walked by directory entries, which come with their sizes on Windows:
    // asking each file for its size takes about ten times as long.
    CacheSummary summary;
    std::error_code error;
    std::filesystem::recursive_directory_iterator walk(
        base / format_folder, std::filesystem::directory_options::skip_permission_denied, error);
    for (; !error && walk != std::filesystem::recursive_directory_iterator(); walk.increment(error)) {
        std::error_code ignored;
        if (!walk->is_regular_file(ignored)) {
            continue;
        }
        if (const std::uintmax_t size = walk->file_size(ignored); !ignored) {
            summary.bytes += size;
        }
        // A comparison's folder holds its context.json (<base>/v2/<hash>).
        if (walk.depth() == 1 && walk->path().filename() == "context.json") {
            ++summary.comparisons;
        }
    }
    return summary;
}

int clear_all(const std::filesystem::path& base) {
    const std::filesystem::path root = base / format_folder;
    std::error_code error;
    if (!std::filesystem::exists(root, error)) {
        return 0;
    }
    int count = 0;
    // increment(error), not a range-for: its ++ throws if reading fails.
    for (std::filesystem::directory_iterator entry(root, error);
         !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        std::error_code ignored;
        if (std::filesystem::is_regular_file(entry->path() / "context.json", ignored)) {
            ++count;
        }
    }
    std::filesystem::remove_all(root, error);
    return error ? 0 : count;
}

}  // namespace vml
