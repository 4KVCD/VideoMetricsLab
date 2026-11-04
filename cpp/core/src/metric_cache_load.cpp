// Loading a request's saved scores (metric_cache.h's load_result): its own
// comparison's, a range's cut from a run over the whole videos, Source
// only's from Auto's where Auto compared the same pictures, the better of
// two where both have one; and other displays' CVVDP scores.
#include "vml/metric_cache.h"

#include "metric_cache_parts.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <variant>

#include "vml/append.h"
#include "vml/frame_range.h"
#include "vml/frame_sync.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/result_json.h"

namespace vml {

namespace {

using namespace cache_parts;

// A duration limit's frame score, saved before ranges, as the range's [0,
// end): its frames past the range left out, and used only if it reaches the
// range's end or the videos' (`video_frames`), with, for a metric that looks
// at the next frame (VMAF's motion), the frame after its last.
bool fit_frames(FrameMetricResult& metric, long long end, long long video_frames, bool looks_ahead) {
    const auto past = std::lower_bound(metric.frame.begin(), metric.frame.end(), end);
    const auto kept = static_cast<std::size_t>(past - metric.frame.begin());
    const long long step = metric.frame.size() > 1 ? metric.frame[1] - metric.frame[0] : 1;
    const long long reached = kept == 0 ? 0 : metric.frame[kept - 1] + step;
    if (reached < std::min(end, video_frames) || (looks_ahead && past == metric.frame.end() && end < video_frames)) {
        return false;
    }
    metric.frame.resize(kept);
    metric.time.resize(std::min(kept, metric.time.size()));
    metric.values.resize(std::min(kept, metric.values.size()));
    return true;
}

// A duration limit's CVVDP, saved before ranges, as the range's [0, end):
// its seconds before `end` pooled again, their JODs as saved (as a ranged
// run pools them), if its run reached `end`: it decoded the frames within
// the limit at the test video's rate (vship_run.cpp before ranges).
bool fit_seconds(SequenceMetricResult& metric, long long end, long long video_frames, const VideoInfo& distorted,
                 double limit) {
    const std::optional<long long> decoded = frames_within(format_fixed(limit, 6), frame_rate(distorted));
    const long long reached = std::min(decoded.value_or(0), video_frames);
    if (!metric.has_timeline() || reached < std::min(end, video_frames)) {
        return false;
    }
    std::vector<CvvdpWindow> windows;
    for (std::size_t at = 0; at < metric.frame.size() && metric.frame[at] < end; ++at) {
        const long long next = at + 1 < metric.frame.size() ? metric.frame[at + 1] : reached;
        windows.push_back({metric.frame[at], std::min({next, end, reached}) - metric.frame[at], metric.values[at]});
    }
    if (windows.empty()) {
        return false;
    }
    metric.score = pool_cvvdp_windows(windows);
    metric.frame.resize(windows.size());
    metric.time.resize(std::min(windows.size(), metric.time.size()));
    metric.values.resize(windows.size());
    return true;
}

// A saved score as the score of `recipe`'s range (frame_range.h). A ranged
// run's (range_frames) is the range's already, and never the whole videos'. A
// duration limit's, saved before ranges, under the same name (the range from
// 0 to the limit), had the frames up to the limit and for VMAF one more:
// fit_frames, fit_seconds.
bool fit_to_range(LoadedMetric& metric, const ComparisonRecipe& recipe, const nlohmann::json& context) {
    const bool marked = provenance_of(metric).parameters.contains("range_frames");
    if (recipe.range_start <= 0 && recipe.range_end <= 0) {
        return !marked;
    }
    if (recipe.range_end <= 0 || marked) {
        return true;  // the range's own
    }
    if (recipe.range_start > 0) {
        return false;  // no score saved before ranges is of it
    }
    try {
        const VideoInfo source = video_info_from_dict(context.at("source_info"));
        const VideoInfo distorted = video_info_from_dict(context.at("distorted_info"));
        const long long end = first_frame_of_second(recipe.range_end, source.fps);
        const long long video_frames = std::min(source.estimated_frame_count(), distorted.estimated_frame_count());
        bool fits = false;
        if (auto* frames = std::get_if<FrameMetricResult>(&metric)) {
            const bool looks_ahead = frames->key == "vmaf" || frames->key == "vmaf_neg" || frames->key == "vmaf_v1";
            fits = fit_frames(*frames, end, video_frames, looks_ahead);
        } else {
            fits = fit_seconds(std::get<SequenceMetricResult>(metric), end, video_frames, distorted,
                               static_cast<double>(recipe.range_end));
        }
        if (fits) {
            // As the range's own now: saved again with a run's other scores, it loads as one.
            std::visit([end](auto& fitted) { mark_range(fitted.provenance, FrameRange{0, end, 0, end}); }, metric);
        }
        return fits;
    } catch (const nlohmann::json::exception&) {
        return false;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

// load_metric, the score fitted to the recipe's range (fit_to_range).
std::optional<LoadedMetric> load_in_range(const std::filesystem::path& directory, const MetricRequestSpec& spec,
                                          const std::string& backend, const nlohmann::json& context,
                                          const ComparisonRecipe& recipe) {
    std::optional<LoadedMetric> metric = load_metric(directory, spec, backend, context);
    if (metric && !fit_to_range(*metric, recipe, context)) {
        return std::nullopt;
    }
    return metric;
}

// A whole run's frame score as a ranged run of [start, end) scores it (end
// none: to the end): the same frames' scores, as a ranged run decodes the
// frames its metrics look at around them (frame_range.h). Not a subsampled
// score: kept for the range to calculate (subsampled runs were seen pairing
// frames wrongly under load, 2026-11-02, so their cut is not yet trusted).
bool cut_frames(FrameMetricResult& metric, long long start, std::optional<long long> end) {
    if (metric.frame.empty() || metric.frame.front() > start ||
        (metric.frame.size() > 1 && metric.frame[1] - metric.frame[0] != 1)) {
        return false;
    }
    const auto first = std::lower_bound(metric.frame.begin(), metric.frame.end(), start);
    const auto past = end ? std::lower_bound(first, metric.frame.end(), *end) : metric.frame.end();
    const auto from = static_cast<std::size_t>(first - metric.frame.begin());
    const auto to = static_cast<std::size_t>(past - metric.frame.begin());
    if (from == to || metric.time.size() < to || metric.values.size() < to) {
        return false;
    }
    const auto keep = [from, to](auto& values) {
        values = std::vector(values.begin() + static_cast<std::ptrdiff_t>(from),
                             values.begin() + static_cast<std::ptrdiff_t>(to));
    };
    keep(metric.frame);
    keep(metric.time);
    keep(metric.values);
    return true;
}

// A whole run's CVVDP as a ranged run of [start, end) scores it: the seconds
// that start in the range, pooled again from their JODs as saved (float), as
// a ranged run pools them (vship_run.cpp's cvvdp_result), each second up to
// the next one's first frame or the range's end. The whole run's last second
// ends where its run reached (cvvdp_frames_reached, saved since 2026-11-02);
// without it, a range with that second is not cut. The first frame past the
// seconds kept, or nothing when they cannot be cut.
std::optional<long long> cut_seconds(SequenceMetricResult& metric, long long start, std::optional<long long> end) {
    if (!metric.has_timeline() || metric.values.size() != metric.frame.size() ||
        metric.time.size() != metric.frame.size()) {
        return std::nullopt;
    }
    std::optional<long long> reached;
    if (const std::optional<nlohmann::json> saved = parameter(metric.provenance, "cvvdp_frames_reached");
        saved && saved->is_number_integer()) {
        reached = saved->get<long long>();
    }
    std::vector<CvvdpWindow> windows;
    std::size_t first = metric.frame.size();
    for (std::size_t at = 0; at < metric.frame.size(); ++at) {
        if (metric.frame[at] < start) {
            continue;
        }
        if (end && metric.frame[at] >= *end) {
            break;
        }
        const std::optional<long long> next =
            at + 1 < metric.frame.size() ? std::optional<long long>(metric.frame[at + 1]) : reached;
        if (!next) {
            return std::nullopt;  // the whole run's last second, of a length not saved
        }
        first = std::min(first, at);
        const long long stop = end ? std::min(*next, *end) : *next;
        windows.push_back({metric.frame[at], stop - metric.frame[at], metric.values[at]});
    }
    if (windows.empty()) {
        return std::nullopt;
    }
    metric.score = pool_cvvdp_windows(windows);
    const auto keep = [first, count = windows.size()](auto& values) {
        values = std::vector(values.begin() + static_cast<std::ptrdiff_t>(first),
                             values.begin() + static_cast<std::ptrdiff_t>(first + count));
    };
    keep(metric.frame);
    keep(metric.time);
    keep(metric.values);
    metric.provenance.parameters.erase("cvvdp_frames_reached");  // a whole run's, which the cut is not
    return windows.back().first + windows.back().frames;
}

// A whole run's score (`metric`, of the whole videos' folder with
// `whole_context`) cut to `recipe`'s range, marked as the range's own
// (range_frames): saved again with a run's other scores, it loads as one.
// False where it cannot stand for the range: not a whole run's, no frame
// rate to count the range's frames by, or nothing to cut (see above).
bool cut_to_range(LoadedMetric& metric, const ComparisonRecipe& recipe, const nlohmann::json& whole_context) {
    if (provenance_of(metric).parameters.contains("range_frames")) {
        return false;
    }
    try {
        const double fps = video_info_from_dict(whole_context.at("source_info")).fps;
        if (!std::isfinite(fps) || fps <= 0.0) {
            return false;
        }
        const long long start = first_frame_of_second(recipe.range_start, fps);
        const std::optional<long long> end =
            recipe.range_end > 0 ? std::optional(first_frame_of_second(recipe.range_end, fps)) : std::nullopt;
        const FrameRange range{start, end, start, end};
        if (auto* frames = std::get_if<FrameMetricResult>(&metric)) {
            if (!cut_frames(*frames, start, end)) {
                return false;
            }
            mark_range(frames->provenance, range);
            return true;
        }
        auto& seconds = std::get<SequenceMetricResult>(metric);
        const std::optional<long long> reached = cut_seconds(seconds, start, end);
        if (!reached) {
            return false;
        }
        mark_range(seconds.provenance, range, reached);
        return true;
    } catch (const nlohmann::json::exception&) {
        return false;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

// A folder's context.json, when it is one this app reads.
std::optional<nlohmann::json> current_context(const std::filesystem::path& directory) {
    std::optional<nlohmann::json> context = read_json(directory / "context.json");
    if (context && (!context->is_object() || !current_format(*context))) {
        return std::nullopt;
    }
    return context;
}

// A recipe's folder found for another recipe's lookup, with its context.json.
struct OtherRecipeFolder {
    std::filesystem::path directory;
    nlohmann::json context;
};

// Black bars "Source only": Auto's saved scores of the same comparison,
// where they are Source only's. They are when Auto
// compared the test video at its whole frame: every route treats a box of
// the whole frame as no box (CropBox::is_noop), and common_picture then
// leaves the source's box as detected, the box Source only cuts. A
// resolution test runs the same code in both modes (both its crops are the
// source's). Where this recipe saved a comparison of its own, Auto's must
// have cut the same source box. The folder by its hash, not recipe_directory:
// a lookup must not move Auto's legacy folder while a save may write to it.
std::optional<OtherRecipeFolder> auto_folder_for_source_only(const std::filesystem::path& base,
                                                             const std::filesystem::path& source,
                                                             const std::filesystem::path& distorted,
                                                             const ComparisonRecipe& recipe,
                                                             const std::optional<nlohmann::json>& own_context) {
    if (recipe.crop_mode != CropMode::source_only) {
        return std::nullopt;
    }
    ComparisonRecipe automatic = recipe;
    automatic.crop_mode = CropMode::automatic;
    std::filesystem::path directory = base / format_folder / recipe_hash(source, distorted, automatic);
    std::optional<nlohmann::json> context = current_context(directory);
    if (!context) {
        return std::nullopt;
    }
    try {
        const std::optional<CropBox> source_crop = crop_from_dict(context->value("source_crop", nlohmann::json()));
        if (own_context && crop_from_dict(own_context->value("source_crop", nlohmann::json())) != source_crop) {
            return std::nullopt;
        }
        if (!recipe.resample_test) {
            const std::optional<CropBox> test_crop = crop_from_dict(context->value("distorted_crop", nlohmann::json()));
            const auto [width, height] = video_info_from_dict(context->at("distorted_info")).frame_size();
            if (test_crop && !test_crop->is_noop(width, height)) {
                return std::nullopt;
            }
        }
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
    return OtherRecipeFolder{std::move(directory), std::move(*context)};
}

// Whether two contexts are of one comparison: the same black bars cut and
// the same resolution test, so that one's scores stand for the other's.
bool same_crops(const nlohmann::json& context, const nlohmann::json& other) {
    for (const char* field : {"source_crop", "distorted_crop", "resample_target"}) {
        if (context.value(field, nlohmann::json()) != other.value(field, nlohmann::json())) {
            return false;
        }
    }
    return true;
}

// A range's comparison over the whole videos: its folder (by its hash, as
// auto_folder_for_source_only finds Auto's) and context, whose scores are cut
// to the range (cut_to_range). Nothing for a recipe of the whole videos, or
// where the range saved a comparison of its own with other black bars (an
// older crop detection's whole-video scores).
std::optional<OtherRecipeFolder> whole_video_folder(const std::filesystem::path& base,
                                                    const std::filesystem::path& source,
                                                    const std::filesystem::path& distorted,
                                                    const ComparisonRecipe& recipe,
                                                    const std::optional<nlohmann::json>& own_context) {
    if (recipe.range_start <= 0 && recipe.range_end <= 0) {
        return std::nullopt;
    }
    ComparisonRecipe whole = recipe;
    whole.range_start = 0;
    whole.range_end = 0;
    std::filesystem::path directory = base / format_folder / recipe_hash(source, distorted, whole);
    std::optional<nlohmann::json> context = current_context(directory);
    if (!context || (own_context && !same_crops(*own_context, *context))) {
        return std::nullopt;
    }
    return OtherRecipeFolder{std::move(directory), std::move(*context)};
}

// The whole videos' recipe of a range, for auto_folder_for_source_only.
ComparisonRecipe whole_video_recipe(ComparisonRecipe recipe) {
    recipe.range_start = 0;
    recipe.range_end = 0;
    return recipe;
}

// Whether a SSIMULACRA2 or Butteraugli score is Vship's, as
// auto_perceptual_candidates ranks them.
bool made_by_vship(const LoadedMetric& metric) { return is_vship_id(provenance_of(metric).implementation_compatibility_id); }

// The score found so far for `spec` (`own`), or another folder's
// (`load_other`, loaded only when needed) where there is none so far, or
// where it is the CPU's and the other's is Vship's for SSIMULACRA2 or
// Butteraugli set to the GPU: one folder's candidates rank Vship's first
// (auto_perceptual_candidates), and so do several.
std::optional<LoadedMetric> better_of(std::optional<LoadedMetric> own,
                                      const std::function<std::optional<LoadedMetric>()>& load_other,
                                      const MetricRequestSpec& spec, const std::string& backend) {
    const bool ranked = is_auto_perceptual(spec) && backend == "gpu";
    if (own && (!ranked || made_by_vship(*own))) {
        return own;
    }
    std::optional<LoadedMetric> other = load_other();
    if (other && (!own || made_by_vship(*other))) {
        return other;
    }
    return own;
}

// The comparison a recipe's folder holds, from its context.json, with the
// scores loaded from it (`results`, for the requests `loaded_for`), and its
// label; nothing when the context cannot be read.
std::optional<std::pair<ComparisonResult, std::string>> comparison_from(
    const nlohmann::json& context, const ComparisonRecipe& recipe, MetricResultSet results,
    const std::vector<MetricRequestSpec>& loaded_for) {
    try {
        // A context saved before some of these fields were recorded lacks
        // them: the recipe's values stand in, the comparison found under the
        // recipe being the same one.
        ComparisonResult result;
        result.source = platform::path_from_text(context.at("source").get<std::string>());
        result.distorted = platform::path_from_text(context.at("distorted").get<std::string>());
        result.fps = context.at("fps").get<double>();
        const nlohmann::json model = context.value("model", nlohmann::json(""));
        result.model = model.is_string() ? model.get<std::string>() : std::string();
        result.source_crop = crop_from_dict(context.value("source_crop", nlohmann::json()));
        result.distorted_crop = crop_from_dict(context.value("distorted_crop", nlohmann::json()));
        result.source_info = video_info_from_dict(context.at("source_info"));
        result.distorted_info = video_info_from_dict(context.at("distorted_info"));
        result.scale_direction = scale_direction_from(
            context.value("scale_direction", std::string(scale_direction_value(recipe.scale_direction))));
        const nlohmann::json algorithm = context.value("scale_algorithm", nlohmann::json());
        result.scale_algorithm = algorithm.is_string() && !algorithm.get<std::string>().empty()
                                     ? algorithm.get<std::string>()
                                     : recipe.scale_algorithm;
        result.resample_target = recipe.resample_test;
        if (const nlohmann::json target = context.value("resample_target", nlohmann::json()); python_truthy(target)) {
            result.resample_target = resample_target_from_dict(target);
        }
        if (recipe.resample_test && algorithm.is_string() && !algorithm.get<std::string>().empty() &&
            algorithm.get<std::string>() != recipe.resample_test->scaler) {
            // A test saved before tests had a scaling algorithm of their own
            // was made with the Options one, under the default's identity:
            // another algorithm made another test video.
            return std::nullopt;
        }
        result.compared_frame_count = context.value("compared_frame_count", 0LL);
        const auto optional_string = [&context](const char* name) -> std::optional<std::string> {
            const nlohmann::json value = context.value(name, nlohmann::json());
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

// The frames a range compares, as a run of it counts them: those of
// [start, end) that both videos have (a comparison over the whole videos
// counts them all).
long long range_frame_count(const ComparisonResult& result, const ComparisonRecipe& recipe) {
    const double fps = result.source_info.fps;
    const long long video_frames =
        std::min(result.source_info.estimated_frame_count(), result.distorted_info.estimated_frame_count());
    const long long start = first_frame_of_second(recipe.range_start, fps);
    const long long end =
        recipe.range_end > 0 ? std::min(first_frame_of_second(recipe.range_end, fps), video_frames) : video_frames;
    return std::max(0LL, end - start);
}

}  // namespace

std::optional<std::pair<ComparisonResult, std::string>> load_result(
    const std::filesystem::path& base, const std::filesystem::path& source, const std::filesystem::path& distorted,
    const ComparisonRecipe& recipe, const std::vector<MetricRequestSpec>& specs,
    const std::map<std::string, std::string>& backends, const std::vector<MetricRequestSpec>& supplemental,
    const std::set<std::string>& not_from_other_recipes) {
    const std::filesystem::path directory = recipe_directory(base, source, distorted, recipe);
    const std::optional<nlohmann::json> context = current_context(directory);  // none: its scores are not read
    const std::optional<OtherRecipeFolder> automatic = auto_folder_for_source_only(base, source, distorted, recipe, context);
    // A range: the whole videos' scores cut to it, where it has none of its
    // own (and, for Source only, Auto's over the whole videos).
    const std::optional<OtherRecipeFolder> whole = whole_video_folder(base, source, distorted, recipe, context);
    const std::optional<OtherRecipeFolder> automatic_whole =
        recipe.range_start > 0 || recipe.range_end > 0
            ? auto_folder_for_source_only(base, source, distorted, whole_video_recipe(recipe),
                                          whole ? std::optional(whole->context) : context)
            : std::nullopt;
    if (!context && !automatic && !whole && !automatic_whole) {
        return std::nullopt;
    }
    MetricResultSet results;
    const auto load = [&](const MetricRequestSpec& spec) {
        const auto chosen = backends.find(spec.key);
        const std::string backend = chosen == backends.end() ? "gpu" : chosen->second;
        std::optional<LoadedMetric> metric;
        if (context) {
            metric = load_in_range(directory, spec, backend, *context, recipe);
        }
        if (not_from_other_recipes.contains(spec.key)) {
            // A Recalculate: not answered by the scores it means to replace.
        } else {
            if (automatic) {
                metric = better_of(std::move(metric),
                                   [&] { return load_in_range(automatic->directory, spec, backend, automatic->context,
                                                              recipe); },
                                   spec, backend);
            }
            for (const std::optional<OtherRecipeFolder>* folder : {&whole, &automatic_whole}) {
                if (*folder) {
                    const OtherRecipeFolder& found = **folder;
                    metric = better_of(std::move(metric),
                                       [&]() -> std::optional<LoadedMetric> {
                                           std::optional<LoadedMetric> cut =
                                               load_metric(found.directory, spec, backend, found.context);
                                           if (cut && !cut_to_range(*cut, recipe, found.context)) {
                                               return std::nullopt;
                                           }
                                           return cut;
                                       },
                                       spec, backend);
                }
            }
        }
        if (metric) {
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
    append(loaded_for, supplemental);
    if (context) {
        return comparison_from(*context, recipe, std::move(results), loaded_for);
    }
    // This recipe's folder has no context this app reads: another recipe's
    // comparison. Auto's as a Source-only run gives it (the test video at its
    // full frame, which Auto's box is; a resolution test's crops are both the
    // source's in either mode); the whole videos' with the range's frames.
    const bool from_auto = automatic || !whole;
    const OtherRecipeFolder& other = automatic ? *automatic : whole ? *whole : *automatic_whole;
    auto found = comparison_from(other.context, recipe, std::move(results), loaded_for);
    if (found && from_auto && !recipe.resample_test) {
        found->first.distorted_crop.reset();
    }
    if (found && !automatic) {
        found->first.compared_frame_count = range_frame_count(found->first, recipe);
    }
    return found;
}

std::optional<OtherCvvdpScores> other_cvvdp_scores(const std::filesystem::path& base,
                                                   const std::filesystem::path& source,
                                                   const std::filesystem::path& distorted,
                                                   const ComparisonRecipe& recipe,
                                                   const std::vector<MetricRequestSpec>& specs,
                                                   const std::vector<MetricRequestSpec>& supplemental,
                                                   const std::set<std::string>& not_from_other_recipes) {
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
    const std::filesystem::path directory = recipe_directory(base, source, distorted, recipe);
    const bool ranged = recipe.range_end > 0;
    std::vector<std::pair<nlohmann::json, double>> saved = other_parameters(directory, *spec, ranged);
    const bool other_recipes = recipe.crop_mode == CropMode::source_only && !not_from_other_recipes.contains("cvvdp");
    const auto automatic = other_recipes ? auto_folder_for_source_only(base, source, distorted, recipe,
                                                                       current_context(directory))
                                         : std::nullopt;
    if (automatic) {
        // Auto's other displays too, where this recipe has none of its own:
        // load_result would take them (not after a Recalculate of CVVDP).
        for (auto& [parameters, score] : other_parameters(automatic->directory, *spec, ranged)) {
            const bool known = std::any_of(saved.begin(), saved.end(),
                                           [&parameters](const auto& own) { return own.first == parameters; });
            if (!known) {
                saved.emplace_back(std::move(parameters), score);
            }
        }
    }
    OtherCvvdpScores found{spec->parameters, {}};
    for (const auto& [parameters, score] : saved) {
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

}  // namespace vml
