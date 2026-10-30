#include "vml/analysis_request.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <system_error>

#include "vml/film_grain_scoring.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/score_identity.h"
#include "vml/vship.h"

namespace vml {

namespace {

// A custom VMAF model file's identity, as the Python app files a request
// under it: its resolved path, size and modification time, or the path
// alone when the file cannot be read.
std::string custom_model_identity(const std::string& path) {
    std::error_code error;
    const std::filesystem::path given = platform::path_from_text(path);
    const std::filesystem::path absolute = std::filesystem::absolute(given, error);
    std::filesystem::path resolved = std::filesystem::weakly_canonical(error ? given : absolute, error);
    if (error) {
        resolved = absolute;
    }
    const auto size = std::filesystem::file_size(resolved, error);
    const std::optional<long long> modified = platform::modified_ns(resolved);
    if (error || !modified) {
        return platform::path_text(resolved);
    }
    return platform::path_text(resolved) + ":" + std::to_string(size) + ":" + std::to_string(*modified);
}

}  // namespace

nlohmann::json ComparisonRecipe::identity() const {
    nlohmann::json identity{
        {"crop_mode", crop_mode == CropMode::automatic ? "auto" : "none"},
        {"scale_direction",
         scale_direction == ScaleDirection::source_to_distorted ? "source_to_distorted" : "distorted_to_source"},
        {"duration_limit", duration_limit},
        {"resample_test", resample_test ? nlohmann::json{{"width", resample_test->width},
                                                         {"label", resample_test->label}}
                                        : nlohmann::json(nullptr)},
        {"manual_source_crop", nullptr},
        {"manual_distorted_crop", nullptr}};
    if (!film_grain.is_null()) {
        identity["film_grain"] = film_grain;
    }
    return identity;
}

nlohmann::json MetricRequestSpec::identity() const {
    return {{"key", key},
            {"parameters", parameters},
            {"coverage", coverage ? nlohmann::json{{"mode", coverage->mode}, {"step", coverage->step}} : nullptr},
            {"implementation_compatibility_id", implementation_compatibility_id}};
}

const MetricRequestSpec* AnalysisRequest::metric(const std::string& key) const {
    const auto found = std::find_if(metrics.begin(), metrics.end(), [&key](const auto& spec) { return spec.key == key; });
    return found == metrics.end() ? nullptr : &*found;
}

AnalysisRequest analysis_request(const VmafOptions& options, const std::vector<std::string>& metric_keys,
                                 const std::map<std::string, std::string>& perceptual_backends,
                                 const CvvdpSettings& cvvdp, const VideoInfo& source, const VideoInfo& distorted) {
    AnalysisRequest request;
    request.recipe = {options.crop_mode, options.scale_algorithm, options.scale_direction, options.duration_limit,
                      options.resample_test};
    const std::optional<std::string> source_grain = film_grain_identity(source);
    const std::optional<std::string> distorted_grain = film_grain_identity(distorted);
    if (source_grain || distorted_grain) {
        const auto text = [](const std::optional<std::string>& grain) {
            return grain ? nlohmann::json(*grain) : nlohmann::json(nullptr);
        };
        request.recipe.film_grain = {{"source", text(source_grain)}, {"distorted", text(distorted_grain)}};
    }
    request.perceptual_backends = automatic_perceptual_backends;
    for (const auto& [key, backend] : perceptual_backends) {
        if (!request.perceptual_backends.contains(key)) {
            throw std::invalid_argument("Unknown perceptual metric backend setting: '" + key + "'");
        }
        if (backend != "gpu" && backend != "cpu") {
            throw std::invalid_argument("Unsupported " + key + " compute backend: '" + backend + "'");
        }
        request.perceptual_backends[key] = backend;
    }
    // libvmaf's metrics share its subsampling; XPSNR alone is scored every
    // frame by its own filter.
    const bool libvmaf_requested = std::any_of(metric_keys.begin(), metric_keys.end(), [](const std::string& key) {
        return metric_definition(key).ffmpeg_binding.has_value() && key != "xpsnr";
    });
    const bool subsampled = options.n_subsample > 1;
    for (const std::string& key : metric_keys) {
        const MetricDefinition& definition = metric_definition(key);
        MetricRequestSpec spec;
        spec.key = key;
        if (!definition.ffmpeg_binding) {
            if (definition.backend_id.empty()) {
                throw std::invalid_argument("No executable backend is registered for metric '" + key + "'");
            }
            spec.backend_id = std::string(definition.backend_id);
            // Each Vship build's scores apart (score_identity.h), by the build
            // that calculates here: 5.1.2's where Vship-fast can use no GPU.
            const bool vship51 = vship_build_in_use(options.vship_build) == VshipBuild::stock;
            if (key == "cvvdp") {
                // GPU only (Vship), and every frame: the display and whether
                // frames fill it are part of what the score means.
                spec.parameters = cvvdp.parameters();
                spec.coverage = FrameCoverage{"full", 1};
                spec.implementation_compatibility_id =
                    std::string(vship51 ? score_identity::vship51_cvvdp : score_identity::cvvdp);
            } else {
                spec.coverage = FrameCoverage{subsampled ? "sampled" : "full", options.n_subsample};
                spec.implementation_compatibility_id =
                    vship51 ? score_identity::vship51_gpu_or_cpu(key) : score_identity::gpu_or_cpu(key);
            }
            request.metrics.push_back(std::move(spec));
            continue;
        }
        spec.backend_id = "ffmpeg";
        spec.coverage = libvmaf_requested && subsampled ? FrameCoverage{"sampled", options.n_subsample}
                                                        : FrameCoverage{"full", 1};
        if (key == "vmaf") {
            // Only what decides the score: an explicit or bundled choice is
            // the model that runs, while Auto's model is known only once the
            // size compared at is, and is recorded with the score instead.
            const std::string& choice = options.model_choice;
            const bool chosen_by_the_run = choice == auto_model_choice || choice == custom_model_choice;
            spec.parameters = {{"model", chosen_by_the_run ? std::string() : choice},
                               {"model_choice", choice},
                               {"custom_model", options.custom_model_path.empty()
                                                    ? std::string()
                                                    : custom_model_identity(options.custom_model_path)}};
        } else if (key == "vmaf_v1") {
            const std::string& choice = options.model_choice_v1;
            spec.parameters = {{"model", choice == auto_model_choice ? std::string() : choice}, {"model_choice", choice}};
        } else if (key == "vmaf_neg") {
            spec.parameters = {{"model", "version=vmaf_v0.6.1neg"}};
        }
        spec.implementation_compatibility_id =
            std::string(key == "xpsnr" ? score_identity::xpsnr : score_identity::libvmaf);
        request.metrics.push_back(std::move(spec));
    }
    return request;
}

std::vector<MetricRequestSpec> displayable_metric_specs(const VmafOptions& options, const CvvdpSettings& cvvdp) {
    std::vector<std::string> ffmpeg;      // every FFmpeg metric, in the registry's order
    std::vector<std::string> perceptual;  // SSIMULACRA2, Butteraugli, CVVDP
    for (const MetricDefinition& metric : metrics()) {
        if (metric.ffmpeg_binding && metric.kind == MetricKind::frame) {
            ffmpeg.emplace_back(metric.key);
        }
        if (metric.backend_id == "perceptual") {
            perceptual.emplace_back(metric.key);
        }
    }
    std::vector<MetricRequestSpec> specs;
    std::set<std::string> found;  // each identity once, the first time it comes
    for (const std::vector<std::string>& keys : {ffmpeg, std::vector<std::string>{"xpsnr"}, perceptual}) {
        // Only the metrics' requests, which the videos' film grain is not part of.
        for (MetricRequestSpec& spec : analysis_request(options, keys, {}, cvvdp, VideoInfo{}, VideoInfo{}).metrics) {
            if (found.insert(spec.identity().dump()).second) {
                specs.push_back(std::move(spec));
            }
        }
    }
    return specs;
}

std::string file_identity(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(path, error);
    std::filesystem::path resolved = std::filesystem::weakly_canonical(error ? path : absolute, error);
    if (error) {
        resolved = absolute;
    }
    const auto size = std::filesystem::file_size(resolved, error);
    const std::optional<long long> modified = platform::modified_ns(resolved);
    const bool read = !error && modified;
    return platform::path_text(resolved) + ":" + (read ? std::to_string(size) : "-1") + ":" +
           (read ? std::to_string(*modified) : "-1");
}

}  // namespace vml
