#include "vml/options.h"

#include <algorithm>
#include <stdexcept>

#include "vml/metrics.h"

namespace vml {

namespace {

// The option that switches an on/off metric on, by its name in the Python
// app's VmafOptions (the registry's bool_option).
bool VmafOptions::*bool_option(std::string_view name) {
    if (name == "compute_vmaf") {
        return &VmafOptions::compute_vmaf;
    }
    if (name == "compute_vmaf_neg") {
        return &VmafOptions::compute_vmaf_neg;
    }
    if (name == "compute_vmaf_v1") {
        return &VmafOptions::compute_vmaf_v1;
    }
    if (name == "compute_xpsnr") {
        return &VmafOptions::compute_xpsnr;
    }
    throw std::logic_error("no option called " + std::string(name));
}

}  // namespace

std::filesystem::path synthetic_scale_direction_variant_path(const std::filesystem::path& distorted,
                                                             ScaleDirection direction) {
    std::filesystem::path name = distorted.stem();
    name += direction == ScaleDirection::distorted_to_source ? " [upscale-distorted-to-source]"
                                                            : " [downscale-source-to-distorted]";
    name += distorted.extension();
    return distorted.parent_path() / name;
}

const std::vector<ResampleTarget>& resample_target_choices() {
    static const std::vector<ResampleTarget> choices{
        {2560, "1440p"},
        {1920, "1080p"},
        {1280, "720p"},
        {854, "480p"},
    };
    return choices;
}

std::vector<std::string> VmafOptions::requested_metrics() const {
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {
        if (metric.kind == MetricKind::frame && metric.ffmpeg_binding && metric_enabled(metric.key)) {
            keys.emplace_back(metric.key);
        }
    }
    return keys;
}

bool VmafOptions::metric_enabled(std::string_view key) const {
    const MetricDefinition& metric = metric_definition(key);
    if (!metric.ffmpeg_binding) {
        return false;
    }
    if (!metric.ffmpeg_binding->bool_option.empty()) {
        return this->*bool_option(metric.ffmpeg_binding->bool_option);
    }
    return std::find(extra_features.begin(), extra_features.end(), metric.ffmpeg_binding->libvmaf_feature) !=
           extra_features.end();
}

void VmafOptions::set_metric_enabled(std::string_view key, bool enabled) {
    const MetricDefinition& metric = metric_definition(key);
    if (!metric.ffmpeg_binding) {
        throw std::invalid_argument("metric '" + std::string(key) + "' is not provided by the FFmpeg options backend");
    }
    if (!metric.ffmpeg_binding->bool_option.empty()) {
        this->*bool_option(metric.ffmpeg_binding->bool_option) = enabled;
        return;
    }
    const std::string feature(metric.ffmpeg_binding->libvmaf_feature);
    if (!enabled) {
        extra_features.erase(std::remove(extra_features.begin(), extra_features.end(), feature), extra_features.end());
    } else if (std::find(extra_features.begin(), extra_features.end(), feature) == extra_features.end()) {
        extra_features.push_back(feature);
    }
}

}  // namespace vml
