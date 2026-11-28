#include "vml/options.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>

#include "vml/metrics.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

// The options that switch an on/off metric on, by the names the registry
// gives them (FfmpegMetricBinding::bool_option): the metrics' (as
// switched_metrics), then the planes' (MetricDefinition::plane_of).
constexpr std::array<std::pair<std::string_view, bool VmafOptions::*>, 29> bool_options{{
    {"compute_vmaf", &VmafOptions::compute_vmaf},
    {"compute_vmaf_neg", &VmafOptions::compute_vmaf_neg},
    {"compute_vmaf_v1", &VmafOptions::compute_vmaf_v1},
    {"compute_vmaf_hdr", &VmafOptions::compute_vmaf_hdr},
    {"compute_xpsnr", &VmafOptions::compute_xpsnr},
    {"compute_psnr_yuv", &VmafOptions::compute_psnr_yuv},
    {"compute_psnr_yuv_mse", &VmafOptions::compute_psnr_yuv_mse},
    {"compute_ssim_yuv", &VmafOptions::compute_ssim_yuv},
    {"compute_ms_ssim", &VmafOptions::compute_ms_ssim},
    {"compute_ms_ssim_yuv", &VmafOptions::compute_ms_ssim_yuv},
    {"compute_psnr_hvs", &VmafOptions::compute_psnr_hvs},
    {"compute_cambi", &VmafOptions::compute_cambi},
    {"compute_ciede2000", &VmafOptions::compute_ciede2000},
    {"compute_vmaf_yuv", &VmafOptions::compute_vmaf_yuv},
    {"compute_psnr_yuv_y", &VmafOptions::compute_psnr_yuv_y},
    {"compute_psnr_yuv_u", &VmafOptions::compute_psnr_yuv_u},
    {"compute_psnr_yuv_v", &VmafOptions::compute_psnr_yuv_v},
    {"compute_ssim_yuv_y", &VmafOptions::compute_ssim_yuv_y},
    {"compute_ssim_yuv_u", &VmafOptions::compute_ssim_yuv_u},
    {"compute_ssim_yuv_v", &VmafOptions::compute_ssim_yuv_v},
    {"compute_ms_ssim_yuv_y", &VmafOptions::compute_ms_ssim_yuv_y},
    {"compute_ms_ssim_yuv_u", &VmafOptions::compute_ms_ssim_yuv_u},
    {"compute_ms_ssim_yuv_v", &VmafOptions::compute_ms_ssim_yuv_v},
    {"compute_psnr_hvs_y", &VmafOptions::compute_psnr_hvs_y},
    {"compute_psnr_hvs_u", &VmafOptions::compute_psnr_hvs_u},
    {"compute_psnr_hvs_v", &VmafOptions::compute_psnr_hvs_v},
    {"compute_vmaf_yuv_y", &VmafOptions::compute_vmaf_yuv_y},
    {"compute_vmaf_yuv_u", &VmafOptions::compute_vmaf_yuv_u},
    {"compute_vmaf_yuv_v", &VmafOptions::compute_vmaf_yuv_v},
}};

// The option that switches an on/off metric on, by its name (bool_options).
bool VmafOptions::*bool_option(std::string_view name) {
    for (const auto& [option_name, option] : bool_options) {
        if (name == option_name) {
            return option;
        }
    }
    throw std::logic_error("no option called " + std::string(name));
}

}  // namespace

std::string_view vship_build_value(VshipBuild build) {
    switch (build) {
    case VshipBuild::stock:
        return "vship";
    case VshipBuild::hip:
        return "vship-hip";
    case VshipBuild::cuda:
        return "vship-cuda";
    default:
        return "vship-fast";
    }
}

VshipBuild vship_build_from(std::string_view value) {
    for (const VshipBuild build : {VshipBuild::stock, VshipBuild::hip, VshipBuild::cuda}) {
        if (value == vship_build_value(build)) {
            return build;
        }
    }
    return VshipBuild::fast;
}

std::string_view vship_build_label(VshipBuild build) {
    switch (build) {
    case VshipBuild::stock:
        return "Vship 5.1.2";
    case VshipBuild::hip:
        return "Vship 5.1.1 HIP";
    case VshipBuild::cuda:
        return "Vship 5.1.2 CUDA";
    default:
        return "Vship-fast";
    }
}

std::filesystem::path synthetic_scale_direction_variant_path(const std::filesystem::path& distorted,
                                                             ScaleDirection direction) {
    std::filesystem::path name = distorted.stem();
    name += scale_direction_variant_suffix(direction);
    name += distorted.extension();
    return distorted.parent_path() / name;
}

std::string_view scale_direction_variant_suffix(ScaleDirection direction) {
    return direction == ScaleDirection::distorted_to_source ? " [upscale-distorted-to-source]"
                                                            : " [downscale-source-to-distorted]";
}

std::string_view film_grain_variant_suffix(bool with_film_grain) {
    return with_film_grain ? " [with film grain]" : " [without film grain]";
}

std::filesystem::path synthetic_film_grain_variant_path(const std::filesystem::path& distorted, bool with_film_grain) {
    std::filesystem::path name = distorted.stem();
    name += film_grain_variant_suffix(with_film_grain);
    name += distorted.extension();
    return distorted.parent_path() / name;
}

std::optional<bool> film_grain_variant_choice(const std::filesystem::path& path,
                                              const std::filesystem::path& distorted) {
    for (const bool with : {true, false}) {
        if (path == synthetic_film_grain_variant_path(distorted, with)) {
            return with;
        }
    }
    return std::nullopt;
}

namespace {

// What a display companion's name holds between the test video's name and
// its extension: " [CVVDP on " the display "]".
constexpr std::string_view cvvdp_variant_opening = " [CVVDP on ";

// Whether a file name cannot hold `c` on some system: a made-up path is
// never opened, but it is split as a path.
bool unfit_for_a_file_name(char c) { return std::string_view("/\\:*?\"<>|").find(c) != std::string_view::npos; }

}  // namespace

std::filesystem::path synthetic_cvvdp_display_variant_path(const std::filesystem::path& row_path,
                                                           const std::string& display) {
    std::string shown = display;
    std::replace_if(shown.begin(), shown.end(), unfit_for_a_file_name, '-');
    std::filesystem::path name = row_path.stem();
    name += platform::path_from_text(std::string(cvvdp_variant_opening) + shown + "]");
    name += row_path.extension();
    return row_path.parent_path() / name;
}

std::optional<std::filesystem::path> cvvdp_display_variant_parent(const std::filesystem::path& path) {
    const std::string stem = platform::path_text(path.stem());
    const std::size_t opening = stem.rfind(cvvdp_variant_opening);
    if (opening == std::string::npos || stem.back() != ']') {
        return std::nullopt;
    }
    std::filesystem::path name = platform::path_from_text(stem.substr(0, opening));
    name += path.extension();
    return path.parent_path() / name;
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

ResampleTarget resample_target_of_width(int width) {
    for (const ResampleTarget& listed : resample_target_choices()) {
        if (listed.width == width) {
            return listed;
        }
    }
    return {width, std::to_string(width) + "w"};
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

const std::vector<std::string_view>& switched_metrics() {
    static const std::vector<std::string_view> keys = [] {
        std::vector<std::string_view> found;
        for (const MetricDefinition& metric : metrics()) {
            if (metric.ffmpeg_binding && !metric.ffmpeg_binding->bool_option.empty() && !metric.is_plane()) {
                found.push_back(metric.key);
            }
        }
        return found;
    }();
    return keys;
}

}  // namespace vml
