#include "vml/run_json.h"

#include "vml/platform/system.h"

namespace vml {

nlohmann::json video_json(const VideoInfo& info) {
    return {{"path", platform::path_text(info.path)},
            {"width", info.width},
            {"height", info.height},
            {"fps", info.fps},
            {"duration", info.duration},
            {"nb_frames", info.nb_frames},
            {"codec_name", info.codec_name},
            {"sar", info.sar},
            {"pix_fmt", info.pix_fmt},
            {"bit_rate", info.bit_rate},
            {"bit_rate_whole_file", info.bit_rate_whole_file},
            {"nominal_fps", info.nominal_fps},
            {"average_fps", info.average_fps},
            {"color_range", info.color_range},
            {"color_space", info.color_space},
            {"color_transfer", info.color_transfer},
            {"color_primaries", info.color_primaries},
            {"chroma_location", info.chroma_location},
            {"format_name", info.format_name},
            {"start_offset", info.start_offset},
            {"timed_raw", info.timed_raw}};
}

VideoInfo video_from(const nlohmann::json& text) {
    VideoInfo info;
    info.path = platform::path_from_text(text.at("path").get<std::string>());
    text.at("width").get_to(info.width);
    text.at("height").get_to(info.height);
    text.at("fps").get_to(info.fps);
    text.at("duration").get_to(info.duration);
    text.at("nb_frames").get_to(info.nb_frames);
    text.at("codec_name").get_to(info.codec_name);
    text.at("sar").get_to(info.sar);
    text.at("pix_fmt").get_to(info.pix_fmt);
    text.at("bit_rate").get_to(info.bit_rate);
    text.at("bit_rate_whole_file").get_to(info.bit_rate_whole_file);
    text.at("nominal_fps").get_to(info.nominal_fps);
    text.at("average_fps").get_to(info.average_fps);
    text.at("color_range").get_to(info.color_range);
    text.at("color_space").get_to(info.color_space);
    text.at("color_transfer").get_to(info.color_transfer);
    text.at("color_primaries").get_to(info.color_primaries);
    text.at("chroma_location").get_to(info.chroma_location);
    text.at("format_name").get_to(info.format_name);
    text.at("start_offset").get_to(info.start_offset);
    text.at("timed_raw").get_to(info.timed_raw);
    return info;
}

nlohmann::json options_json(const VmafOptions& options) {
    return {{"model", options.model},
            {"model_choice", options.model_choice},
            {"custom_model_path", options.custom_model_path},
            {"extra_features", options.extra_features},
            {"n_threads", options.n_threads},
            {"n_subsample", options.n_subsample},
            {"scale_algorithm", options.scale_algorithm},
            {"scale_direction", static_cast<int>(options.scale_direction)},
            {"compute_xpsnr", options.compute_xpsnr},
            {"compute_vmaf", options.compute_vmaf},
            {"compute_vmaf_neg", options.compute_vmaf_neg},
            {"compute_vmaf_v1", options.compute_vmaf_v1},
            {"model_choice_v1", options.model_choice_v1},
            {"model_v1", options.model_v1},
            {"duration_limit", options.duration_limit},
            {"gpu_decode", options.gpu_decode},
            {"gpu_vendor", static_cast<int>(options.gpu_vendor)},
            {"vmaf_on_gpu", options.vmaf_on_gpu},
            {"vship_build", static_cast<int>(options.vship_build)},
            {"crop_mode", static_cast<int>(options.crop_mode)}};
}

VmafOptions options_from(const nlohmann::json& text) {
    VmafOptions options;
    text.at("model").get_to(options.model);
    text.at("model_choice").get_to(options.model_choice);
    text.at("custom_model_path").get_to(options.custom_model_path);
    text.at("extra_features").get_to(options.extra_features);
    text.at("n_threads").get_to(options.n_threads);
    text.at("n_subsample").get_to(options.n_subsample);
    text.at("scale_algorithm").get_to(options.scale_algorithm);
    options.scale_direction = static_cast<ScaleDirection>(text.at("scale_direction").get<int>());
    text.at("compute_xpsnr").get_to(options.compute_xpsnr);
    text.at("compute_vmaf").get_to(options.compute_vmaf);
    text.at("compute_vmaf_neg").get_to(options.compute_vmaf_neg);
    text.at("compute_vmaf_v1").get_to(options.compute_vmaf_v1);
    text.at("model_choice_v1").get_to(options.model_choice_v1);
    text.at("model_v1").get_to(options.model_v1);
    text.at("duration_limit").get_to(options.duration_limit);
    text.at("gpu_decode").get_to(options.gpu_decode);
    options.gpu_vendor = static_cast<GpuVendor>(text.at("gpu_vendor").get<int>());
    text.at("vmaf_on_gpu").get_to(options.vmaf_on_gpu);
    options.vship_build = static_cast<VshipBuild>(text.at("vship_build").get<int>());
    options.crop_mode = static_cast<CropMode>(text.at("crop_mode").get<int>());
    return options;
}

nlohmann::json crop_json(const std::optional<CropBox>& crop) {
    return crop ? nlohmann::json{{"w", crop->w}, {"h", crop->h}, {"x", crop->x}, {"y", crop->y}} : nlohmann::json();
}

std::optional<CropBox> crop_from(const nlohmann::json& text) {
    if (text.is_null()) {
        return std::nullopt;
    }
    return CropBox{text.at("w").get<int>(), text.at("h").get<int>(), text.at("x").get<int>(), text.at("y").get<int>()};
}

}  // namespace vml
