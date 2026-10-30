#include "vml/result_json.h"

#include <stdexcept>

#include "vml/film_grain_detect.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"

namespace vml {

std::string python_path_text(const std::filesystem::path& path) {
    std::filesystem::path preferred = path;
    preferred.make_preferred();
    return platform::path_text(preferred);
}

nlohmann::ordered_json crop_to_dict(const std::optional<CropBox>& crop) {
    if (!crop) {
        return nullptr;
    }
    return {{"w", crop->w}, {"h", crop->h}, {"x", crop->x}, {"y", crop->y}};
}

std::optional<CropBox> crop_from_dict(const nlohmann::json& data) {
    if (data.is_null()) {
        return std::nullopt;
    }
    if (!data.is_object() || data.size() != 4) {
        throw std::invalid_argument("A crop needs w, h, x and y.");
    }
    return CropBox{python_item(data, "w").get<int>(), python_item(data, "h").get<int>(), python_item(data, "x").get<int>(),
                   python_item(data, "y").get<int>()};
}

nlohmann::ordered_json video_info_to_dict(const VideoInfo& info) {
    nlohmann::ordered_json dict = {{"path", python_path_text(info.path)},
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
            {"chroma_location", info.chroma_location}};
    // Each only where set, so that a dict without them stays as the Python
    // app wrote it. Without the profile, pick_decode takes H.264 as one that
    // Intel's and AMD's GPUs may decode wrongly.
    if (!info.profile.empty()) {
        dict["profile"] = info.profile;
    }
    if (info.scaled_to) {
        dict["scaled_to"] = {info.scaled_to->first, info.scaled_to->second};
    }
    if (info.film_grain) {
        dict["film_grain"] = film_grain_names(*info.film_grain);
    }
    if (info.with_film_grain) {
        dict["with_film_grain"] = *info.with_film_grain;
    }
    return dict;
}

VideoInfo video_info_from_dict(const nlohmann::json& data) {
    VideoInfo info;
    info.path = platform::path_from_text(python_item(data, "path").get<std::string>());
    info.width = python_item(data, "width").get<int>();
    info.height = python_item(data, "height").get<int>();
    info.fps = python_item(data, "fps").get<double>();
    info.duration = python_item(data, "duration").get<double>();
    info.nb_frames = python_item(data, "nb_frames").get<long long>();
    info.codec_name = python_item(data, "codec_name").get<std::string>();
    info.sar = data.value("sar", std::string("1:1"));
    info.pix_fmt = data.value("pix_fmt", std::string());
    info.bit_rate = data.value("bit_rate", 0LL);
    info.bit_rate_whole_file = python_truthy(data.value("bit_rate_whole_file", nlohmann::json(false)));
    info.nominal_fps = data.value("nominal_fps", 0.0);
    info.average_fps = data.value("average_fps", 0.0);
    info.color_range = data.value("color_range", std::string());
    info.color_space = data.value("color_space", std::string());
    info.color_transfer = data.value("color_transfer", std::string());
    info.color_primaries = data.value("color_primaries", std::string());
    info.chroma_location = data.value("chroma_location", std::string());
    info.profile = data.value("profile", std::string());  // saved from 2.3 on
    if (const auto scaled = data.find("scaled_to"); scaled != data.end() && scaled->is_array() && scaled->size() == 2) {
        info.scaled_to = std::pair(scaled->at(0).get<int>(), scaled->at(1).get<int>());
    }
    if (const auto grain = data.find("film_grain"); grain != data.end() && grain->is_array()) {  // from 2.3 on
        std::vector<std::string> names;
        for (const nlohmann::json& name : *grain) {
            if (name.is_string()) {
                names.push_back(name.get<std::string>());
            }
        }
        info.film_grain = film_grain_from_names(names);
    }
    if (const auto with = data.find("with_film_grain"); with != data.end() && with->is_boolean()) {
        info.with_film_grain = with->get<bool>();
    }
    return info;
}

nlohmann::ordered_json provenance_to_dict(const MetricProvenance& provenance) {
    nlohmann::ordered_json parameters = nlohmann::ordered_json::object();
    for (const auto& [name, text] : provenance.parameters) {
        parameters[name] = nlohmann::ordered_json::parse(text);
    }
    return {{"implementation", provenance.implementation},
            {"implementation_version", provenance.implementation_version},
            {"compute_backend", provenance.compute_backend},
            {"implementation_compatibility_id", provenance.implementation_compatibility_id},
            {"parameters", parameters}};
}

MetricProvenance provenance_from_dict(const nlohmann::json& data) {
    const nlohmann::json parameters = data.value("parameters", nlohmann::json::object());
    if (!parameters.is_object()) {
        throw std::invalid_argument("metric provenance parameters must be an object");
    }
    MetricProvenance provenance{python_str(python_item(data, "implementation")),
                                python_str(python_item(data, "implementation_version")),
                                python_str(python_item(data, "compute_backend")),
                                python_str(python_item(data, "implementation_compatibility_id")),
                                {}};
    for (auto item = parameters.begin(); item != parameters.end(); ++item) {
        provenance.parameters[item.key()] = item.value().dump();
    }
    return provenance;
}

nlohmann::ordered_json resample_target_to_dict(const std::optional<ResampleTarget>& target) {
    if (!target) {
        return nullptr;
    }
    return {{"width", target->width}, {"label", target->label}};
}

const char* scale_direction_value(ScaleDirection direction) {
    return direction == ScaleDirection::source_to_distorted ? "source_to_distorted" : "distorted_to_source";
}

ScaleDirection scale_direction_from(const std::string& value) {
    if (value == "source_to_distorted") {
        return ScaleDirection::source_to_distorted;
    }
    if (value == "distorted_to_source") {
        return ScaleDirection::distorted_to_source;
    }
    throw std::invalid_argument("'" + value + "' is not a valid ScaleDirection");
}

}  // namespace vml
