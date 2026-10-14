#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "vml/platform/system.h"

namespace vml::test {

namespace {

VideoInfo video_from(const nlohmann::json& data) {
    VideoInfo info;
    info.path = platform::path_from_text(data.at("path").get<std::string>());
    info.width = data.at("width");
    info.height = data.at("height");
    info.fps = *test::reference_number(data.at("fps"));
    info.duration = *test::reference_number(data.at("duration"));
    info.nb_frames = data.at("nb_frames");
    info.codec_name = data.at("codec_name");
    info.sar = data.at("sar");
    info.pix_fmt = data.at("pix_fmt");
    info.bit_rate = data.at("bit_rate");
    info.bit_rate_whole_file = data.at("bit_rate_whole_file");
    info.nominal_fps = *test::reference_number(data.at("nominal_fps"));
    info.average_fps = *test::reference_number(data.at("average_fps"));
    info.color_range = data.at("color_range");
    info.color_space = data.at("color_space");
    info.color_transfer = data.at("color_transfer");
    info.color_primaries = data.at("color_primaries");
    info.chroma_location = data.at("chroma_location");
    return info;
}

std::optional<CropBox> crop_from(const nlohmann::json& data) {
    if (data.is_null()) {
        return std::nullopt;
    }
    return CropBox{data[0], data[1], data[2], data[3]};
}

std::optional<std::string> text_or_none(const nlohmann::json& data) {
    if (data.is_null()) {
        return std::nullopt;
    }
    return data.get<std::string>();
}

}  // namespace

nlohmann::json load_reference(std::string_view file_name) {
    const std::string path = std::string(VML_REFERENCE_DIR) + "/" + std::string(file_name);
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    return nlohmann::json::parse(file);
}

std::optional<double> reference_number(const nlohmann::json& text) {
    if (text.is_null()) {
        return std::nullopt;
    }
    // strtod reads float.hex()'s "0x1.7ap+6" as well as "inf" and "nan".
    return std::strtod(text.get<std::string>().c_str(), nullptr);
}

std::vector<double> reference_numbers(const nlohmann::json& list) {
    std::vector<double> numbers;
    for (const nlohmann::json& text : list) {
        numbers.push_back(*reference_number(text));
    }
    return numbers;
}

bool same_number(double expected, double actual) {
    if (std::isnan(expected) || std::isnan(actual)) {
        return std::isnan(expected) && std::isnan(actual);
    }
    if (expected == actual) {
        return true;
    }
#ifdef _WIN32
    return false;
#else
    if (std::isinf(expected) || std::isinf(actual)) {
        return false;
    }
    return std::abs(expected - actual) <= 1e-12 * std::max(std::abs(expected), std::abs(actual));
#endif
}

MetricProvenance provenance_from(const nlohmann::json& data) {
    MetricProvenance provenance{data.at("implementation"), data.at("implementation_version"),
                                data.at("compute_backend"), data.at("implementation_compatibility_id"), {}};
    for (auto item = data.at("parameters").begin(); item != data.at("parameters").end(); ++item) {
        provenance.parameters[item.key()] = item.value().dump();
    }
    return provenance;
}

std::vector<float> floats_from(const nlohmann::json& list) {
    std::vector<float> values;
    for (const double value : test::reference_numbers(list)) {
        values.push_back(static_cast<float>(value));
    }
    return values;
}

ComparisonResult result_from(const nlohmann::json& data) {
    ComparisonResult result;
    result.source = platform::path_from_text(data.at("source").get<std::string>());
    result.distorted = platform::path_from_text(data.at("distorted").get<std::string>());
    result.fps = *test::reference_number(data.at("fps"));
    result.model = data.at("model");
    result.source_crop = crop_from(data.at("source_crop"));
    result.distorted_crop = crop_from(data.at("distorted_crop"));
    result.source_info = video_from(data.at("source_info"));
    result.distorted_info = video_from(data.at("distorted_info"));
    result.scale_direction = data.at("scale_direction") == "distorted_to_source" ? ScaleDirection::distorted_to_source
                                                                                 : ScaleDirection::source_to_distorted;
    result.scale_algorithm = data.at("scale_algorithm");
    result.compared_frame_count = data.at("compared_frame_count");
    result.model_choice = text_or_none(data.at("model_choice"));
    result.model_v1 = data.at("model_v1");
    result.model_choice_v1 = text_or_none(data.at("model_choice_v1"));
    if (const nlohmann::json& target = data.at("resample_target"); !target.is_null()) {
        result.resample_target = ResampleTarget{target.at("width"), target.at("label")};
    }
    for (const nlohmann::json& metric : data.at("metrics")) {
        const std::string key = metric.at("key");
        if (metric.at("kind") == "sequence") {
            SequenceMetricResult sequence{key, *test::reference_number(metric.at("score")),
                                          provenance_from(metric.at("provenance")), {}, {}, {}};
            if (metric.contains("values")) {
                sequence.frame = metric.at("frame").get<std::vector<int>>();
                sequence.time = test::reference_numbers(metric.at("time"));
                sequence.values = floats_from(metric.at("values"));
            }
            result.metric_results.add(sequence);
        } else {
            result.metric_results.add(FrameMetricResult{key, metric.at("frame").get<std::vector<int>>(),
                                                        test::reference_numbers(metric.at("time")),
                                                        floats_from(metric.at("values")),
                                                        provenance_from(metric.at("provenance"))});
        }
    }
    return result;
}

}  // namespace vml::test
