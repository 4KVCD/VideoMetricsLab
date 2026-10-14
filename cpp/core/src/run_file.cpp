#include "vml/run_file.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <variant>

#include "vml/model_select.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/result_json.h"

namespace vml {

namespace {

// The run files' format, as the Python app's FORMAT_VERSION.
constexpr int format_version = 2;

using Json = nlohmann::ordered_json;

// A score as JSON allows it: NaN as null, the infinities as text.
Json json_float(double value) {
    if (std::isnan(value)) {
        return nullptr;
    }
    if (std::isinf(value)) {
        return value > 0 ? "Infinity" : "-Infinity";
    }
    return value;
}

// Python's float() of a text: surrounding spaces, "inf", "nan" and the
// like allowed, nothing else after the number.
double python_float(const std::string& text) {
    std::size_t first = 0;
    std::size_t last = text.size();
    while (first < last && std::isspace(static_cast<unsigned char>(text[first])) != 0) {
        ++first;
    }
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1])) != 0) {
        --last;
    }
    const std::string number = text.substr(first, last - first);
    std::string folded = number;
    for (char& c : folded) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    const bool negative = !folded.empty() && folded.front() == '-';
    const std::string bare = !folded.empty() && (folded.front() == '-' || folded.front() == '+') ? folded.substr(1) : folded;
    if (bare == "inf" || bare == "infinity") {
        return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    }
    if (bare == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Digits only from here: strtod would also take hexadecimal, which
    // Python's float() does not.
    const bool digits = !bare.empty() && bare.find_first_not_of("0123456789.e+-") == std::string::npos;
    char* end = nullptr;
    const double value = digits ? std::strtod(number.c_str(), &end) : 0.0;
    if (!digits || end != number.c_str() + number.size()) {
        throw std::invalid_argument("could not convert string to float: " + python_string_repr(text));
    }
    return value;
}

// A score read back from json_float's form.
double float_from_json(const nlohmann::json& value) {
    if (value.is_null()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (value.is_string()) {
        return python_float(value.get<std::string>());
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? 1.0 : 0.0;
    }
    return value.get<double>();
}

Json optional_text(const std::optional<std::string>& text) { return text ? Json(*text) : Json(nullptr); }

// The frame numbers, times and scores of a metric or of its timeline.
Json timeline_arrays(const std::vector<int>& frame, const std::vector<double>& time, const std::vector<float>& values) {
    Json frames = Json::array();
    for (const int number : frame) {
        frames.push_back(number);
    }
    Json times = Json::array();
    for (const double seconds : time) {
        times.push_back(json_float(seconds));
    }
    Json scores = Json::array();
    for (const float score : values) {
        scores.push_back(json_float(score));
    }
    return {{"frame", std::move(frames)}, {"time", std::move(times)}, {"values", std::move(scores)}};
}

Json metric_to_dict(const FrameMetricResult& metric) {
    Json data{{"key", metric.key}, {"provenance", provenance_to_dict(metric.provenance)}, {"kind", "frame"}};
    data.update(timeline_arrays(metric.frame, metric.time, metric.values));
    return data;
}

Json metric_to_dict(const SequenceMetricResult& metric) {
    Json data{{"key", metric.key},
              {"provenance", provenance_to_dict(metric.provenance)},
              {"kind", "sequence"},
              {"score", json_float(metric.score)}};
    if (metric.has_timeline()) {
        data["timeline"] = timeline_arrays(metric.frame, metric.time, metric.values);
    }
    return data;
}

// The frame numbers, times and scores read back, which must be as many each.
struct Timeline {
    std::vector<int> frame;
    std::vector<double> time;
    std::vector<float> values;
};

Timeline timeline_from(const nlohmann::json& frames, const nlohmann::json& times, const nlohmann::json& values,
                       const char* unequal) {
    Timeline timeline;
    for (const nlohmann::json& number : frames) {
        timeline.frame.push_back(number.get<int>());
    }
    for (const nlohmann::json& seconds : times) {
        timeline.time.push_back(float_from_json(seconds));
    }
    for (const nlohmann::json& score : values) {
        timeline.values.push_back(static_cast<float>(float_from_json(score)));
    }
    if (timeline.frame.size() != timeline.time.size() || timeline.time.size() != timeline.values.size()) {
        throw std::invalid_argument(unequal);
    }
    return timeline;
}

// A metric's result read back from metric_to_dict's form (_metric_from_dict).
std::variant<FrameMetricResult, SequenceMetricResult> metric_from_dict(const nlohmann::json& data) {
    if (!data.is_object()) {
        throw std::invalid_argument("metric result must be an object");
    }
    const nlohmann::json& key = python_item(data, "key");
    if (!key.is_string() || key.get_ref<const std::string&>().empty()) {
        throw std::invalid_argument("metric result key must be a non-empty string");
    }
    const nlohmann::json& provenance_data = python_item(data, "provenance");
    if (!provenance_data.is_object()) {
        throw std::invalid_argument("metric provenance must be an object");
    }
    const MetricProvenance provenance = provenance_from_dict(provenance_data);
    const nlohmann::json kind = data.value("kind", nlohmann::json());
    if (kind == "frame") {
        const nlohmann::json& frames = python_item(data, "frame");
        const nlohmann::json& times = python_item(data, "time");
        const nlohmann::json& values = python_item(data, "values");
        if (!frames.is_array() || !times.is_array() || !values.is_array()) {
            throw std::invalid_argument("frame metric arrays must be lists");
        }
        Timeline timeline =
            timeline_from(frames, times, values, "frame, time, and metric values must have equal lengths");
        return FrameMetricResult{key.get<std::string>(), std::move(timeline.frame), std::move(timeline.time),
                                 std::move(timeline.values), provenance};
    }
    if (kind == "sequence") {
        const double score = float_from_json(python_item(data, "score"));
        SequenceMetricResult result{key.get<std::string>(), score, provenance, {}, {}, {}};
        if (const auto timeline = data.find("timeline"); timeline != data.end() && timeline->is_object()) {
            Timeline read = timeline_from(python_item(*timeline, "frame"), python_item(*timeline, "time"),
                                          python_item(*timeline, "values"),
                                          "timeline frame, time, and values must have equal lengths");
            result.frame = std::move(read.frame);
            result.time = std::move(read.time);
            result.values = std::move(read.values);
        }
        return result;
    }
    throw std::invalid_argument("Unsupported metric result kind: " + python_repr_of(kind));
}

// A resolution test's target read back: ResampleTarget(**data).
std::optional<ResampleTarget> resample_target_from(const nlohmann::json& data) {
    if (data.is_null()) {
        return std::nullopt;
    }
    if (!data.is_object() || data.size() != 2) {
        throw std::invalid_argument("A resolution test's target needs a width and a label.");
    }
    return ResampleTarget{python_item(data, "width").get<int>(), python_item(data, "label").get<std::string>()};
}

// Python's name of a JSON value's type, for the error it gives when a file
// holds something other than an object.
std::string python_type_name(const nlohmann::json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::null:
            return "NoneType";
        case nlohmann::json::value_t::boolean:
            return "bool";
        case nlohmann::json::value_t::number_integer:
        case nlohmann::json::value_t::number_unsigned:
            return "int";
        case nlohmann::json::value_t::number_float:
            return "float";
        case nlohmann::json::value_t::string:
            return "str";
        case nlohmann::json::value_t::array:
            return "list";
        default:
            return "dict";
    }
}

// A file saved while VMAF v1 was a model of the one VMAF column holds its v1
// scores as "vmaf": they become VMAF v1's, so that they show in the VMAF v1
// column and are never taken for VMAF v0.6.1's (_v1_from_old_vmaf_column).
void v1_from_old_vmaf_column(ComparisonResult& result) {
    const FrameMetricResult* vmaf = result.metric_results.frame("vmaf");
    const bool v1_choice = result.model_choice && !result.model_choice->empty() &&
                           is_builtin_model_choice(*result.model_choice);
    if (!v1_choice || vmaf == nullptr || result.metric_results.has("vmaf_v1")) {
        return;
    }
    MetricResultSet moved;
    for (const std::string& key : result.metric_results.keys()) {
        if (key == "vmaf") {
            moved.add(FrameMetricResult{"vmaf_v1", vmaf->frame, vmaf->time, vmaf->values, vmaf->provenance});
        } else if (const FrameMetricResult* frame = result.metric_results.frame(key)) {
            moved.add(*frame);
        } else if (const SequenceMetricResult* sequence = result.metric_results.sequence(key)) {
            moved.add(*sequence);
        }
    }
    result.metric_results = std::move(moved);
    result.model_v1 = result.model;
    result.model_choice_v1 = result.model_choice;
    result.model.clear();
    result.model_choice = std::string(auto_model_choice);
}

}  // namespace

std::string run_file_text(const ComparisonResult& result, const std::string& label) {
    Json metrics = Json::array();
    for (const std::string& key : result.metric_results.keys()) {
        if (const FrameMetricResult* frame = result.metric_results.frame(key)) {
            metrics.push_back(metric_to_dict(*frame));
        } else if (const SequenceMetricResult* sequence = result.metric_results.sequence(key)) {
            metrics.push_back(metric_to_dict(*sequence));
        }
    }
    const Json payload{{"format_version", format_version},
                       {"label", label.empty() ? platform::path_text(result.distorted.stem()) : label},
                       {"source", python_path_text(result.source)},
                       {"distorted", python_path_text(result.distorted)},
                       {"fps", result.fps},
                       {"model", result.model},
                       {"model_choice", optional_text(result.model_choice)},
                       {"model_v1", result.model_v1},
                       {"model_choice_v1", optional_text(result.model_choice_v1)},
                       {"source_crop", crop_to_dict(result.source_crop)},
                       {"distorted_crop", crop_to_dict(result.distorted_crop)},
                       {"source_info", video_info_to_dict(result.source_info)},
                       {"distorted_info", video_info_to_dict(result.distorted_info)},
                       {"scale_direction", scale_direction_value(result.scale_direction)},
                       {"scale_algorithm", result.scale_algorithm},
                       {"resample_target", resample_target_to_dict(result.resample_target)},
                       {"compared_frame_count", result.compared_frame_count},
                       {"metric_results", std::move(metrics)}};
    return python_json_line(payload);
}

void save_run(const ComparisonResult& result, const std::filesystem::path& path, const std::string& label) {
    const std::string text = run_file_text(result, label);
    std::ofstream file(path, std::ios::binary);
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    file.close();
    if (!file) {
        throw std::runtime_error("Could not write " + platform::path_text(path));
    }
}

std::pair<ComparisonResult, std::string> run_from_text(const std::string& text) {
    try {
        const nlohmann::json data = nlohmann::json::parse(text);
        if (!data.is_object()) {
            throw std::invalid_argument("'" + python_type_name(data) + "' object has no attribute 'get'");
        }
        const nlohmann::json version = data.value("format_version", nlohmann::json());
        if (!version.is_number() || version.get<double>() != format_version) {
            throw std::invalid_argument("Unsupported analysis result format version: " + python_repr_of(version));
        }
        const nlohmann::json metric_data = data.value("metric_results", nlohmann::json());
        if (!metric_data.is_array()) {
            throw std::invalid_argument("metric_results must be a list");
        }
        ComparisonResult result;
        for (const nlohmann::json& item : metric_data) {
            std::visit([&result](auto&& metric) { result.metric_results.add(std::move(metric)); },
                       metric_from_dict(item));
        }
        // In the order Python reads them, so that a file missing several
        // gives the same first complaint.
        result.source = platform::path_from_text(python_item(data, "source").get<std::string>());
        result.distorted = platform::path_from_text(python_item(data, "distorted").get<std::string>());
        result.fps = python_item(data, "fps").get<double>();
        result.model = python_item(data, "model").get<std::string>();
        result.source_crop = crop_from_dict(data.value("source_crop", nlohmann::json()));
        result.distorted_crop = crop_from_dict(data.value("distorted_crop", nlohmann::json()));
        result.source_info = video_info_from_dict(python_item(data, "source_info"));
        result.distorted_info = video_info_from_dict(python_item(data, "distorted_info"));
        result.scale_direction = scale_direction_from(python_item(data, "scale_direction").get<std::string>());
        result.scale_algorithm = python_item(data, "scale_algorithm").get<std::string>();
        result.resample_target = resample_target_from(python_item(data, "resample_target"));
        result.compared_frame_count = python_item(data, "compared_frame_count").get<long long>();
        const nlohmann::json& choice = python_item(data, "model_choice");
        result.model_choice = choice.is_null() ? std::nullopt : std::optional(choice.get<std::string>());
        const nlohmann::json model_v1 = data.value("model_v1", nlohmann::json());
        result.model_v1 = python_truthy(model_v1) ? model_v1.get<std::string>() : std::string();
        const nlohmann::json choice_v1 = data.value("model_choice_v1", nlohmann::json());
        result.model_choice_v1 = choice_v1.is_null() ? std::nullopt : std::optional(choice_v1.get<std::string>());
        // A file that does not count its frames counts those its scores
        // have, as the Python app's ComparisonResult does.
        if (const FrameScores frames = frame_scores_from_results(result.metric_results);
            result.compared_frame_count <= 0 && !frames.frame.empty()) {
            result.compared_frame_count = frames.frame.back() + 1LL;
        }
        v1_from_old_vmaf_column(result);
        return {std::move(result), python_str(python_item(data, "label"))};
    } catch (const RunFileError&) {
        throw;
    } catch (const std::exception& error) {
        throw RunFileError(error.what());
    }
}

std::pair<ComparisonResult, std::string> load_run(const std::filesystem::path& path) {
    const std::optional<std::string> text = platform::read_file(path);
    if (!text) {
        throw RunFileError("Could not read " + platform::path_text(path));
    }
    return run_from_text(*text);
}

}  // namespace vml
