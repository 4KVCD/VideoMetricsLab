// The JSON Schemas in docs/schemas hold every JSON document the command line
// writes, and the window's exports of the same: each document is made here as
// the program makes it and checked against its schema, so a key changed in
// the code without its schema (or the other way round) fails.
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "bdrate.h"
#include "bitrate.h"
#include "devices.h"
#include "report.h"
#include "weak_report.h"
#include "vml/bd_rate_report.h"
#include "vml/export_csv.h"
#include "vml/results_summary.h"
#include "vml/run_record.h"
#include "vml/weak_frames.h"

namespace vml::cli {
namespace {

using Json = nlohmann::json;

Json load_schema(const std::string& name) {
    std::ifstream file(std::filesystem::path(VML_SCHEMA_DIR) / (name + ".schema.json"), std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    return Json::parse(text.str());
}

// The part of JSON Schema (2020-12) the schemas use: $ref to their own
// $defs, type, const, enum, required, properties, additionalProperties,
// items and anyOf. Says each place `value` does not hold, in `errors`.
class Validator {
public:
    explicit Validator(const Json& root) : root_(root) {}

    void check(const Json& schema, const Json& value, const std::string& at, std::vector<std::string>& errors) const {
        if (schema.contains("$ref")) {
            const std::string ref = schema["$ref"];
            check(root_.at(Json::json_pointer(ref.substr(1))), value, at, errors);
        }
        if (schema.contains("type") && !has_type(schema["type"], value)) {
            errors.push_back(at + ": not of type " + schema["type"].dump() + ": " + value.dump().substr(0, 80));
            return;
        }
        if (schema.contains("const") && schema["const"] != value) {
            errors.push_back(at + ": not " + schema["const"].dump());
        }
        if (schema.contains("enum") &&
            std::find(schema["enum"].begin(), schema["enum"].end(), value) == schema["enum"].end()) {
            errors.push_back(at + ": " + value.dump() + " is none of " + schema["enum"].dump());
        }
        if (schema.contains("anyOf")) {
            bool any = false;
            for (const Json& option : schema["anyOf"]) {
                std::vector<std::string> theirs;
                check(option, value, at, theirs);
                any = any || theirs.empty();
            }
            if (!any) {
                errors.push_back(at + ": matches none of anyOf: " + value.dump().substr(0, 80));
            }
        }
        if (value.is_object()) {
            for (const Json& key : schema.value("required", Json::array())) {
                if (!value.contains(key.get<std::string>())) {
                    errors.push_back(at + ": no " + key.get<std::string>());
                }
            }
            const Json properties = schema.value("properties", Json::object());
            for (auto item = value.begin(); item != value.end(); ++item) {
                if (properties.contains(item.key())) {
                    check(properties[item.key()], item.value(), at + "/" + item.key(), errors);
                } else if (schema.contains("additionalProperties")) {
                    const Json& extra = schema["additionalProperties"];
                    if (extra.is_boolean() && !extra.get<bool>()) {
                        errors.push_back(at + ": " + item.key() + " is not allowed");
                    } else if (extra.is_object()) {
                        check(extra, item.value(), at + "/" + item.key(), errors);
                    }
                }
            }
        }
        if (value.is_array() && schema.contains("items")) {
            for (std::size_t index = 0; index < value.size(); ++index) {
                check(schema["items"], value[index], at + "/" + std::to_string(index), errors);
            }
        }
    }

private:
    static bool is_type(const std::string& type, const Json& value) {
        return type == "object"    ? value.is_object()
               : type == "array"   ? value.is_array()
               : type == "string"  ? value.is_string()
               : type == "integer" ? value.is_number_integer()
               : type == "number"  ? value.is_number()
               : type == "boolean" ? value.is_boolean()
                                   : type == "null" && value.is_null();
    }
    static bool has_type(const Json& type, const Json& value) {
        if (type.is_string()) {
            return is_type(type.get<std::string>(), value);
        }
        return std::any_of(type.begin(), type.end(), [&value](const Json& one) { return is_type(one, value); });
    }

    const Json& root_;
};

// The places `document` does not hold to the schema `name`.
std::vector<std::string> schema_errors(const std::string& name, const nlohmann::ordered_json& document) {
    const Json schema = load_schema(name);
    std::vector<std::string> errors;
    Validator(schema).check(schema, Json::parse(document.dump()), "", errors);
    return errors;
}

// A scored video: VMAF and PSNR (an identical frame: PSNR's "inf"), a
// CVVDP score, how its run went, and its weak segments.
TestVideo scored_video() {
    TestVideo video;
    video.path = "a.mkv";
    video.metrics = {"vmaf", "psnr", "cvvdp"};
    ComparisonResult& result = video.result.emplace();
    result.source = "r.mkv";
    result.distorted = "a.mkv";
    result.fps = 24.0;
    result.compared_frame_count = 3;
    result.metric_results.add(FrameMetricResult{"vmaf", {0, 1, 2}, {0.0, 1 / 24.0, 2 / 24.0}, {95, 80, 70}, {}});
    result.metric_results.add(FrameMetricResult{
        "psnr", {0, 1, 2}, {0.0, 1 / 24.0, 2 / 24.0}, {std::numeric_limits<float>::infinity(), 40, 38}, {}});
    result.metric_results.add(SequenceMetricResult{"cvvdp", 8.5, {}, {0, 2}, {0.0, 1.0}, {8.0F, 9.0F}});
    RunFacts facts;
    facts.settings.metrics = video.metrics;
    facts.metrics = {{"vmaf", {metric_calculated, ""}},
                     {"psnr", {metric_calculated, ""}},
                     {"cvvdp", {metric_failed, "the GPU ran out of memory"}}};
    video.facts = facts;
    video.notes = {"the reference has no colour tags"};
    return video;
}

TEST(Schemas, HoldEveryDocument) {
    std::vector<TestVideo> videos{scored_video()};
    TestVideo& unread = videos.emplace_back();
    unread.path = "missing.mkv";
    unread.error = "it could not be read";
    const SegmentRules rules{{WeakCondition{"vmaf", '<', 85.0}}};
    nlohmann::ordered_json results = results_document(VideoInfo{}, videos, summarize(videos), false);
    add_segments_json(results, videos, rules);
    EXPECT_EQ(schema_errors("results", results), std::vector<std::string>{});
    const nlohmann::ordered_json batch{
        {"batch", {results, {{"reference", "gone.mkv"}, {"error", "it could not be read"}}}}};
    EXPECT_EQ(schema_errors("results", batch), std::vector<std::string>{});
    const Json model = results["videos"][0]["settings"]["vmaf_model"];
    EXPECT_TRUE(model.contains("id")) << model.dump();

    const ComparisonResult& result = *videos[0].result;
    const std::vector<WeakSegment> segments = weak_segments(result, rules);
    ASSERT_EQ(segments.size(), 1u);
    EXPECT_EQ(schema_errors("segments", segments_file_json(segments, rules)), std::vector<std::string>{});
    const nlohmann::ordered_json report =
        segments_report_json({SegmentsReportVideo{"a.mkv", "a.mkv", "r.mkv", &result}}, rules);
    EXPECT_EQ(report["format"], "segments-report-1");
    EXPECT_EQ(schema_errors("segments-report", report), std::vector<std::string>{});

    const MetricResultSet& scores = result.metric_results;
    const std::optional<ScoreSamples> vmaf = score_samples(scores, "vmaf");
    ASSERT_TRUE(vmaf);
    const std::vector<WorstFrame> dips = worst_frames(*vmaf, MetricDirection::higher_is_better, 10);
    ASSERT_FALSE(dips.empty());
    const nlohmann::ordered_json worst = worst_frames_json(scores, "vmaf", dips, 10, {});
    EXPECT_EQ(worst["format"], "worst-frames-1");
    EXPECT_EQ(schema_errors("worst-frames", worst), std::vector<std::string>{});

    const nlohmann::ordered_json frames = nlohmann::ordered_json::parse(per_frame_json(result));
    EXPECT_EQ(frames["rows"][0][4], "inf");  // PSNR of the identical frame: JSON, not Python's Infinity
    EXPECT_EQ(schema_errors("frames", frames), std::vector<std::string>{});

    std::vector<BitrateVideo> read(1);
    read[0].path = "missing.mkv";
    read[0].error = "it could not be read";
    EXPECT_EQ(schema_errors("bitrate", bitrate_json(read, std::nullopt)), std::vector<std::string>{});

    const RateTable anchor = read_rate_table("sequence,kbps,vmaf\ns,1000,80\ns,2000,85\ns,4000,90\ns,8000,94\n", "a");
    const RateTable test = read_rate_table("sequence,kbps,vmaf\ns,900,80\ns,1800,85\ns,3600,90\ns,7200,94\n", "t");
    const nlohmann::ordered_json rates = bd_rate_json(bd_rate_report(anchor, test, BdRateOptions{}), "a", "t");
    EXPECT_EQ(schema_errors("bdrate", rates), std::vector<std::string>{});

    const DeviceFindings found;  // FFmpeg not found, no GPU checked
    const nlohmann::ordered_json devices = devices_json(found, {"NVIDIA GeForce RTX 5090"});
    EXPECT_EQ(devices["format"], "devices-1");
    EXPECT_TRUE(devices["metrics"]["cvvdp"]["on"].is_null());  // no Vship GPU: CVVDP not available
    EXPECT_EQ(schema_errors("devices", devices), std::vector<std::string>{});

    // A document that breaks its schema is caught: the validator checks.
    nlohmann::ordered_json broken = devices;
    broken["format"] = "devices-2";
    broken["metrics"].erase("vmaf");
    EXPECT_EQ(schema_errors("devices", broken).size(), 2u);
}

}  // namespace
}  // namespace vml::cli
