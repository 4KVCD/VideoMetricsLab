// The JSON Schemas in docs/schemas hold every JSON document the command line
// writes, and the window's exports of the same: each document is made here as
// the program makes it and checked against its schema, so a key changed in
// the code without its schema (or the other way round) fails.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
#include "loop.h"
#include "report.h"
#include "saved.h"
#include "stills.h"
#include "weak_report.h"
#include "vml/bd_rate_report.h"
#include "vml/export_csv.h"
#include "vml/platform/system.h"
#include "vml/results_summary.h"
#include "vml/run_file.h"
#include "vml/run_record.h"
#include "vml/score_summary.h"
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

// A condition on a metric a video has no scores of is left out of its
// segments in the --json and the text as in its --segments-dir file, and
// said so there, not counted as holding on no frame.
TEST(Schemas, ConditionsAVideoHasNoScoresOfAreSaidLeftOut) {
    std::vector<TestVideo> videos{scored_video()};
    const SegmentRules rules{{WeakCondition{"vmaf", '<', 85.0}, WeakCondition{"ssimulacra2", '<', 70.0}}};
    nlohmann::ordered_json results = results_document(VideoInfo{}, videos, summarize(videos), false);
    add_segments_json(results, videos, rules);
    const nlohmann::ordered_json& weak = results["videos"][0]["weak_segments"];
    EXPECT_EQ(weak["conditions"], nlohmann::ordered_json::array({"vmaf<85"}));
    EXPECT_EQ(weak["left_out"], nlohmann::ordered_json::array({"ssimulacra2<70"}));
    EXPECT_EQ(schema_errors("results", results), std::vector<std::string>{});
    const std::string text = segments_text(videos, rules, 10);
    EXPECT_NE(text.find("where vmaf<85"), std::string::npos) << text;
    EXPECT_EQ(text.find("ssimulacra2<70: no frame"), std::string::npos) << text;
    EXPECT_NE(text.find("ssimulacra2<70: left out, as it has no scores of its metric"), std::string::npos) << text;
    // None of its conditions: no segments, and said why.
    const SegmentRules none{{WeakCondition{"ssimulacra2", '<', 70.0}}};
    EXPECT_NE(segments_text(videos, none, 10).find("no weak segments: it has no scores of the conditions' metrics"),
              std::string::npos);
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
    EXPECT_EQ(results["url"], "https://github.com/4KVCD/VideoMetricsLab");  // the program's home, after its name
    const nlohmann::ordered_json batch{
        {"batch", {results, {{"reference", "gone.mkv"}, {"error", "it could not be read"}}}}};
    EXPECT_EQ(schema_errors("results", batch), std::vector<std::string>{});
    const Json model = results["videos"][0]["settings"]["vmaf_model"];
    EXPECT_TRUE(model.contains("id")) << model.dump();

    const ComparisonResult& result = *videos[0].result;
    const std::vector<WeakSegment> segments = weak_segments(result, rules);
    ASSERT_EQ(segments.size(), 1u);
    EXPECT_EQ(schema_errors("segments", segments_file_json(result, segments, rules)), std::vector<std::string>{});
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
    const nlohmann::ordered_json every = worst_frames_report_json(scores, 10);
    EXPECT_EQ(every["format"], "worst-frames-report-1");
    EXPECT_EQ(schema_errors("worst-frames-report", every), std::vector<std::string>{});

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

// Score summary: each metric's score with the columns asked for, in their
// own order, the statistics of its frames and its bands; an endless score
// (PSNR of an identical frame) as "inf"; the saved time only where known. Its
// JSON holds to its schema; its CSV is a row a score.
TEST(Schemas, ScoreSummaryHoldsTheColumnsAskedFor) {
    const TestVideo video = scored_video();
    const auto saved = std::chrono::system_clock::from_time_t(1764079331);  // 2025-11-25T14:02:11Z
    const std::vector<ScoreSummaryEntry> entries{
        {"a", "a.mkv", "r.mkv", score_summary(*video.result, "", {{"vmaf", saved}})}};
    const std::vector<std::string> columns{"saved", "max", "bands", "no_such_column", "min"};
    const nlohmann::ordered_json json = score_summary_json(entries, columns);
    EXPECT_EQ(schema_errors("score-summary", json), std::vector<std::string>{});
    EXPECT_EQ(json["url"], "https://github.com/4KVCD/VideoMetricsLab");
    EXPECT_EQ(json["columns"], nlohmann::ordered_json({"min", "max", "bands", "saved"}));
    const nlohmann::ordered_json& scores = json["comparisons"][0]["scores"];
    ASSERT_EQ(scores.size(), 3U);  // VMAF, PSNR, CVVDP, in the metrics' order
    EXPECT_EQ(scores[0]["metric"], "vmaf");
    EXPECT_EQ(scores[0]["min"], 70.0);
    EXPECT_EQ(scores[0]["max"], 95.0);
    EXPECT_EQ(scores[0]["saved"], "2025-11-25T14:02:11Z");
    EXPECT_FALSE(scores[0]["bands"].empty());
    EXPECT_EQ(scores[1]["max"], "inf");
    EXPECT_TRUE(scores[1]["saved"].is_null());
    EXPECT_FALSE(scores[0].contains("median"));  // not asked for
    const std::string csv = score_summary_csv(entries, columns);
    EXPECT_EQ(csv.substr(0, csv.find('\n')), "Comparison,Metric,Score,Min,Max,Score bands,Saved");
    EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), 4);  // the heading, a row a score
}

// saved show and delete find a comparison by the first hex digits of its ID// saved show and delete find a comparison by the first hex digits of its ID
// (at least 6 to show, 12 to delete) or, to show, by its test video's path;
// saved list --json holds to its schema.
TEST(Schemas, SavedComparisonsAreNamedAndListed) {
    // An absolute path on each system; a path given in another case finds
    // its video on Windows only, as its file names are compared there.
#ifdef _WIN32
    const std::string clips = "Y:/clips/";
    const std::string other_case = "Y:/clips/A.mkv";
#else
    const std::string clips = "/clips/";
    const std::string other_case = "/clips/a.mkv";
#endif
    const auto comparison = [&clips](const std::string& id, const std::string& test) {
        SavedComparison saved;
        saved.folder = std::filesystem::path(clips + "cache/v2") / id;
        saved.source = clips + "reference.mkv";
        saved.distorted = clips + test;
        saved.bytes = 4096;
        saved.scores = {SavedScoreFile{saved.folder / "vmaf_1.npz", "vmaf", {}, 4096}};
        return saved;
    };
    const std::vector<SavedComparison> all{comparison(std::string(64, 'a'), "a.mkv"),
                                           comparison("ab" + std::string(62, 'c'), "b.mkv"),
                                           comparison(std::string(64, 'd'), "a.mkv")};
    EXPECT_EQ(saved_named(all, "AAAAAA", 6), std::vector<const SavedComparison*>{&all[0]});  // any case
    EXPECT_EQ(saved_named(all, "abcccc", 6), std::vector<const SavedComparison*>{&all[1]});
    EXPECT_TRUE(saved_named(all, "eeeeee", 6).empty());
    EXPECT_THROW(saved_named(all, "aaa", 6), UsageError);          // too short, and no path
    EXPECT_THROW(saved_named(all, "aaaaaaaaa", 12), UsageError);   // delete's 12
    EXPECT_EQ(saved_named(all, other_case, 6),                     // every comparison of that video
              (std::vector<const SavedComparison*>{&all[0], &all[2]}));

    const nlohmann::ordered_json listed = saved_list_json({&all[0], &all[1]});
    EXPECT_EQ(listed["comparisons"][1]["id"], all[1].folder.filename().string());
    EXPECT_EQ(listed["comparisons"][0]["metrics"], nlohmann::ordered_json::array({"vmaf"}));
    EXPECT_EQ(schema_errors("saved-list", listed), std::vector<std::string>{});

    // A score file of a metric this build doesn't know (saved by a newer
    // version sharing the folder, say) is listed by its key, as the Saved
    // Results tab lists it, rather than ending the list with an error.
    SavedComparison newer = comparison(std::string(64, 'e'), "Y:/clips/e.mkv");
    newer.scores.push_back(SavedScoreFile{newer.folder / "futuremetric_1.npz", "futuremetric", {}, 4096});
    std::string text;
    EXPECT_NO_THROW(text = saved_list_text({&all[0], &newer}));
    EXPECT_NE(text.find("futuremetric"), std::string::npos) << text;
}

// saved show and report take --segments saved as compare does: a wrong
// preset is a usage error (exit code 2, its message printed), not a failure
// (exit code 1, "Error: ..."), and a profile with no condition on the
// metrics there takes the fallback, said in a note as compare says it.
TEST(Schemas, SavedSegmentsOfScoresAreTakenAsCompareTakesThem) {
    std::vector<std::string> notes;
    const auto message = [&notes](const std::string& text) { notes.push_back(text); };
    CompareArguments wrong;
    wrong.segments_saved = true;
    wrong.segments_preset = "no-such-preset";
    EXPECT_THROW(take_saved_segments_of_scores(wrong, Settings{}, {"vmaf"}, message), UsageError);

    Settings settings;
    settings.weak_segment_conditions = {"ssimulacra2<70"};
    CompareArguments output;
    output.segments_saved = true;
    take_saved_segments_of_scores(output, settings, {"vmaf"}, message);
    ASSERT_EQ(output.segments.size(), 1U);
    EXPECT_EQ(output.segments[0].key, "vmaf");
    ASSERT_EQ(notes.size(), 2U);
    EXPECT_NE(notes[0].find("ssimulacra2<70 is left out"), std::string::npos) << notes[0];
    EXPECT_NE(notes[1].find("the profile has no condition on the metrics here"), std::string::npos) << notes[1];
}

// --stills picks each metric's worst dips (a frame two metrics chose once),
// or each weak segment's worst frame, the weakest segments first; the
// results' JSON lists them, as its schema says.
TEST(Schemas, StillsArePickedAndListed) {
    TestVideo video = scored_video();
    const ComparisonResult& result = *video.result;
    const std::vector<StillFrame> worst = still_frames(result, "worst", 1, SegmentRules{});
    ASSERT_EQ(worst.size(), 2U);  // in frame order: CVVDP's worst second, at frame 0
    EXPECT_EQ(worst[0].frame, 0);
    EXPECT_EQ(worst[0].metrics, std::vector<std::string>{"cvvdp"});
    EXPECT_EQ(worst[1].frame, 2);  // VMAF's 70 and PSNR's 38, both frame 2: one still
    EXPECT_EQ(worst[1].metrics, (std::vector<std::string>{"vmaf", "psnr"}));
    EXPECT_EQ(worst[1].from, "worst");

    const SegmentRules rules{{WeakCondition{"vmaf", '<', 85.0}, WeakCondition{"psnr", '<', 39.0}}};
    const std::vector<StillFrame> segments = still_frames(result, "segments", 5, rules);
    ASSERT_EQ(segments.size(), 1U);  // frames 1-2, worst by VMAF, the first condition
    EXPECT_EQ(segments[0].frame, 2);
    EXPECT_EQ(segments[0].metrics, std::vector<std::string>{"vmaf"});
    EXPECT_EQ(segments[0].segment, 0);
    EXPECT_DOUBLE_EQ(*segments[0].score, 70.0);
    EXPECT_TRUE(still_frames(result, "segments", 5, SegmentRules{{WeakCondition{"vmaf", '<', 50.0}}}).empty());

    video.stills = {VideoStill{worst[1], "s/a.frame000002.reference.png", "s/a.frame000002.test.png",
                               "s/a.frame000002.side-by-side.png"}};
    std::vector<TestVideo> videos{video};
    const nlohmann::ordered_json results = results_document(VideoInfo{}, videos, summarize(videos), false);
    EXPECT_EQ(results["videos"][0]["stills"][0]["metrics"], nlohmann::ordered_json::array({"vmaf", "psnr"}));
    EXPECT_EQ(schema_errors("results", results), std::vector<std::string>{});
}

// The stills of weak segments: each segment's worst frame is the worst of
// its own frames, by a scan of every frame here, for many segments whose
// edges fall anywhere (their first or last frame the worst, a gap merged
// into one, a NaN frame); a scan that started or stopped a frame wrong,
// or looked outside the segment, would pick another.
TEST(Schemas, SegmentStillsAreEachSegmentsWorstFrame) {
    ComparisonResult result;
    result.fps = 25.0;
    std::vector<int> frames;
    std::vector<double> times;
    std::vector<float> values;
    std::uint32_t state = 12345;  // a fixed sequence (a linear congruential generator)
    for (int frame = 0; frame < 3000; ++frame) {
        state = state * 1664525u + 1013904223u;
        frames.push_back(frame);
        times.push_back(frame / 25.0);
        values.push_back(frame % 97 == 5 ? std::numeric_limits<float>::quiet_NaN()
                                         : static_cast<float>(40 + (state >> 8) % 60));
    }
    result.compared_frame_count = 3000;
    result.metric_results.add(FrameMetricResult{"vmaf", frames, times, values, {}});
    SegmentRules rules{{WeakCondition{"vmaf", '<', 55.0}}};
    rules.merge_frames = 2;
    const std::vector<WeakSegment> segments = weak_segments(result, rules);
    ASSERT_GT(segments.size(), 100U);
    const std::vector<StillFrame> stills = still_frames(result, "segments", segments.size(), rules);
    ASSERT_EQ(stills.size(), segments.size());
    for (const StillFrame& still : stills) {
        const WeakSegment& segment = segments.at(static_cast<std::size_t>(*still.segment));
        float lowest = std::numeric_limits<float>::infinity();
        int at = -1;
        for (int frame = segment.first_frame; frame <= segment.last_frame; ++frame) {
            const float value = values[static_cast<std::size_t>(frame)];
            if (!std::isnan(value) && value < lowest) {
                lowest = value;
                at = frame;
            }
        }
        EXPECT_EQ(still.frame, at) << "segment " << segment.first_frame << "-" << segment.last_frame;
        EXPECT_DOUBLE_EQ(*still.score, lowest);
    }
}

// target's answers: each sequence's, answered or not (with why), by the
// rate or a column; the JSON holds to its schema.
TEST(Schemas, TargetsAreAnsweredAndListed) {
    RateTable table;
    table.metric_keys = {"vmaf"};
    table.columns = {"sequence", "kbps", "vmaf", "crf"};
    const auto encode = [](const char* sequence, double kbps, std::optional<double> vmaf, const char* crf) {
        return RateTable::Encode{sequence, kbps, {{"vmaf", vmaf}}, {{"crf", crf}}, ""};
    };
    table.encodes = {encode("park", 1000, 90, "30"), encode("park", 2000, 94, "26"), encode("park", 4000, 97, "22"),
                     encode("park", 3000, std::nullopt, "24"), encode("tree", 900, 91, "x")};
    TargetArguments arguments;
    arguments.scores = {"x264.csv"};
    arguments.metric = "vmaf";
    arguments.value = 95;
    const TargetAnswers rate = target_answers(arguments, table);
    EXPECT_EQ(rate.sequences, 2U);
    EXPECT_FALSE(rate.all_answered);  // tree has one encode
    const nlohmann::ordered_json& park = rate.document["sequences"][0];
    EXPECT_EQ(park["encodes"], 3);  // the one without a score left out, with a warning
    EXPECT_GT(park["setting"].get<double>(), 2000);
    EXPECT_LT(park["setting"].get<double>(), 4000);
    EXPECT_EQ(rate.document["sequences"][1]["why_code"], "too_few_encodes");
    EXPECT_EQ(rate.warnings.size(), 1U);
    EXPECT_EQ(schema_errors("target", rate.document), std::vector<std::string>{});

    arguments.by = "crf";
    arguments.sequence = "park";
    const TargetAnswers crf = target_answers(arguments, table);
    EXPECT_EQ(crf.sequences, 1U);
    EXPECT_TRUE(crf.all_answered);
    const double setting = crf.document["sequences"][0]["setting"].get<double>();
    EXPECT_GT(setting, 22);
    EXPECT_LT(setting, 26);
    EXPECT_EQ(schema_errors("target", crf.document), std::vector<std::string>{});
}

// diff matches weak segments by overlapping frames, chained: those of the
// old run nothing new overlaps are fixed, the new ones nothing old overlaps
// appeared; its JSON holds to its schema.
TEST(Schemas, DiffsMatchSegmentsAndHoldToTheirSchema) {
    const auto segment = [](int first, int last) {
        WeakSegment made;
        made.first_frame = first;
        made.last_frame = last;
        return made;
    };
    const SegmentMatch match = match_segments({segment(0, 3), segment(10, 12), segment(20, 22)},
                                              {segment(2, 11), segment(30, 31), segment(12, 14)});
    EXPECT_EQ(match.fixed, std::vector<std::size_t>{2});
    ASSERT_EQ(match.remaining.size(), 1U);
    EXPECT_EQ(match.remaining[0].old_segments, (std::vector<std::size_t>{0, 1}));
    EXPECT_EQ(match.remaining[0].new_segments, (std::vector<std::size_t>{0, 2}));
    EXPECT_EQ(match.appeared, std::vector<std::size_t>{1});

    const ComparisonResult old_result = *scored_video().result;
    ComparisonResult new_result = old_result;
    new_result.distorted = "b.mkv";
    new_result.metric_results.add(FrameMetricResult{"vmaf", {0, 1, 2}, {0.0, 1 / 24.0, 2 / 24.0}, {95, 90, 70}, {}});
    const SegmentRules rules{{WeakCondition{"vmaf", '<', 85.0}}};
    const nlohmann::ordered_json document = diff_json(old_result, "a.metrics.json", new_result, "3fa9c1", rules);
    const nlohmann::ordered_json& vmaf = document["metrics"]["vmaf"]["stats"];
    EXPECT_DOUBLE_EQ(vmaf["min"]["change"].get<double>(), 0.0);
    EXPECT_DOUBLE_EQ(vmaf["median"]["change"].get<double>(), 10.0);
    EXPECT_TRUE(document["metrics"]["psnr"]["stats"]["max"]["change"].is_null());  // inf to inf
    EXPECT_EQ(document["segments"]["remaining"][0]["old"], nlohmann::ordered_json::array({0}));
    EXPECT_EQ(document["segments"]["new"]["segments"][0]["first_frame"], 2);
    EXPECT_EQ(schema_errors("diff", document), std::vector<std::string>{});
    EXPECT_EQ(schema_errors("diff", diff_json(old_result, "a", new_result, "b", SegmentRules{})),
              std::vector<std::string>{});
}

// diff takes --segments saved as saved show and report do: a preset that
// does not exist is a usage error (exit code 2, its message printed), not a
// failure (exit code 1, "Error: ...").
TEST(Schemas, DiffTakesSavedSegmentsAsSavedShowDoes) {
    const std::filesystem::path folder = std::filesystem::temp_directory_path() / "vml-diff-segments-test";
    std::filesystem::create_directories(folder);
    const std::filesystem::path run = folder / "a.metrics.json";
    save_run(*scored_video().result, run, "a");
    SavedArguments arguments;
    arguments.action = "diff";
    arguments.items = {platform::path_text(run), platform::path_text(run)};
    arguments.output.segments_saved = true;
    arguments.output.segments_preset = "no-such-preset";
    arguments.output.quiet = true;
    EXPECT_THROW(diff(arguments, Settings{}), UsageError);
    std::filesystem::remove_all(folder);
}

}  // namespace
}  // namespace vml::cli
