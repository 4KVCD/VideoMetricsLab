// Export results (results_export.h): CVVDP's further displays are listed in
// the entry of the video they were scored for, not as videos of their own.
#include <gtest/gtest.h>

#include <QTemporaryDir>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "results_export.h"
#include "vml/bd_rate_report.h"
#include "vml/python_json.h"
#include "vml/cvvdp.h"
#include "vml/export_csv.h"
#include "vml/platform/system.h"
#include "vml/segment_profiles.h"
#include "vml/settings.h"
#include "vml/weak_frames.h"

namespace vml::app {
namespace {

// A result whose CVVDP score of `score` was made for the preset `name`.
std::shared_ptr<const ComparisonResult> cvvdp_on(const std::string& name, double score) {
    const nlohmann::ordered_json parameters = cvvdp_preset_named(name, "")->settings.parameters();
    MetricProvenance made_for;
    for (const auto& [key, value] : parameters.items()) {
        made_for.parameters[key] = value.dump();
    }
    auto result = std::make_shared<ComparisonResult>();
    result->metric_results.add(SequenceMetricResult{"cvvdp", score, made_for, {0}, {0.0}, {static_cast<float>(score)}});
    return result;
}

TEST(ResultsExport, FurtherDisplaysAreListedWithTheirVideo) {
    const std::string sdr(sdr_cvvdp_preset);
    const std::string hdr(hdr_cvvdp_preset);
    const ExportedRun run{cvvdp_on(sdr, 9.5), QStringLiteral("encode"), {}, sdr, {{hdr, cvvdp_on(hdr, 8.25)}}};
    const nlohmann::ordered_json document = export_document(std::nullopt, {run}, {});
    ASSERT_EQ(document.at("videos").size(), 1U);
    const nlohmann::ordered_json& entry = document.at("videos").at(0);
    EXPECT_EQ(entry.at("cvvdp_display"), sdr);
    EXPECT_EQ(entry.at("metrics").at("cvvdp").at("score"), 9.5);
    const nlohmann::ordered_json& displays = entry.at("cvvdp_displays");
    ASSERT_EQ(displays.size(), 2U);
    EXPECT_EQ(displays.at(0).at("display"), sdr);
    EXPECT_EQ(displays.at(0).at("cvvdp"), entry.at("metrics").at("cvvdp"));
    EXPECT_EQ(displays.at(1).at("display"), hdr);
    EXPECT_EQ(displays.at(1).at("cvvdp").at("score"), 8.25);
    EXPECT_EQ(displays.at(1).at("display_parameters"),
              nlohmann::ordered_json(cvvdp_preset_named(hdr, "")->settings.parameters()));

    // Without CVVDP, no list.
    const ExportedRun plain{std::make_shared<ComparisonResult>(), QStringLiteral("plain")};
    EXPECT_FALSE(export_document(std::nullopt, {plain}, {}).at("videos").at(0).contains("cvvdp_displays"));
}

// PR 124's review: BD-rate's import of an exported document takes the
// encodes rate_encode takes, also from runs without facts (loaded from run
// files or Saved Results): not a resolution test (at the reference's
// bitrate), scores of a range, or another row's take on the same file.
TEST(ResultsExport, BdRatesImportTakesWhatRateEncodeTakes) {
    const auto scored = [](const std::string& test, long long rate) {
        auto result = std::make_shared<ComparisonResult>();
        result->source = "Y:/clips/park_joy_1080p.y4m";
        result->distorted = test;
        result->distorted_info.bit_rate = rate;
        result->metric_results.add(FrameMetricResult{"vmaf", {0, 1}, {0.0, 0.04}, {90.0F, 92.0F}, {}});
        return result;
    };
    const auto encode = scored("Y:/encodes/park_q27.mkv", 4000000);
    auto resolution_test = scored("Y:/clips/park_joy_1080p [720p].y4m", 80000000);  // the reference's rate
    resolution_test->resample_target = ResampleTarget{1280, "720p"};
    auto ranged = scored("Y:/encodes/park_q30.mkv", 3000000);
    FrameMetricResult part{"psnr", {0, 1}, {0.0, 0.04}, {40.0F, 41.0F}, {}};
    part.provenance.parameters["range_frames"] = "[0,1000]";
    ranged->metric_results.add(part);
    const auto twin = scored("Y:/encodes/park_q27 [without film grain].mkv", 4000000);
    const std::vector<ExportedRun> runs{
        {encode, QStringLiteral("q27")},
        {resolution_test, QStringLiteral("720p")},
        {ranged, QStringLiteral("q30")},
        {twin, QStringLiteral("q27 without"), {}, {}, {}, {}, {}, std::filesystem::path("Y:/encodes/park_q27.mkv")}};
    const nlohmann::ordered_json document = export_document(std::nullopt, runs, {});
    const nlohmann::ordered_json& videos = document.at("videos");
    for (const nlohmann::ordered_json& video : videos) {
        EXPECT_TRUE(video.at("settings").is_null());  // no facts: as loaded
    }
    EXPECT_EQ(videos.at(0).at("whole_file"), true);
    EXPECT_TRUE(videos.at(0).at("resolution_test").is_null());
    EXPECT_TRUE(videos.at(0).at("variant_of").is_null());
    EXPECT_EQ(videos.at(1).at("resolution_test").at("label"), "720p");
    EXPECT_EQ(videos.at(2).at("whole_file"), false);
    EXPECT_EQ(videos.at(3).at("variant_of"), "Y:/encodes/park_q27.mkv");
    const RateTable table = read_rate_file(python_json_dumps(document), "results.json");
    ASSERT_EQ(table.encodes.size(), 1U);
    EXPECT_EQ(table.encodes[0], *rate_encode(*encode));
}

// Issue #14: the document says which of the window's runs its videos came
// from, as one, and each video how it was made, where that is known.
TEST(ResultsExport, RecordsTheRunsAsOneAndEachVideosFacts) {
    MachineFacts earlier_pc;
    earlier_pc.cpu = "earlier";
    MachineFacts later_pc;
    later_pc.cpu = "later";
    const auto later = std::make_shared<const RunRecord>(
        RunRecord{"gui", std::nullopt, "2026-11-14T10:00:00Z", "2026-11-14T10:05:00Z", 300.0, 10.0, 1, later_pc});
    const auto earlier = std::make_shared<const RunRecord>(
        RunRecord{"gui", std::nullopt, "2026-11-14T09:00:00Z", "2026-11-14T09:01:00Z", 60.0, 0.0, 1, earlier_pc});
    auto facts = std::make_shared<RunFacts>();
    facts->settings.metrics = {"psnr"};
    facts->settings.options.crop_mode = CropMode::none;
    facts->started_utc = "2026-11-14T10:00:01Z";
    facts->passes = {PassTiming{"ffmpeg", {"psnr"}, "cpu", 4.0, 0.0, 1.0, 48, 16.0, "done", std::nullopt, {}}};
    facts->metrics = {{"psnr", MetricOutcome{metric_calculated, ""}}};
    const auto result = std::make_shared<const ComparisonResult>();
    // Two videos of the later run, one of the earlier (a saved run loaded:
    // no facts).
    const std::vector<ExportedRun> runs{{result, QStringLiteral("a"), {}, {}, {}, facts, later},
                                        {result, QStringLiteral("b"), {}, {}, {}, facts, later},
                                        {result, QStringLiteral("c"), {}, {}, {}, nullptr, earlier}};
    const nlohmann::ordered_json document = export_document(std::nullopt, runs, {});
    const nlohmann::ordered_json& run = document.at("run");
    EXPECT_EQ(run.at("runs"), 2);
    EXPECT_EQ(run.at("started_utc"), "2026-11-14T09:00:00Z");
    EXPECT_EQ(run.at("finished_utc"), "2026-11-14T10:05:00Z");
    EXPECT_EQ(run.at("wall_seconds"), 360.0);
    EXPECT_EQ(run.at("paused_seconds"), 10.0);
    EXPECT_EQ(document.at("machine").at("cpu"), "later");

    const nlohmann::ordered_json& made = document.at("videos").at(0);
    EXPECT_EQ(made.at("metric_status").at("psnr").at("status"), "calculated");
    EXPECT_TRUE(made.at("settings").at("vmaf_model").is_null());  // VMAF not asked
    EXPECT_EQ(made.at("timings").at("passes").at(0).at("fps"), 16.0);
    EXPECT_TRUE(document.at("videos").at(2).at("settings").is_null());

    // While a run is still going, the runs have no end yet.
    RunRecord unfinished;
    unfinished.user_interface = "gui";
    unfinished.started_utc = "2026-11-14T11:00:00Z";
    const auto running = std::make_shared<const RunRecord>(unfinished);
    const nlohmann::ordered_json going =
        export_document(std::nullopt, {{result, QStringLiteral("d"), {}, {}, {}, facts, running}}, earlier_pc);
    EXPECT_TRUE(going.at("run").at("finished_utc").is_null());
    EXPECT_TRUE(going.at("run").at("wall_seconds").is_null());
    EXPECT_EQ(going.at("machine").at("cpu"), "earlier");  // none finished: the machine given
}

// An export's files besides its graphs: results and weak segments in each
// format ticked, and only those (CSV unticked, no weak segments CSV); with
// no format, none; weak segments of several videos, one report; per-frame
// scores, a file for each video.
TEST(ResultsExport, FilesAreWrittenInTheFormatsTicked) {
    auto result = std::make_shared<ComparisonResult>();
    FrameMetricResult vmaf;
    vmaf.key = "vmaf";
    vmaf.frame = {0, 1, 2};
    vmaf.time = {0.0, 1 / 24.0, 2 / 24.0};
    vmaf.values = {95.0F, 60.0F, 95.0F};
    result->metric_results.add(vmaf);
    const std::vector<BundleVideo> one{{result, QStringLiteral("a"), "a.mkv"}};
    const nlohmann::ordered_json document{{"videos", nlohmann::ordered_json::array()}};
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<90"};
    const SegmentRules rules = segment_profile_in_use(settings).rules;
    ASSERT_FALSE(rules.conditions.empty());
    std::vector<FrameFiles> frames;
    const auto names = [&](const std::vector<BundleVideo>& videos, bool several) {
        bool skipped = true;
        std::vector<std::string> found;
        for (const BundleFile& file : export_files(settings, videos, "r.mkv", [&document] { return document; }, rules,
                                                   several, QStringLiteral("a"), frames, skipped)) {
            found.push_back(file.first);
        }
        EXPECT_FALSE(skipped);
        return found;
    };
    using Names = std::vector<std::string>;
    const std::string suffix = segments_file_suffix(rules);
    settings.export_formats = {"json"};
    EXPECT_EQ(names(one, false), (Names{"a - results.json", "a." + suffix + ".json"}));
    settings.export_formats = {"csv"};
    EXPECT_EQ(names(one, false), (Names{"a - results.csv", "a." + suffix + ".csv"}));
    settings.export_formats = {"json", "csv"};
    settings.export_items = {"weak_segments"};
    EXPECT_EQ(names({one[0], {result, QStringLiteral("b"), "b.mkv"}}, true),
              (Names{"a - weak segments." + suffix + ".json", "a - weak segments." + suffix + ".csv"}));
    // Per-frame scores: a file for each video, named as its graphs are. Issue
    // #140: made only as they are written (made_files, in the file-write
    // job), not on the window's thread as the export is planned.
    settings.export_items = {"frames"};
    EXPECT_EQ(names({one[0], {result, QStringLiteral("b"), "b.mkv"}}, true), Names{});
    ASSERT_EQ(frames.size(), 2U);
    const std::vector<BundleFile> made = made_files({{"x.json", "x"}}, frames);
    Names made_names;
    for (const BundleFile& file : made) {
        made_names.push_back(file.first);
    }
    EXPECT_EQ(made_names, (Names{"x.json", "a - frames.json", "a - frames.csv", "b - frames.json", "b - frames.csv"}));
    EXPECT_EQ(made[1].second, per_frame_json(*result));
    EXPECT_EQ(made[2].second, per_frame_csv(*result));
    settings.export_items = export_item_keys();
    settings.export_formats.clear();
    EXPECT_EQ(names(one, false), Names{});
    EXPECT_EQ(exported_items(settings), Names{"graphs"});
}

// Issue #131: the after-run export names its files after the test video's
// file name, which another test video's may share (x265\movie.mkv and
// svt\movie.mkv, or clip.mkv and clip.mp4): the second replaced the first's
// files. Now an export replaces none, and its files are numbered alike.
TEST(ResultsExport, AnExportIntoAFolderReplacesNoneOfAnothersFiles) {
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const std::filesystem::path folder = platform::path_from_text(temporary.path().toStdString());
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    };
    const std::vector<BundleFile> first{{"movie - results.json", "x265"}, {"movie - VMAF.png", "x265 graph"}};
    EXPECT_EQ(write_loose_export(folder, first),
              (std::vector<std::filesystem::path>{folder / "movie - results.json", folder / "movie - VMAF.png"}));
    // Only one of the second's names is taken: both are numbered.
    const std::vector<BundleFile> second{{"movie - results.json", "svt"}, {"movie - frames.csv", "svt frames"}};
    EXPECT_EQ(write_loose_export(folder, second),
              (std::vector<std::filesystem::path>{folder / "movie - results_2.json", folder / "movie - frames_2.csv"}));
    EXPECT_EQ(read(folder / "movie - results.json"), "x265");
    EXPECT_EQ(read(folder / "movie - results_2.json"), "svt");
    EXPECT_FALSE(std::filesystem::exists(folder / "movie - frames.csv"));
}

// Issue #132: a profile where every condition must hold, one on a metric the
// video was not scored with. The export leaves that one out, as the Weak
// segments dialog and vml-cli --segments saved do, and says so; it found no
// segment at all before.
TEST(ResultsExport, WeakSegmentsLeaveOutAMetricNotScored) {
    auto result = std::make_shared<ComparisonResult>();
    FrameMetricResult vmaf;
    vmaf.key = "vmaf";
    vmaf.frame = {0, 1, 2};
    vmaf.time = {0.0, 1 / 24.0, 2 / 24.0};
    vmaf.values = {95.0F, 60.0F, 95.0F};
    result->metric_results.add(vmaf);
    Settings settings;
    settings.export_items = {"weak_segments"};
    settings.export_formats = {"json"};
    SegmentRules rules{{{"vmaf", '<', 90.0}, {"cvvdp", '<', 9.0}}};
    rules.all = true;
    bool skipped = true;
    std::vector<FrameFiles> frames;
    const std::vector<BundleFile> files =
        export_files(settings, {{result, QStringLiteral("a"), "a.mkv"}}, "r.mkv",
                     [] { return nlohmann::ordered_json::object(); }, rules, false, QStringLiteral("a"), frames, skipped);
    EXPECT_FALSE(skipped);
    ASSERT_EQ(files.size(), 1U);
    EXPECT_EQ(files[0].first, "a.vmaf.segments.json");  // as the dialog names it
    const nlohmann::ordered_json json = nlohmann::ordered_json::parse(files[0].second);
    EXPECT_EQ(json["conditions"], nlohmann::ordered_json::array({"vmaf<90"}));
    EXPECT_EQ(json["left_out"], nlohmann::ordered_json::array({"cvvdp<9"}));
    ASSERT_EQ(json["segments"].size(), 1U);
    EXPECT_EQ(json["segments"][0]["first_frame"], 1);
}

}  // namespace
}  // namespace vml::app
