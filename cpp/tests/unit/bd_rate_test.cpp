// BD-rate as the VVenC test scripts calculate it (fasteval.py's bd_rate,
// scipy's PchipInterpolator): the expected values were printed by that
// function itself (scipy 1.17.1).
#include "vml/bd_rate.h"
#include "vml/bd_rate_report.h"
#include "vml/platform/system.h"
#include "vml/run_file.h"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vml {
namespace {

// A curve from its rates and scores.
std::vector<RatePoint> curve(const std::vector<double>& kbps, const std::vector<double>& scores) {
    std::vector<RatePoint> points;
    for (std::size_t index = 0; index < kbps.size(); ++index) {
        points.push_back({kbps[index], scores[index]});
    }
    return points;
}

double percent(const std::vector<RatePoint>& anchor, const std::vector<RatePoint>& test) {
    const BdRateResult result = bd_rate(anchor, test);
    EXPECT_EQ(result.gap, BdRateGap::none);
    return result.percent.value_or(1e9);
}

// fasteval's numbers to 1e-9: 4 QPs; 5 unsorted points with a slope that
// changes sign and a flat secant; 2 points (a straight line); 3 points where
// an end slope is held to three times its secant; half the rate at every
// score; identical curves exactly 0.
TEST(BdRate, GivesTheVvencScriptsNumbers) {
    EXPECT_NEAR(percent(curve({1000, 1800, 3200, 6000}, {80.0, 86.5, 91.0, 94.8}),
                        curve({950, 1700, 3100, 5900}, {80.4, 86.9, 91.3, 95.0})),
                -8.354292889192227, 1e-9);
    EXPECT_NEAR(percent(curve({4000, 900, 2000, 2000, 7000}, {90.0, 70.0, 82.0, 85.0, 95.0}),
                        curve({1200, 3000, 6500, 800, 2200}, {78.0, 89.0, 94.0, 69.0, 83.5})),
                -14.843652285311403, 1e-9);
    EXPECT_NEAR(percent(curve({1000, 4000}, {30.0, 36.0}), curve({1100, 4200}, {30.5, 36.5})), -4.06852237532116,
                1e-9);
    EXPECT_NEAR(percent(curve({1000, 1100, 5000}, {30.0, 35.0, 36.0}), curve({1000, 3000, 3100}, {31.0, 32.0, 37.0})),
                135.67590820297738, 1e-9);
    const std::vector<RatePoint> anchor = curve({1000, 2000, 4000}, {30, 33, 36});
    EXPECT_NEAR(percent(anchor, curve({500, 1000, 2000}, {30, 33, 36})), -50.0, 1e-9);
    EXPECT_EQ(percent(anchor, anchor), 0.0);
}

// Lower is better (Butteraugli): the scores negated, so a test with lower
// scores at the same rates needs less rate.
TEST(BdRate, NegatesScoresWhereLowerIsBetter) {
    const BdRateResult result =
        bd_rate(curve({1000, 2000, 4000}, {3.0, 2.0, 1.0}), curve({1000, 2000, 4000}, {2.5, 1.5, 0.5}), true);
    ASSERT_TRUE(result.percent);
    EXPECT_LT(*result.percent, 0);
}

// No BD-rate, and why: a tie, one point, no common scores, a rate of 0.
TEST(BdRate, SaysWhyThereIsNone) {
    const std::vector<RatePoint> anchor = curve({1000, 2000, 4000}, {30, 33, 36});
    EXPECT_EQ(bd_rate(anchor, curve({1000, 2000, 4000}, {30, 33, 33})).gap, BdRateGap::tied_scores);
    EXPECT_EQ(bd_rate(anchor, curve({1000}, {30})).gap, BdRateGap::too_few_points);
    EXPECT_EQ(bd_rate(anchor, curve({1000, 2000}, {40, 45})).gap, BdRateGap::no_overlap);
    EXPECT_EQ(bd_rate(anchor, curve({0, 2000}, {30, 36})).gap, BdRateGap::bad_number);
    EXPECT_FALSE(bd_rate(anchor, curve({1000, 2000, 4000}, {30, 33, 33})).percent);
}

// The drawn curve: through the points at its ends and scipy's value between
// them (printed by PchipInterpolator, as above); for lower is better, its
// scores the real ones, highest first; none where there is no curve.
TEST(BdRate, DrawsTheCurveItIntegrates) {
    const std::vector<RatePoint> line = pchip_curve(curve({1000, 1800, 3200, 6000}, {80.0, 86.5, 91.0, 94.8}), false, 3);
    ASSERT_EQ(line.size(), 3u);
    EXPECT_NEAR(line[0].kbps, 1000, 1e-9);
    EXPECT_NEAR(line[1].kbps, 1990.605200701846, 1e-9);
    EXPECT_NEAR(line[1].score, 87.4, 1e-12);
    EXPECT_NEAR(line[2].kbps, 6000, 1e-9);
    const std::vector<RatePoint> lower = pchip_curve(curve({1000, 2000, 4000}, {3.0, 2.0, 1.0}), true, 5);
    ASSERT_EQ(lower.size(), 5u);
    EXPECT_EQ(lower[0].score, 3.0);
    EXPECT_NEAR(lower[1].kbps, 1414.2135623730949, 1e-9);
    EXPECT_NEAR(lower[1].score, 2.5, 1e-12);
    EXPECT_TRUE(pchip_curve(curve({1000, 2000}, {30, 30}), false, 5).empty());
}

// A scored video as an encode: its reference's name, its own bitrate in
// kb/s, each metric's score as the app shows it; none without a bitrate.
TEST(BdRate, TakesAnEncodeFromAScoredVideo) {
    ComparisonResult result;
    result.source = "videos/baseball_1920x1080_60.y4m";
    result.distorted = "videos/baseball_q27.mkv";
    result.distorted_info.bit_rate = 6551340;
    result.metric_results.add(FrameMetricResult{"vmaf", {0, 1}, {0.0, 0.04}, {87.0F, 89.0F}, {}});
    result.metric_results.add(SequenceMetricResult{"cvvdp", 9.67, {}, {}, {}, {}});
    const std::optional<RateTable::Encode> encode = rate_encode(result);
    ASSERT_TRUE(encode);
    EXPECT_EQ(encode->sequence, "baseball_1920x1080_60");
    EXPECT_DOUBLE_EQ(encode->kbps, 6551.34);
    EXPECT_DOUBLE_EQ(*encode->scores.at("vmaf"), 88.0);
    EXPECT_DOUBLE_EQ(*encode->scores.at("cvvdp"), 9.67);
    EXPECT_FALSE(encode->scores.contains("psnr"));
    // The file's bitrate does not go with scores of a part of it, or of a
    // resolution test's picture.
    ComparisonResult ranged = result;
    FrameMetricResult part{"psnr", {0, 1}, {0.0, 0.04}, {40.0F, 41.0F}, {}};
    part.provenance.parameters["range_frames"] = "[0,2]";
    ranged.metric_results.add(part);
    EXPECT_FALSE(rate_encode(ranged));
    ComparisonResult resized = result;
    resized.resample_target = ResampleTarget{};
    EXPECT_FALSE(rate_encode(resized));
    result.distorted_info.bit_rate = 0;
    EXPECT_FALSE(rate_encode(result));
}

// A JSON file's encodes: a results document's videos (the command line's
// --json, a --batch's, Export results'), or a run file's one video; a CSV
// file as read_rate_table reads it.
TEST(BdRate, ReadsEncodesFromJson) {
    const nlohmann::json video = {
        {"path", "Y:/encodes/park_q27.mkv"}, {"reference", "Y:/clips/park_joy_1080p.y4m"}, {"status", "ok"},
        {"bit_rate", 4250500}, {"metrics", {{"vmaf", {{"score", 91.5}}}, {"psnr", {{"score", "inf"}}}}}};
    nlohmann::json failed = video;
    failed["status"] = "failed";
    nlohmann::json ranged = video;
    ranged["settings"] = {{"range", {{"whole", false}, {"start", 0}, {"end", 10}, {"unit", "seconds"}}}};
    nlohmann::json whole = video;
    whole["settings"] = {{"range", {{"whole", true}, {"start", 0}, {"end", 0}, {"unit", "seconds"}}}};
    nlohmann::json no_rate = video;
    no_rate.erase("bit_rate");
    const nlohmann::json document = {{"format", 2}, {"videos", {video, failed, ranged, whole, no_rate}}};
    const RateTable table = read_rate_file(" \n" + document.dump(), "results.json");
    ASSERT_EQ(table.encodes.size(), 2U);  // the first, and the whole file's
    EXPECT_EQ(table.encodes[0].sequence, "park_joy_1080p");
    EXPECT_EQ(table.encodes[0].reference, "Y:/clips/park_joy_1080p.y4m");
    EXPECT_DOUBLE_EQ(table.encodes[0].kbps, 4250.5);
    EXPECT_DOUBLE_EQ(*table.encodes[0].scores.at("vmaf"), 91.5);
    EXPECT_TRUE(std::isinf(*table.encodes[0].scores.at("psnr")));
    EXPECT_EQ(table.metric_keys, (std::vector<std::string>{"vmaf", "psnr"}));
    EXPECT_EQ(table.columns, (std::vector<std::string>{"sequence", "kbps", "vmaf", "psnr"}));

    // --batch: each reference's document, one that could not be read too.
    const nlohmann::json batch = {
        {"batch", {document, {{"reference", "Y:/clips/gone.y4m"}, {"error", "not found"}}, document}}};
    EXPECT_EQ(read_rate_file("\xEF\xBB\xBF" + batch.dump(), "batch.json").encodes.size(), 4U);

    // A run file: its video, as rate_encode takes it.
    ComparisonResult result;
    result.source = "Y:/clips/baseball_1920x1080_60.y4m";
    result.distorted = "Y:/encodes/baseball_q27.mkv";
    result.distorted_info.bit_rate = 6551340;
    result.metric_results.add(FrameMetricResult{"vmaf", {0, 1}, {0.0, 0.04}, {87.0F, 89.0F}, {}});
    const RateTable run = read_rate_file(run_file_text(result, "baseball_q27"), "baseball_q27.metrics.json");
    ASSERT_EQ(run.encodes.size(), 1U);
    const RateTable::Encode expected = *rate_encode(result);
    EXPECT_EQ(run.encodes[0].sequence, expected.sequence);
    // The same path: a run file saves it with the system's separators.
    EXPECT_EQ(platform::path_from_text(run.encodes[0].reference).make_preferred(),
              platform::path_from_text(expected.reference).make_preferred());
    EXPECT_DOUBLE_EQ(run.encodes[0].kbps, expected.kbps);
    EXPECT_DOUBLE_EQ(*run.encodes[0].scores.at("vmaf"), *expected.scores.at("vmaf"));
    // target --stat: a statistic of the summary, of a run file's frames or a
    // results document's entry (none where it has not got it), not of a CSV.
    const RateTable lowest = read_rate_file(run_file_text(result, "baseball_q27"), "b.metrics.json", "min");
    EXPECT_DOUBLE_EQ(*lowest.encodes[0].scores.at("vmaf"), 87.0);
    EXPECT_FALSE(read_rate_file(document.dump(), "results.json", "min").encodes[0].scores.at("vmaf"));
    EXPECT_THROW(read_rate_file("sequence,kbps,vmaf\nclip,100,90\n", "a.csv", "min"), std::runtime_error);

    // CSV as ever; and a file with nothing to go by says so, by its name.
    EXPECT_EQ(read_rate_file("sequence,kbps,vmaf\nclip,100,90\n", "a.csv").encodes.size(), 1U);
    try {
        read_rate_file(nlohmann::json{{"videos", {failed}}}.dump(), "empty.json");
        ADD_FAILURE() << "no encodes, no error";
    } catch (const std::runtime_error& error) {
        EXPECT_TRUE(std::string(error.what()).starts_with("empty.json: ")) << error.what();
    }
    EXPECT_THROW(read_rate_file("{not json", "broken.json"), std::runtime_error);
}

// target reads the curve bdrate integrates (the log of the rate against the
// score, PCHIP) at a score: between the two encodes either side, at an
// encode's own score, or past the ends on the straight line through the two
// nearest; a setting such as CRF as it is, not its log.
TEST(BdRate, FindsTheSettingForAScore) {
    const std::vector<SettingPoint> rates{{2000, 94}, {1000, 90}, {8000, 98.5}, {4000, 97}};  // in any order
    const TargetResult between = setting_for_score(rates, 95, true);
    ASSERT_TRUE(between.setting);
    EXPECT_FALSE(between.extrapolated);
    EXPECT_EQ(between.bracket, (std::pair<std::size_t, std::size_t>{0, 3}));
    // The rate bdrate's curve has there: pchip_curve's at 95, its 11th of
    // 18 points from 90 to 98.5
    const std::vector<RatePoint> drawn =
        pchip_curve(curve({2000, 1000, 8000, 4000}, {94, 90, 98.5, 97}), false, 18);
    ASSERT_DOUBLE_EQ(drawn[10].score, 95);
    EXPECT_NEAR(*between.setting, drawn[10].kbps, 1e-9);

    const TargetResult exact = setting_for_score(rates, 94, true);
    EXPECT_NEAR(*exact.setting, 2000, 1e-9);
    EXPECT_EQ(exact.bracket, (std::pair<std::size_t, std::size_t>{0, 0}));

    const TargetResult past = setting_for_score(rates, 99.25, true);
    EXPECT_TRUE(past.extrapolated);
    EXPECT_EQ(past.bracket, (std::pair<std::size_t, std::size_t>{3, 2}));
    EXPECT_NEAR(*past.setting, 4000 * std::pow(2.0, 1.5), 1e-6);  // 1.5 doublings further
    EXPECT_NEAR(*setting_for_score(rates, 86, true).setting, 500, 1e-9);  // before the first: one halving

    const std::vector<SettingPoint> crf{{30, 90}, {26, 94}};
    EXPECT_NEAR(*setting_for_score(crf, 92, false).setting, 28, 1e-12);

    EXPECT_EQ(setting_for_score(std::vector<SettingPoint>{{1000, 90}}, 95, true).gap, BdRateGap::too_few_points);
    EXPECT_EQ(setting_for_score(std::vector<SettingPoint>{{1000, 90}, {2000, 90}}, 95, true).gap,
              BdRateGap::tied_scores);
    EXPECT_EQ(setting_for_score(std::vector<SettingPoint>{{0, 90}, {2000, 94}}, 92, true).gap, BdRateGap::bad_number);
    EXPECT_TRUE(setting_for_score(std::vector<SettingPoint>{{0, 90}, {2000, 94}}, 92, false).setting);  // CRF 0
}

// numpy's nanmean and nanmedian.
TEST(BdRate, AveragesSequences) {
    EXPECT_EQ(mean_of({1.0, std::nullopt, 3.0}), 2.0);
    EXPECT_EQ(median_of({4.0, 1.0, std::nullopt, 3.0, 2.0}), 2.5);
    EXPECT_EQ(median_of({5.0, 1.0, 3.0}), 3.0);
    EXPECT_FALSE(mean_of({std::nullopt}));
}

// --scores-csv's table, read back by bdrate as it was written.
TEST(BdRate, AScoresTableReadsBackAsWritten) {
    RateTable table;
    table.metric_keys = {"vmaf", "psnr"};
    table.columns = {"sequence", "kbps", "vmaf", "psnr", "test", "class"};
    RateTable::Encode encode;
    encode.sequence = "clip, one";
    encode.kbps = 1311.154;
    encode.scores = {{"vmaf", 93.123456789}, {"psnr", std::nullopt}};
    encode.text = {{"test", "C:\\a \"b\".mkv"}, {"class", ""}};
    table.encodes = {encode, encode};
    table.encodes[1].kbps = 0.5;
    const RateTable read = read_rate_table(rate_table_csv(table), "scores.csv");
    EXPECT_EQ(read.metric_keys, table.metric_keys);
    EXPECT_EQ(read.columns, table.columns);
    EXPECT_EQ(read.encodes, table.encodes);
}

}  // namespace
}  // namespace vml
