// BD-rate as the VVenC test scripts calculate it (fasteval.py's bd_rate,
// scipy's PchipInterpolator): the expected values were printed by that
// function itself (scipy 1.17.1).
#include "vml/bd_rate.h"
#include "vml/bd_rate_report.h"

#include <gtest/gtest.h>

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
    result.distorted_info.bit_rate = 0;
    EXPECT_FALSE(rate_encode(result));
}

// numpy's nanmean and nanmedian.
TEST(BdRate, AveragesSequences) {
    EXPECT_EQ(mean_of({1.0, std::nullopt, 3.0}), 2.0);
    EXPECT_EQ(median_of({4.0, 1.0, std::nullopt, 3.0, 2.0}), 2.5);
    EXPECT_EQ(median_of({5.0, 1.0, 3.0}), 3.0);
    EXPECT_FALSE(mean_of({std::nullopt}));
}

}  // namespace
}  // namespace vml
