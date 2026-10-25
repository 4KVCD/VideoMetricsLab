// The results' summary (issue #8): the share of frames in each of a
// metric's bands, CVVDP's seconds' statistics, and the CSV's columns, one
// for each value any video has, in the order they first come.
#include "vml/results_summary.h"

#include <gtest/gtest.h>

namespace vml {
namespace {

TEST(ResultsSummary, HasTheBandsAndSecondsAndCsvColumnsOfEveryVideo) {
    ComparisonResult result;
    result.compared_frame_count = 4;
    result.metric_results.add(FrameMetricResult{"vmaf", {0, 1, 2, 3}, {0.0, 0.1, 0.2, 0.3}, {96, 92, 80, 60}, {}});
    result.metric_results.add(SequenceMetricResult{"cvvdp", 8.5, {}, {0, 2}, {0.0, 1.0}, {8.0f, 9.0f}});

    const nlohmann::ordered_json vmaf = *metric_summary(result, "vmaf");
    EXPECT_EQ(vmaf.at("score"), 82.0);
    EXPECT_EQ(vmaf.at("frames"), 4);
    EXPECT_EQ(vmaf.at("percent_of_frames"),
              (nlohmann::ordered_json{{"> 95", 25.0}, {"> 90", 50.0}, {"> 85", 50.0},
                                      {"< 85", 50.0}, {"< 80", 25.0}, {"< 70", 25.0}}));
    const nlohmann::ordered_json cvvdp = *metric_summary(result, "cvvdp");
    EXPECT_EQ(cvvdp.at("score"), 8.5);
    EXPECT_EQ(cvvdp.at("per_second").at("seconds"), 2);
    EXPECT_EQ(cvvdp.at("per_second").at("min"), 8.0);
    EXPECT_FALSE(metric_summary(result, "psnr").has_value());

    const nlohmann::ordered_json document{
        {"videos",
         {{{"path", "a,b.mkv"}, {"metrics", {{"vmaf", {{"score", 82.0}}}}}},
          {{"path", "c.mkv"}, {"label", "say \"hi\""}, {"reference_crop", nullptr}}}}};
    EXPECT_EQ(results_csv(document), "\xEF\xBB\xBFpath,metrics.vmaf.score,label,reference_crop\r\n"
                                     "\"a,b.mkv\",82.0,,\r\n"
                                     "c.mkv,,\"say \"\"hi\"\"\",\r\n");
}

}  // namespace
}  // namespace vml
