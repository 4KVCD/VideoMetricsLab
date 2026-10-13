#include "vml/stats.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "reference.h"

namespace vml {
namespace {

MetricAggregation aggregation_named(const std::string& name) {
    return name == "square_mean_root" ? MetricAggregation::square_mean_root_db : MetricAggregation::arithmetic;
}

MetricDirection direction_named(const std::string& name) {
    return name == "lower_is_better" ? MetricDirection::lower_is_better : MetricDirection::higher_is_better;
}

void expect_same(const nlohmann::json& expected, double actual, const char* what) {
    const double value = *test::reference_number(expected);
    EXPECT_TRUE(test::same_number(value, actual)) << what << ": Python " << expected << ", C++ " << actual;
}

// Every statistic the app shows, over the shapes of data it meets (see
// make_reference.py), is the Python app's.
TEST(Stats, ComputeStatsGivesThePythonAppsStatistics) {
    const nlohmann::json reference = test::load_reference("stats.json");
    ASSERT_FALSE(reference["cases"].empty());
    for (const nlohmann::json& item : reference["cases"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        const std::vector<double> values = test::reference_numbers(reference["shapes"][item["shape"].get<std::string>()]);
        std::vector<Threshold> thresholds;
        for (const nlohmann::json& band : item["thresholds"]) {
            thresholds.push_back({band[0].get<std::string>()[0], *test::reference_number(band[1])});
        }
        const MetricAggregation aggregation = aggregation_named(item["aggregation"].get<std::string>());
        const ScoreStats stats =
            compute_stats(values, thresholds, aggregation, direction_named(item["direction"].get<std::string>()));

        const nlohmann::json& expected = item["expected"];
        EXPECT_EQ(stats.count, expected["count"].get<std::size_t>());
        EXPECT_EQ(stats.identical, expected["identical"].get<std::size_t>());
        EXPECT_EQ(stats.worst_is_high, expected["worst_is_high"].get<bool>());
        expect_same(expected["mean"], stats.mean, "mean");
        expect_same(expected["median"], stats.median, "median");
        expect_same(expected["stdev"], stats.stdev, "stdev");
        expect_same(expected["minimum"], stats.minimum, "minimum");
        expect_same(expected["maximum"], stats.maximum, "maximum");
        expect_same(expected["worst_10_percent"], stats.worst_10_percent, "worst 10%");
        expect_same(expected["worst_5_percent"], stats.worst_5_percent, "worst 5%");
        expect_same(expected["worst_1_percent"], stats.worst_1_percent, "worst 1%");
        expect_same(expected["worst_0_1_percent"], stats.worst_0_1_percent, "worst 0.1%");
        ASSERT_EQ(stats.thresholds.size(), expected["thresholds"].size());
        for (std::size_t t = 0; t < stats.thresholds.size(); ++t) {
            EXPECT_EQ(stats.thresholds[t].count, expected["thresholds"][t]["count"].get<std::size_t>());
            expect_same(expected["thresholds"][t]["percentage"], stats.thresholds[t].percentage, "band percentage");
        }

        const std::optional<double> aggregate = aggregate_scores(values, aggregation);
        ASSERT_EQ(aggregate.has_value(), !item["aggregate"].is_null());
        if (aggregate) {
            expect_same(item["aggregate"], *aggregate, "aggregate");
        }
    }
}

}  // namespace
}  // namespace vml
