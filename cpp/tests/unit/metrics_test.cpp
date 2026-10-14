#include "vml/metrics.h"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "reference.h"

namespace vml {
namespace {

std::string kind_name(MetricKind kind) { return kind == MetricKind::frame ? "frame" : "sequence"; }

std::string direction_name(MetricDirection direction) {
    return direction == MetricDirection::higher_is_better ? "higher_is_better" : "lower_is_better";
}

std::string aggregation_name(MetricAggregation aggregation) {
    return aggregation == MetricAggregation::arithmetic ? "arithmetic" : "square_mean_root";
}

// The registry is the Python app's, metric for metric and in its order:
// labels, formats, directions, averaging, bands and backends.
TEST(Metrics, RegistryIsThePythonAppsInItsOrder) {
    const nlohmann::json reference = test::load_reference("metrics.json")["metrics"];
    ASSERT_EQ(metrics().size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const nlohmann::json& expected = reference[i];
        const MetricDefinition& metric = metrics()[i];
        SCOPED_TRACE(expected["key"].get<std::string>());
        EXPECT_EQ(metric.key, expected["key"].get<std::string>());
        EXPECT_EQ(metric.label, expected["label"].get<std::string>());
        EXPECT_EQ(metric.short_label, expected["short_label"].get<std::string>());
        EXPECT_EQ(metric.table_header, expected["table_header"].get<std::string>());
        EXPECT_EQ(metric.axis_label, expected["axis_label"].get<std::string>());
        EXPECT_EQ("{:." + std::to_string(metric.decimals) + "f}", expected["value_format"].get<std::string>());
        EXPECT_EQ(metric.value_suffix, expected["value_suffix"].get<std::string>());
        EXPECT_EQ(kind_name(metric.kind), expected["kind"].get<std::string>());
        EXPECT_EQ(direction_name(metric.direction), expected["direction"].get<std::string>());
        EXPECT_EQ(aggregation_name(metric.aggregation), expected["aggregation"].get<std::string>());
        EXPECT_EQ(metric.fixed_y_max, test::reference_number(expected["fixed_y_max"]));
        ASSERT_EQ(metric.thresholds.size(), expected["thresholds"].size());
        for (std::size_t t = 0; t < metric.thresholds.size(); ++t) {
            EXPECT_EQ(std::string(1, metric.thresholds[t].comparison), expected["thresholds"][t][0].get<std::string>());
            EXPECT_EQ(metric.thresholds[t].value, *test::reference_number(expected["thresholds"][t][1]));
        }
        const auto text_or_empty = [](const nlohmann::json& text) {
            return text.is_null() ? std::string() : text.get<std::string>();
        };
        EXPECT_EQ(metric.ffmpeg_binding.has_value(),
                  !expected["bool_option"].is_null() || !expected["libvmaf_feature"].is_null());
        if (metric.ffmpeg_binding) {
            EXPECT_EQ(metric.ffmpeg_binding->bool_option, text_or_empty(expected["bool_option"]));
            EXPECT_EQ(metric.ffmpeg_binding->libvmaf_feature, text_or_empty(expected["libvmaf_feature"]));
        }
        EXPECT_EQ(metric.backend_id, text_or_empty(expected["backend_id"]));
    }
}

// Scores as shown: the metric's precision, "∞" and "−∞", and "—" for none.
TEST(Metrics, FormatValueShowsScoresAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("metrics.json");
    for (const nlohmann::json& expected : reference["metrics"]) {
        const MetricDefinition& metric = metric_definition(expected["key"].get<std::string>());
        for (const nlohmann::json& item : expected["format_value"]) {
            EXPECT_EQ(metric.format_value(test::reference_number(item["value"])), item["text"].get<std::string>())
                << metric.key << " of " << item["value"];
        }
    }
}

// The videos' best and worst scores, as the Python app marks them for
// Highlight best/worst results: by the metric's direction, compared as
// shown, none with fewer than two scores or all showing the same.
TEST(Metrics, BestAndWorstAreThePythonApps) {
    const nlohmann::json reference = test::load_reference("metrics.json");
    for (const nlohmann::json& expected : reference["metrics"]) {
        const MetricDefinition& metric = metric_definition(expected["key"].get<std::string>());
        for (const nlohmann::json& item : expected["best_and_worst"]) {
            std::vector<std::optional<double>> scores;
            for (const nlohmann::json& score : item["scores"]) {
                scores.push_back(test::reference_number(score));
            }
            std::vector<std::string> marks;
            for (const ScoreMark mark : metric.best_and_worst(scores)) {
                marks.emplace_back(mark == ScoreMark::best ? "best" : mark == ScoreMark::worst ? "worst" : "");
            }
            std::vector<std::string> wanted;
            for (const nlohmann::json& mark : item["marks"]) {
                wanted.push_back(mark.is_null() ? "" : mark.get<std::string>());
            }
            EXPECT_EQ(marks, wanted) << metric.key << " of " << item["scores"];
        }
    }
}

TEST(Metrics, UnknownKeyIsNotFound) {
    EXPECT_EQ(find_metric("vmaf_v2"), nullptr);
    EXPECT_THROW(metric_definition("vmaf_v2"), std::out_of_range);
}

}  // namespace
}  // namespace vml
