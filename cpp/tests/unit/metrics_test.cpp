#include "vml/metrics.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "reference.h"
#include "vml/results.h"

namespace vml {
namespace {

std::string kind_name(MetricKind kind) { return kind == MetricKind::frame ? "frame" : "sequence"; }

std::string direction_name(MetricDirection direction) {
    return direction == MetricDirection::higher_is_better ? "higher_is_better" : "lower_is_better";
}

std::string aggregation_name(MetricAggregation aggregation) {
    return aggregation == MetricAggregation::arithmetic            ? "arithmetic"
           : aggregation == MetricAggregation::square_mean_root_db ? "square_mean_root"
                                                                   : "mean_error_db";
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

// Pictures too small for a metric are refused, naming each metric with the
// size it needs; the limit is on both sides, and a metric with none (PSNR,
// CVVDP) takes any size.
TEST(Metrics, TooSmallPicturesNameTheMetricsAndTheSizeEachNeeds) {
    const std::vector<std::string> all{"vmaf", "vmaf_neg", "psnr", "ssim", "ssimulacra2", "butteraugli", "cvvdp"};
    EXPECT_EQ(too_small_problem(all, 6, 6),
              "The compared pictures, 6x6, are too small for VMAF v0.6.1 and VMAF NEG (at least 32x32); SSIM "
              "(at least 11x11); SSIMULACRA2 and Butteraugli (at least 8x8).");
    EXPECT_EQ(too_small_problem(all, 64, 10),
              "The compared pictures, 64x10, are too small for VMAF v0.6.1 and VMAF NEG (at least 32x32); SSIM "
              "(at least 11x11).");
    EXPECT_EQ(too_small_problem(all, 32, 32), std::nullopt);
    EXPECT_EQ(too_small_problem({"psnr", "xpsnr", "cvvdp"}, 2, 2), std::nullopt);
    EXPECT_EQ(too_small_problem({"ssimulacra2"}, 8, 7),
              "The compared pictures, 8x7, are too small for SSIMULACRA2 (at least 8x8).");
}

// Add/remove metrics' and Settings' two sections come from the one list,
// metrics_hidden_at_first, in the registry's order; a plane is in its
// metric's and listed in neither.
// The order the user set comes first, each plane after its metric; unknown
// keys, planes and repeats in it are passed over, and the metrics it leaves
// out follow in the registry's order. No order is the registry's.
TEST(Metrics, OrderedMetricsPutTheUsersOrderFirst) {
    const auto keys_of = [](const std::vector<const MetricDefinition*>& ordered) {
        std::vector<std::string> keys;
        for (const MetricDefinition* metric : ordered) {
            keys.emplace_back(metric->key);
        }
        return keys;
    };
    const std::vector<std::string> keys = keys_of(ordered_metrics({"cvvdp", "nope", "psnr_yuv_u", "psnr_yuv", "cvvdp"}));
    ASSERT_EQ(keys.size(), metrics().size());
    EXPECT_EQ(std::vector<std::string>(keys.begin(), keys.begin() + 7),
              (std::vector<std::string>{"cvvdp", "psnr_yuv", "psnr_yuv_y", "psnr_yuv_u", "psnr_yuv_v", "vmaf", "vmaf_neg"}));
    std::vector<std::string> registry;
    for (const MetricDefinition& metric : metrics()) {
        registry.emplace_back(metric.key);
    }
    EXPECT_EQ(keys_of(ordered_metrics({})), registry);
}

// A run saves the planes of the YUV metrics it asks for, which are
// calculated with them (with_planes), and the views drop the planes not
// shown (drop_planes_not_in), keeping every other metric.
TEST(Metrics, PlanesAreSavedWithTheirMetricAndDroppedWhereNotShown) {
    EXPECT_EQ(with_planes({"vmaf", "psnr_yuv", "psnr_yuv_u"}),
              (std::vector<std::string>{"vmaf", "psnr_yuv", "psnr_yuv_u", "psnr_yuv_y", "psnr_yuv_v"}));
    EXPECT_EQ(with_planes({"psnr", "ssim"}), (std::vector<std::string>{"psnr", "ssim"}));

    MetricResultSet scores;
    for (const char* key : {"psnr_yuv", "psnr_yuv_y", "psnr_yuv_u", "vmaf", "vmaf_yuv_v"}) {
        scores.add(FrameMetricResult{key, {0}, {0.0}, {1.0F}, {}});
    }
    drop_planes_not_in(scores, {"psnr_yuv_u"});
    EXPECT_EQ(scores.keys(), (std::vector<std::string>{"psnr_yuv", "psnr_yuv_u", "vmaf"}));
}

TEST(Metrics, SectionsComeFromTheListHiddenAtFirst) {
    using Keys = std::vector<std::string_view>;
    EXPECT_EQ(metrics_in(MetricSection::general_purpose),
              (Keys{"vmaf", "vmaf_neg", "vmaf_v1", "vmaf_hdr", "psnr", "ssim", "xpsnr", "ssimulacra2", "butteraugli",
                    "cvvdp"}));
    EXPECT_EQ(metrics_in(MetricSection::specialized), Keys(metrics_hidden_at_first.begin(), metrics_hidden_at_first.end()));
    EXPECT_EQ(section_of("psnr_yuv_u"), MetricSection::specialized);
    EXPECT_EQ(metric_guide("ssim_yuv_v").measures, metric_guide("ssim_yuv").measures);
}

// docs/ARCHITECTURE.md's "Adding a metric", for every metric in the
// registry: its few words and tooltip text; in exactly one section, the one
// metrics_hidden_at_first gives it (each of whose keys is a metric, not a
// plane); ranges exactly where a source is named (ranges_source), and ranges
// wherever it has bands. A plane takes all of these from its metric.
TEST(Metrics, EveryMetricFollowsTheRulesForAddingOne) {
    const std::vector<std::string_view> general = metrics_in(MetricSection::general_purpose);
    const std::vector<std::string_view> specialized = metrics_in(MetricSection::specialized);
    const auto in = [](const std::vector<std::string_view>& keys, std::string_view key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    };
    for (const std::string_view key : metrics_hidden_at_first) {
        ASSERT_NE(find_metric(key), nullptr) << key;
        EXPECT_FALSE(find_metric(key)->is_plane()) << key;
    }
    for (const MetricDefinition& metric : metrics()) {
        const std::string_view key = metric.key;
        if (metric.is_plane()) {
            EXPECT_FALSE(in(general, key) || in(specialized, key)) << key;
            EXPECT_EQ(section_of(key), section_of(metric.plane_of)) << key;
            continue;
        }
        const MetricGuide guide = metric_guide(key);
        EXPECT_FALSE(guide.brief.empty()) << key << ": no few words beside its name";
        EXPECT_FALSE(guide.measures.empty()) << key << ": no tooltip text";
        EXPECT_NE(in(general, key), in(specialized, key)) << key << ": not in exactly one section";
        EXPECT_EQ(in(specialized, key), in({metrics_hidden_at_first.begin(), metrics_hidden_at_first.end()}, key))
            << key << ": its section is not metrics_hidden_at_first's";
        EXPECT_EQ(guide.ranges.empty(), guide.ranges_source.empty())
            << key << ": ranges without a source, or a source without ranges";
        if (!metric.thresholds.empty()) {
            EXPECT_FALSE(guide.ranges.empty()) << key << ": bands, but no ranges in its tooltip";
        }
    }
}

}  // namespace
}  // namespace vml
