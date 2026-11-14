// Export results (results_export.h): CVVDP's further displays are listed in
// the entry of the video they were scored for, not as videos of their own.
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "results_export.h"
#include "vml/cvvdp.h"

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
    const nlohmann::ordered_json document = export_document(std::nullopt, {run});
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
    EXPECT_FALSE(export_document(std::nullopt, {plain}).at("videos").at(0).contains("cvvdp_displays"));
}

}  // namespace
}  // namespace vml::app
