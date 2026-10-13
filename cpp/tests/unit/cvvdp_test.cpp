// CVVDP's displays and scores, held to the Python app's (cvvdp.json, from
// cvvdp.py and perceptual_vship.py): the presets, built-in and saved (a
// malformed one skipped), each display's identity and JSON for Vship, a
// request's settings read back, the size CVVDP is calculated at, and the
// whole video's JOD pooled from its seconds.
#include "vml/cvvdp.h"

#include <gtest/gtest.h>

#include <string>

#include "reference.h"

namespace vml {
namespace {

using Json = nlohmann::json;

const Json& cvvdp_reference() {
    static const Json reference = test::load_reference("cvvdp.json");
    return reference;
}

// A preset as the reference describes it.
void expect_preset(const CvvdpPreset& preset, const Json& expected) {
    EXPECT_EQ(preset.name, expected["name"].get<std::string>());
    EXPECT_EQ(preset.builtin, expected["builtin"].get<bool>()) << preset.name;
    EXPECT_EQ(Json(preset.settings.display.identity()), expected["identity"]) << preset.name;
    EXPECT_EQ(preset.settings.resize_to_display, expected["resize_to_display"].get<bool>()) << preset.name;
    EXPECT_EQ(vship_display_json(preset.settings.display), expected["vship_json"].get<std::string>()) << preset.name;
}

TEST(Cvvdp, DescribesDisplaysAsThePythonApp) {
    const Json& lists = cvvdp_reference()["preset_lists"];
    ASSERT_FALSE(lists.empty());
    for (const Json& list : lists) {
        const std::string saved = list["saved"].get<std::string>();
        const std::vector<CvvdpPreset> presets = cvvdp_presets(saved);
        ASSERT_EQ(presets.size(), list["presets"].size()) << saved;
        for (std::size_t i = 0; i < presets.size(); ++i) {
            expect_preset(presets[i], list["presets"][i]);
        }
        for (const auto& [name, expected] : list["named"].items()) {
            const std::optional<CvvdpPreset> preset = cvvdp_preset_named(name, saved);
            ASSERT_EQ(preset.has_value(), !expected.is_null()) << name << " in " << saved;
            if (preset) {
                expect_preset(*preset, expected);
            }
        }
    }
    // What a request says is scored: the identity's rounded values.
    for (const Json& item : cvvdp_reference()["read_back"]) {
        const CvvdpSettings settings = CvvdpSettings::from_parameters(item["parameters"]);
        EXPECT_EQ(Json(settings.display.identity()), item["identity"]);
        EXPECT_EQ(settings.resize_to_display, item["resize_to_display"].get<bool>());
        EXPECT_EQ(vship_display_json(settings.display), item["vship_json"].get<std::string>());
    }
}

TEST(Cvvdp, SizesAndPoolsAsThePythonApp) {
    for (const Json& item : cvvdp_reference()["sizes"]) {
        CvvdpSettings settings;
        settings.display.width = item["display"][0].get<double>();
        settings.display.height = item["display"][1].get<double>();
        settings.resize_to_display = item["resize_to_display"].get<bool>();
        const std::pair<int, int> size{item["size"][0].get<int>(), item["size"][1].get<int>()};
        const std::pair<int, int> computed = cvvdp_computed_size(size, settings);
        EXPECT_EQ(computed.first, item["computed"][0].get<int>()) << item;
        EXPECT_EQ(computed.second, item["computed"][1].get<int>()) << item;
    }
    const Json& pools = cvvdp_reference()["pools"];
    ASSERT_FALSE(pools.empty());
    for (const Json& item : pools) {
        std::vector<CvvdpWindow> windows;
        for (const Json& window : item["windows"]) {
            windows.push_back({window[0].get<long long>(), window[1].get<long long>(),
                               test::reference_number(window[2]).value()});
        }
        EXPECT_TRUE(test::same_number(test::reference_number(item["jod"]).value(), pool_cvvdp_windows(windows)))
            << item;
    }
}

}  // namespace
}  // namespace vml
