// CVVDP's displays and scores, held to the Python app's (cvvdp.json, from
// cvvdp.py and perceptual_vship.py): the presets, built-in and saved (a
// malformed one skipped), each display's identity and JSON for Vship, a
// request's settings read back, the size CVVDP is calculated at, and the
// whole video's JOD pooled from its seconds.
#include "vml/cvvdp.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "reference.h"
#include "vml/python_json.h"

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
    EXPECT_EQ(preset.description, expected["description"].get<std::string>()) << preset.name;
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

// The user's presets saved and deleted as the Python app saves and deletes
// them: the list it keeps in Settings, written as it writes it (key for key,
// a whole resolution as a whole number), or why it refuses.
TEST(Cvvdp, SavesAndDeletesTheUsersPresetsAsThePythonApp) {
    const Json& cases = cvvdp_reference()["user_presets"];
    ASSERT_FALSE(cases.empty());
    const auto written = [](const std::string& presets) {
        return python_json_dumps(nlohmann::ordered_json::parse(presets), 2);
    };
    for (const Json& item : cases) {
        const std::string saved = item["saved"].get<std::string>();
        if (item.contains("deleted")) {
            EXPECT_EQ(written(without_user_preset(saved, item["deleted"].get<std::string>())),
                      item["text"].get<std::string>())
                << item["deleted"] << " from " << saved;
            continue;
        }
        const Json& values = item["display"];
        CvvdpDisplay display;
        display.width = values["width"].get<double>();
        display.height = values["height"].get<double>();
        display.diagonal_inches = values["diagonal_inches"].get<double>();
        display.viewing_distance_m = values["viewing_distance_m"].get<double>();
        display.peak_luminance = values["peak_luminance"].get<double>();
        display.contrast = values["contrast"].get<double>();
        display.ambient_lux = values["ambient_lux"].get<double>();
        display.reflectivity = values["reflectivity"].get<double>();
        display.exposure = values["exposure"].get<double>();
        display.hdr = values["hdr"].get<bool>();
        const std::string name = item["name"].get<std::string>();
        try {
            const std::string presets = with_user_preset(saved, name, display);
            ASSERT_TRUE(item["error"].is_null()) << name << ": " << item["error"];
            EXPECT_EQ(written(presets), item["text"].get<std::string>()) << name << " in " << saved;
        } catch (const std::invalid_argument& error) {
            ASSERT_FALSE(item["error"].is_null()) << name << ": " << error.what();
            EXPECT_EQ(error.what(), item["error"].get<std::string>()) << name;
        }
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
