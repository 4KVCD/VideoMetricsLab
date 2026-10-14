// Video Compare's native playback decisions held to the Python app's
// (reference/native.json, from its gstreamer_playback.py,
// locked_presentation.py and d3d11_tonemap.py): which comparisons play
// natively and why not, what the presenter's shader does to each video's
// colours and with which parameters, and the YUV-to-RGB rows.
#include "vml/native_playback.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <string>

#include "reference.h"

namespace vml {
namespace {

using Json = nlohmann::json;

VideoInfo video_from(const Json& data) {
    VideoInfo info;
    info.path = data.at("name").get<std::string>();
    info.width = 3840;
    info.height = 2160;
    info.pix_fmt = data.at("pix_fmt");
    info.color_primaries = data.at("color_primaries");
    info.color_space = data.at("color_space");
    info.color_transfer = data.at("color_transfer");
    return info;
}

PreviewColorSettings colours_from(const Json& data) {
    PreviewColorSettings colours;
    colours.mode = preview_color_mode_from(data.at("mode").get<std::string>());
    if (!data.at("hdr").is_null()) {
        colours.display_hdr_enabled = data.at("hdr").get<bool>();
    }
    if (!data.at("white").is_null()) {
        colours.display_sdr_white_nits = data.at("white").get<double>();
    }
    return colours;
}

std::string shading_name(NativeShading shading) {
    return shading == NativeShading::sdr ? "sdr" : shading == NativeShading::hdr ? "hdr" : "none";
}

TEST(NativePlayback, DecidesAndShadesAsThePythonAppDoes) {
    const Json reference = test::load_reference("native.json");

    const Json& shadings = reference.at("shadings");
    ASSERT_FALSE(shadings.empty());
    for (const Json& item : shadings) {
        const std::string wanted = item.at("shading").is_null() ? "none" : item.at("shading").get<std::string>();
        EXPECT_EQ(shading_name(native_shading_mode(video_from(item.at("video")), colours_from(item.at("colours")))),
                  wanted)
            << item.dump();
    }

    const Json& problems = reference.at("problems");
    ASSERT_FALSE(problems.empty());
    for (const Json& item : problems) {
        FrameComparison comparison;
        comparison.source_info = video_from(item.at("source"));
        comparison.distorted_info = video_from(item.at("distorted"));
        const std::optional<std::string> problem = native_playback_problem(comparison, colours_from(item.at("colours")));
        EXPECT_EQ(problem.value_or("none"), item.at("problem").is_null() ? "none" : item.at("problem").get<std::string>())
            << item.dump();
    }

    // The shader's parameters, as C floats; numpy's matrix inverse leaves
    // residues near 1e-17 where an exact inverse gives 0.
    const Json& parameters = reference.at("parameters");
    ASSERT_FALSE(parameters.empty());
    for (const Json& item : parameters) {
        const std::array<float, 16> made =
            shading_parameters(item.at("kind").get<std::string>(), item.at("primaries").get<std::string>(),
                               item.at("hdr").get<bool>());
        const Json& wanted = item.at("parameters");
        ASSERT_EQ(wanted.size(), made.size());
        for (std::size_t index = 0; index < made.size(); ++index) {
            const double expected = *test::reference_number(wanted[index]);
            EXPECT_NEAR(made[index], expected, 1e-6 * std::max(1.0, std::abs(expected)))
                << item.at("kind") << " " << item.at("primaries") << " hdr " << item.at("hdr") << " [" << index << "]";
        }
    }

    // The rows exactly: worked out in the same order, they round alike.
    const Json& rows = reference.at("rows");
    ASSERT_FALSE(rows.empty());
    for (const Json& item : rows) {
        const std::array<float, 12> made = yuv_to_rgb(item.at("matrix").get<std::string>(),
                                                      item.at("full_range").get<bool>(), item.at("ten_bit").get<bool>());
        const Json& wanted = item.at("rows");
        ASSERT_EQ(wanted.size(), made.size());
        for (std::size_t index = 0; index < made.size(); ++index) {
            EXPECT_EQ(static_cast<double>(made[index]), *test::reference_number(wanted[index]))
                << item.dump() << " [" << index << "]";
        }
    }
}

}  // namespace
}  // namespace vml
