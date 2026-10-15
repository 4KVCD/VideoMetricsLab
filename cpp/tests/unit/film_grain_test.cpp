// A VVC frame's film grain message turned into the film grain library's
// parameters (film_grain.h), held to the Python app's own answers
// (reference/film_grain.json, from d3d11_film_grain.py's plane_sizes and
// native_params): the film grain test's messages, and messages that reach
// each of the conversion's rules, on frames of each chroma subsampling.
#include "vml/film_grain.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "reference.h"

namespace vml {
namespace {

TEST(FilmGrain, ParametersAreThePythonAppsForEachMessageAndFrame) {
    const nlohmann::json reference = test::load_reference("film_grain.json");
    const nlohmann::json& cases = reference.at("cases");
    ASSERT_FALSE(cases.empty());
    for (const nlohmann::json& item : cases) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        FilmGrainMessage message;
        message.model_id = item.at("model_id").get<int>();
        message.log2_scale_factor = item.at("log2_scale_factor").get<int>();
        const nlohmann::json& components = item.at("components");
        for (std::size_t component = 0; component < components.size(); ++component) {
            for (const nlohmann::json& interval : components[component]) {
                message.components[component].push_back({interval.at("lower").get<int>(),
                                                         interval.at("upper").get<int>(),
                                                         interval.at("values").get<std::vector<int>>()});
            }
        }
        const std::pair<int, int> chroma_shift{item.at("chroma_shift")[0].get<int>(),
                                               item.at("chroma_shift")[1].get<int>()};
        const std::vector<std::pair<int, int>> sizes = grain_plane_sizes(
            item.at("width").get<int>(), item.at("height").get<int>(), chroma_shift, item.at("planes").get<int>());
        ASSERT_EQ(sizes.size(), item.at("sizes").size());
        for (std::size_t plane = 0; plane < sizes.size(); ++plane) {
            EXPECT_EQ(sizes[plane].first, item.at("sizes")[plane][0].get<int>());
            EXPECT_EQ(sizes[plane].second, item.at("sizes")[plane][1].get<int>());
        }

        const std::optional<GrainParams> params =
            grain_params(message, item.at("seed").get<int>(), item.at("depth").get<int>(), sizes, chroma_shift);
        const nlohmann::json& expected = item.at("expected");
        ASSERT_EQ(params.has_value(), !expected.is_null());
        if (!params) {
            continue;
        }
        EXPECT_EQ(params->depth, expected.at("depth").get<int>());
        EXPECT_EQ(params->log2_scale_factor, expected.at("log2_scale_factor").get<int>());
        EXPECT_EQ(params->poc, expected.at("poc").get<int>());
        for (std::size_t index = 0; index < 3; ++index) {
            const GrainPlane& plane = params->plane[index];
            const nlohmann::json& wanted = expected.at("planes")[index];
            EXPECT_EQ(plane.width, wanted.at("width").get<int>()) << "plane " << index;
            EXPECT_EQ(plane.height, wanted.at("height").get<int>()) << "plane " << index;
            ASSERT_EQ(plane.intervals, wanted.at("intervals").get<int>()) << "plane " << index;
            for (int interval = 0; interval < plane.intervals; ++interval) {
                const auto at = static_cast<std::size_t>(interval);
                EXPECT_EQ(plane.lower[at], wanted.at("lower")[at].get<int>()) << "plane " << index << ", " << at;
                EXPECT_EQ(plane.upper[at], wanted.at("upper")[at].get<int>()) << "plane " << index << ", " << at;
                EXPECT_EQ(plane.scale[at], wanted.at("scale")[at].get<int>()) << "plane " << index << ", " << at;
                EXPECT_EQ(plane.h[at], wanted.at("h")[at].get<int>()) << "plane " << index << ", " << at;
                EXPECT_EQ(plane.v[at], wanted.at("v")[at].get<int>()) << "plane " << index << ", " << at;
            }
        }
    }
}

}  // namespace
}  // namespace vml
