// A VVC frame's film grain message turned into the film grain library's
// parameters (film_grain.h), held to the Python app's own answers
// (reference/film_grain.json, from d3d11_film_grain.py's plane_sizes and
// native_params): the film grain test's messages, and messages that reach
// each of the conversion's rules, on frames of each chroma subsampling. And
// the grain those parameters give on the CPU (native/h274/grain_synthesis.h),
// held to the GPU drawer's (d3d11_film_grain.dll).
#include "vml/film_grain.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "grain_synthesis.h"
#include "reference.h"
#include "vml/sha256.h"

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

// The test's own random numbers, as the cases' maker (a script driving
// d3d11_film_grain.dll's vmaf_grain_planes) made them: a 32-bit LCG.
class Lcg {
public:
    explicit Lcg(std::uint32_t seed) : state_(seed) {}
    int next(int bound) {
        state_ = state_ * 1664525U + 1013904223U;
        return static_cast<int>((state_ >> 8) % static_cast<std::uint32_t>(bound));
    }

private:
    std::uint32_t state_;
};

// One case: a frame's grain parameters, its planes' codes, and the SHA-256
// of each plane the GPU drawer (d3d11_film_grain.dll on Windows' WARP
// renderer, which an RTX 5090 matched) gave for them.
struct GrainCase {
    int width, height, shift_x, shift_y, planes, depth, log2_scale_factor, poc;
    std::vector<std::string> digests;
};

// The parameters and the planes' codes of case `number`, made as the
// script made them: intervals that tile the 8-bit codes, some left out,
// and planes holding a gradient with noise, so that blocks fall in many
// intervals.
std::pair<GrainParams, std::vector<std::vector<int>>> make_case(const GrainCase& c, int number) {
    Lcg random(static_cast<std::uint32_t>(1000 + number));
    GrainParams params{};
    params.depth = c.depth;
    params.log2_scale_factor = c.log2_scale_factor;
    params.poc = c.poc;
    std::vector<std::vector<int>> planes;
    for (int index = 0; index < c.planes; ++index) {
        GrainPlane& plane = params.plane[index];
        plane.width = index == 0 ? c.width : (c.width + (1 << c.shift_x) - 1) >> c.shift_x;
        plane.height = index == 0 ? c.height : (c.height + (1 << c.shift_y) - 1) >> c.shift_y;
        std::vector<std::pair<int, int>> bounds;
        for (int lower = 0; lower < 256;) {
            const int upper = std::min(255, lower + 8 + random.next(60));
            if (random.next(5) != 0) {
                bounds.emplace_back(lower, upper);
            }
            lower = upper + 1;
        }
        plane.intervals = static_cast<int>(bounds.size());
        for (std::size_t i = 0; i < bounds.size(); ++i) {
            plane.lower[i] = static_cast<std::uint8_t>(bounds[i].first);
            plane.upper[i] = static_cast<std::uint8_t>(bounds[i].second);
            plane.scale[i] = static_cast<std::int16_t>(random.next(256) - (index != 0 ? 40 : 0));
            plane.h[i] = static_cast<std::uint8_t>(random.next(13));
            plane.v[i] = static_cast<std::uint8_t>(random.next(13));
        }
        const int top = (1 << c.depth) - 1;
        std::vector<int> codes;
        for (int y = 0; y < plane.height; ++y) {
            for (int x = 0; x < plane.width; ++x) {
                const int base = x * top / std::max(1, plane.width - 1) / 2 + y * top / std::max(1, plane.height - 1) / 2;
                const int noise = random.next(1 << (c.depth - 4)) - (1 << (c.depth - 5));
                codes.push_back(std::clamp(base + noise, 0, top));
            }
        }
        planes.push_back(std::move(codes));
    }
    return {params, planes};
}

// The pattern database, built once for every case.
const std::vector<std::int8_t>& pattern_database() {
    static const std::vector<std::int8_t> database = h274::build_pattern_database();
    return database;
}

// Draws plane `index` of `params` on `codes` with the CPU synthesis, its
// rows of blocks split at `split`, and gives the drawn samples' bytes.
template <typename Sample>
std::vector<std::uint8_t> draw_plane(const GrainParams& params, int index, const std::vector<int>& codes, int split) {
    const GrainPlane& plane = params.plane[index];
    std::vector<Sample> in;
    for (const int code : codes) {
        in.push_back(static_cast<Sample>(code));
    }
    std::vector<Sample> out(in.size());
    std::vector<std::uint32_t> offsets;
    h274::block_offsets(offsets, params.poc, index, (plane.width + 15) / 16, (plane.height + 15) / 16);
    const h274::PlaneDrawer<Sample> drawer(in.data(), plane.width, plane, params.depth, params.log2_scale_factor,
                                           offsets, pattern_database().data());
    const int rows = drawer.block_rows();
    drawer.draw_rows(out.data(), plane.width, std::min(split, rows), rows);  // in any order
    drawer.draw_rows(out.data(), plane.width, 0, std::min(split, rows));
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(out.data());
    return {bytes, bytes + out.size() * sizeof(Sample)};
}

// The CPU synthesis draws the GPU drawer's grain, sample for sample: on
// planes whose sizes are not multiples of 8, at a negative picture order
// count, in 4:2:0, 4:2:2, 4:4:4 and luma alone, at 8, 10 and 12 bits; its
// rows of blocks drawn in two pieces, the later first.
TEST(FilmGrain, TheCpuSynthesisDrawsTheGpusGrain) {
    const std::vector<GrainCase> cases{
        {64, 64, 1, 1, 3, 8, 3, 0,
         {"cceeba74b7bedea8d5afad6c68a6812263695e643c6b63e295eccb2d2ba345ed",
          "888d634c66d4f827655a28886eec08b38fa811a93ee0285bd49682bde48e20c7",
          "b67b90527cee4f3186898ce12e6763336928a1a5d894aec654953dc77e29fa04"}},
        {50, 38, 1, 1, 3, 8, 2, 7,
         {"78474f593c4b33ff62f0223eb7b66fb090e675b429d0be7bc8cc3d1182fcc9e0",
          "219b91b87181bd195dadbbf2d48b3130953eab9d7cc7e15cb17b708e676b32eb",
          "c054a9aaa814c01ce54c4659b75f54fd23e31653b1a63570bf334e74af51e54b"}},
        {72, 40, 1, 1, 3, 10, 2, 300,
         {"f38e6ed5a492d41c94ab2fab47803486333429f3e20dcf099497aaea346609d2",
          "7e7078916be9500485c658358cfbb596e7da1aee180ff6502226890d62c3d0a0",
          "dc5e44ebdb408bf19147a7ef0be79d1a3eda7e05e1f7d5eda2bf38ab441f1575"}},
        {37, 29, 1, 1, 3, 10, 4, -3,
         {"8de2cccddba661dca1fef00f0b931a435fe485cd6a6bdc109fded8301c3fb965",
          "ba03a6d2a736ea33a3948115e2474339c883f7f5db24a01d70ba395c94ce42cb",
          "32c93b54a0e7eea8c0302cee46e92370417934e09806eef3dbaafc931056a3ea"}},
        {48, 24, 0, 0, 3, 10, 3, 85,
         {"b7d18d7f46598cf422739e3a27aa258cfa4286f86fda7c25b55e6316b5f298b3",
          "dc8e82cd544ea841767d7e574dd550b4edfa90861dd6d68e9b9610bae3524904",
          "56f99a74d8982b2b6de414456c824794ff680ea3f4e5916d97227d97967ef1b2"}},
        {40, 32, 1, 0, 3, 8, 2, 12,
         {"1979bb0b972c7116fcba76b57f413bfbfe82dfa22ab7864e7382067d8d3a7845",
          "57f8cbf079e0ed7b17f8e109a842de393652bb81234c2afc862b03bb9d2fd3c6",
          "5cbcc56a2765e8c53a1b4afbcfed03f5d679433a875993483448d51c1ae62c22"}},
        {56, 40, 1, 1, 1, 10, 5, 1, {"f9c23dcafce8e182bbe8b922aa267e339d3b9e46fcb4076e0bbfe38541d71b44"}},
        {33, 17, 1, 1, 3, 12, 2, 255,
         {"5a5b6e5292c39d17dc5ed6a3ab938d7c574a2ca4a7faf237c72dcde26fa85c6d",
          "d20f3c9da7149cd4b9e4b353542554b7b1982221fee92df01f66812e1933a0fd",
          "31846f3c30d362c45e91d9b46a085ca1f1bf9e3bf39130d04da59e9c79778478"}},
    };
    for (std::size_t number = 0; number < cases.size(); ++number) {
        const GrainCase& c = cases[number];
        const auto [params, planes] = make_case(c, static_cast<int>(number));
        for (int index = 0; index < c.planes; ++index) {
            const std::vector<std::uint8_t> drawn =
                c.depth > 8 ? draw_plane<std::uint16_t>(params, index, planes[static_cast<std::size_t>(index)], 2)
                            : draw_plane<std::uint8_t>(params, index, planes[static_cast<std::size_t>(index)], 2);
            Sha256 digest;
            digest.update(drawn.data(), drawn.size());
            EXPECT_EQ(digest.hexdigest(), c.digests[static_cast<std::size_t>(index)])
                << "case " << number << ", plane " << index;
        }
    }
}

}  // namespace
}  // namespace vml
