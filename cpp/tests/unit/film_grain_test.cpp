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
         {"3407d7e4feb235bf7e4f16b5314b43658402966397250ffb9caae96dd1c8fd28",
          "2106190d324657e629eddcdf05493a54db42464d42ff62a016f478a60bce0a3d",
          "7d11ad7d242f1ff9bf32e6169ef18997944520288b29b02cae14e1dcc0b59e1d"}},
        {50, 38, 1, 1, 3, 8, 2, 7,
         {"5af42bf738665fe7f5e558695ca99b72a16399831f13ca09a207d3afc20eff93",
          "3eb63f0320ddadf776e10061e27cf4973b5af4804ec1cb04a8b43100770f7d83",
          "3e344333bc63cc984fc3f5c4a3ad82629efcb91df141fde53191f257e6ab520c"}},
        {72, 40, 1, 1, 3, 10, 2, 300,
         {"437a16d52a254c7e7325af50ad79388c8cd0fa4a85723f80b83000ff69d5d14f",
          "7f6cffd7db2eee0c0b1186fee1837e525cf04d59851cb9ab2da67ef6c62231d5",
          "b7e2fef0f118bc54a89c4f99914466c611a845c556859e854c287aa56dcf9b77"}},
        {37, 29, 1, 1, 3, 10, 4, -3,
         {"50151b4c70636fc96bb6523245fdf65875939456354c744c2e0b2c0dfac05f84",
          "535e50df75e722561628aa0ca4fe3a05f8b7fac12ef363ee306d80e75cf333df",
          "d1b6e04d6f9d6a3c573d62c2cc1634dc3c22a7046f403c994dc765eead5a0b10"}},
        {48, 24, 0, 0, 3, 10, 3, 85,
         {"ce79bc64fdecabc93a2692411735213c362cac9f43c0496b90e7d09990c16b4a",
          "e049f0dfc474b1a698320045ce74e78648b24bf12590a65c8cc565840196f240",
          "610bb8e2ac13ce8341cc950ea33e5621999ae9154faabcc145a65c5eb0914c25"}},
        {40, 32, 1, 0, 3, 8, 2, 12,
         {"0274e7b9916229e92f700bf12bfd7ba206a07d27ef58e4d4200f77b303c1e944",
          "99c57f2b414b6fbed96797b86c7c226ce270dab6ff09e2dd3eabc18f71efec91",
          "b0363a3918999d2148398ec188703f98fa51f0bdc957be003692c210470d4429"}},
        {56, 40, 1, 1, 1, 10, 5, 1, {"bc2aa2f507381efa97cc1f35c527c3bebecc9c11b1212083166876b3a121c97c"}},
        {33, 17, 1, 1, 3, 12, 2, 255,
         {"5fea77e38acf54c67a86e23e7664e75ee511134e2223cce630353e49daaef373",
          "b28958a7b35f4cc910d77f6d0d56f465f5fccc1b6ca0e38fc550c0aa11f60ef4",
          "fca3fefaa792d5491733af89ce70322e9bdeb01a7157ae2857e8150f6b5c3843"}},
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
