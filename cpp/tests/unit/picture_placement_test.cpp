// Where a test video's picture sits in the source's frame when the black-bar
// boxes alone cannot say (issues #18 and #20): the places the boxes leave,
// and the pictures that choose among them.
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "vml/crop_detect.h"
#include "vml/picture_placement.h"

namespace vml {
namespace {

VideoInfo frame(int width, int height) {
    VideoInfo info;
    info.width = width;
    info.height = height;
    return info;
}

// A picture of detail: every sample its own value (a fixed pseudo-random
// sequence), so a picture moved by a line is far from itself.
LumaPicture textured(int width, int height, std::uint32_t seed) {
    LumaPicture picture{width, height, std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height)};
    std::uint32_t state = seed;
    for (std::uint8_t& sample : picture.luma) {
        state = state * 1664525u + 1013904223u;
        sample = static_cast<std::uint8_t>(32 + (state >> 24) % 192);
    }
    return picture;
}

// `box` of `picture`, as a picture of its own.
LumaPicture cut(const LumaPicture& picture, const CropBox& box) {
    LumaPicture part{box.w, box.h, {}};
    for (int y = 0; y < box.h; ++y) {
        const auto* line = &picture.luma[static_cast<std::size_t>(box.y + y) * picture.width + box.x];
        part.luma.insert(part.luma.end(), line, line + box.w);
    }
    return part;
}

// `picture` halved on both axes, each sample the mean of four.
LumaPicture halved(const LumaPicture& picture) {
    LumaPicture half{picture.width / 2, picture.height / 2, {}};
    for (int y = 0; y < half.height; ++y) {
        for (int x = 0; x < half.width; ++x) {
            const auto at = [&picture](int column, int row) {
                return picture.luma[static_cast<std::size_t>(row) * picture.width + column];
            };
            const int sum = at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1);
            half.luma.push_back(static_cast<std::uint8_t>((sum + 2) / 4));
        }
    }
    return half;
}

// The places each kind of encode leaves: one where the boxes decide alone,
// several where only the pictures can.
TEST(PicturePlacement, TheBoxesLeaveEachPlaceTheEncodeCouldHaveBeenCutFrom) {
    const CropBox bars{1920, 800, 0, 140};
    // Cut to 1920x800 from a source whose picture ends in a dark row its detection cut (1920x798 between bars
    // above and below): the test's picture starts at row 138 or 140 of the source.
    EXPECT_EQ(unscaled_places(1920, 1080, 1920, 800, CropBox{1920, 798, 0, 140}, CropBox{1920, 800, 0, 0}),
              (std::vector<CropBox>{{1920, 800, 0, 138}, {1920, 800, 0, 140}}));
    // The dark column at the frame's edge: the kept edge can only be on the open side, one place.
    EXPECT_EQ(unscaled_places(1920, 1080, 1920, 800, CropBox{1918, 800, 0, 140}, CropBox{1920, 800, 0, 0}),
              (std::vector<CropBox>{{1920, 800, 0, 140}}));
    // Cut 16 rows into the picture: every even row from the box's top to 16 below it.
    const std::vector<CropBox> overcut = unscaled_places(1920, 1080, 1920, 784, bars, CropBox{1920, 784, 0, 0});
    ASSERT_EQ(overcut.size(), 9U);
    EXPECT_EQ(overcut.front(), (CropBox{1920, 784, 0, 140}));
    EXPECT_EQ(overcut.back(), (CropBox{1920, 784, 0, 156}));
    // Cut deeper than deepest_cut, or the same size as the source: not a cut this looks for.
    EXPECT_TRUE(unscaled_places(1920, 1080, 1920, 700, bars, CropBox{1920, 700, 0, 0}).empty());
    EXPECT_TRUE(unscaled_places(1920, 1080, 1920, 1080, bars, bars).empty());
    // Source only, a test frame 2 rows taller than the source's box: decided by the pictures.
    EXPECT_TRUE(placed_by_content(CropMode::source_only, frame(1920, 1080), frame(1920, 802), bars, std::nullopt));
    EXPECT_FALSE(placed_by_content(CropMode::source_only, frame(1920, 1080), frame(1920, 800), bars, std::nullopt));
}

// Scaled encodes: one cut to a box that kept a thin dark column, and a whole frame whose bars are thinner than a
// sample once scaled; the first place is what the boxes alone compare.
TEST(PicturePlacement, AScaledEncodeMayShowTheBoxGrownByAThinEdge) {
    EXPECT_EQ(scaled_places(1920, 1080, 960, 400, CropBox{1918, 800, 0, 140}),
              (std::vector<CropBox>{{1918, 800, 0, 140}, {1920, 800, 0, 140}}));
    EXPECT_EQ(scaled_places(1920, 1080, 720, 404, CropBox{1920, 1076, 0, 2}),
              (std::vector<CropBox>{{1920, 1080, 0, 0}, {1920, 1076, 0, 2}}));
    // Bars a scale cannot hide, and a box the test frame's shape does not fit: the boxes decide.
    EXPECT_TRUE(scaled_places(1920, 1080, 1280, 720, CropBox{1920, 800, 0, 140}).empty());
    EXPECT_TRUE(scaled_places(1920, 1080, 960, 400, CropBox{1920, 800, 0, 140}).empty());
}

// The pictures choose the place the test's picture was cut from, scaled or not, and say nothing where they
// cannot tell: a flat picture, or a test frame that is the source scaled rather than cut.
TEST(PicturePlacement, ThePicturesChooseThePlaceTheTestWasCutFrom) {
    const std::vector<CropBox> rows{{64, 32, 0, 6}, {64, 32, 0, 8}, {64, 32, 0, 10}};
    const CropBox whole{64, 32, 0, 0};
    std::vector<LumaPicture> source;
    std::vector<LumaPicture> test;
    for (std::uint32_t seed = 1; seed <= 3; ++seed) {
        source.push_back(textured(64, 48, seed));
        test.push_back(cut(source.back(), rows[1]));
    }
    EXPECT_EQ(place_shown(source, test, whole, rows), std::optional<std::size_t>(1));

    const std::vector<CropBox> grown{{62, 32, 0, 8}, {64, 32, 0, 8}};
    std::vector<LumaPicture> scaled;
    for (const LumaPicture& picture : source) {
        scaled.push_back(halved(cut(picture, grown[1])));
    }
    EXPECT_EQ(place_shown(source, scaled, CropBox{32, 16, 0, 0}, grown), std::optional<std::size_t>(1));

    const LumaPicture flat{64, 48, std::vector<std::uint8_t>(64 * 48, 90)};
    const LumaPicture flat_test{64, 32, std::vector<std::uint8_t>(64 * 32, 90)};
    EXPECT_EQ(place_shown({flat}, {flat_test}, whole, rows), std::nullopt);

    // 64x48 halved to 32x24 and read as a cut 8 rows into a 32x32 box: no place fits clearly.
    std::vector<LumaPicture> halves;
    for (const LumaPicture& picture : source) {
        halves.push_back(halved(picture));
    }
    const std::vector<CropBox> cuts{{32, 24, 0, 0}, {32, 24, 0, 2}, {32, 24, 0, 4}, {32, 24, 0, 6}, {32, 24, 0, 8}};
    EXPECT_EQ(place_shown(source, halves, CropBox{32, 24, 0, 0}, cuts), std::nullopt);
}

}  // namespace
}  // namespace vml
