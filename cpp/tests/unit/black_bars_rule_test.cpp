// HandBrake's black-bar rule (black_bars_rule.h): the bars of synthetic
// pictures, the box chosen from several, and the Y4M stream FFmpeg gives the
// pictures in. The cases are the ones the rule was chosen on (GitHub issue
// #9's packed alpha matte; bars lifted above black).
#include "vml/black_bars_rule.h"

#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vml {
namespace {

// A 192x108 picture: picture content (a ramp from 60 to 187) with `bars`
// rows of `bar` luma at the top and bottom and `pillars` columns at the
// sides.
LumaPicture picture(int bars, int pillars = 0, std::uint8_t bar = 16) {
    LumaPicture p{192, 108, std::vector<std::uint8_t>(192 * 108)};
    for (int y = 0; y < p.height; ++y) {
        for (int x = 0; x < p.width; ++x) {
            const bool in_bar = y < bars || y >= p.height - bars || x < pillars || x >= p.width - pillars;
            p.luma[y * p.width + x] = in_bar ? bar : static_cast<std::uint8_t>(60 + (x * 128) / p.width);
        }
    }
    return p;
}

TEST(BlackBarsRule, FindsEachPicturesBars) {
    // Letterbox and pillarbox
    EXPECT_EQ(picture_edges(picture(14)), (BarEdges{14, 14, 0, 0}));
    EXPECT_EQ(picture_edges(picture(0, 24)), (BarEdges{0, 0, 24, 24}));
    // Bars lifted above black, with noise (26 +- 5): still bars, where an
    // average under cropdetect's 24 is not
    LumaPicture lifted = picture(14, 0, 26);
    for (std::size_t i = 0; i < lifted.luma.size(); ++i) {
        if (lifted.luma[i] == 26) {
            lifted.luma[i] = static_cast<std::uint8_t>(21 + (i * 7) % 11);
        }
    }
    EXPECT_EQ(picture_edges(lifted), (BarEdges{14, 14, 0, 0}));
    // Issue #9: the bottom bar holds a packed alpha matte, a few bright
    // samples in black (row means about 20): picture, not bar
    LumaPicture matte = picture(14);
    for (int y = 108 - 14; y < 108; ++y) {
        for (int x = (y * 7) % 50; x < 192; x += 50) {
            matte.luma[y * 192 + x] = 235;
        }
    }
    EXPECT_EQ(picture_edges(matte), (BarEdges{14, 0, 0, 0}));
    // A dim line at the very top (Bad Boys 2's first row): within the first
    // hundredth of the height, with picture right below it, so no bar
    LumaPicture dim_top = picture(0);
    std::memset(dim_top.luma.data(), 20, 192);
    EXPECT_EQ(picture_edges(dim_top), (BarEdges{0, 0, 0, 0}));
    // A dark picture (a quarter or more of an edge dark) says nothing
    EXPECT_EQ(picture_edges(picture(27)), std::nullopt);
    EXPECT_EQ(picture_edges(LumaPicture{192, 108, std::vector<std::uint8_t>(192 * 108, 16)}), std::nullopt);
}

TEST(BlackBarsRule, TakesTheMedianOfThePicturesBars) {
    const BarEdges letterbox{138, 138, 0, 0};
    // The median of each edge; odd lines rounded down to even
    EXPECT_EQ(handbrake_box({letterbox, letterbox, {131, 131, 0, 0}, letterbox, letterbox}, 10, 1920, 1080),
              (CropBox{1920, 804, 0, 138}));
    EXPECT_EQ(handbrake_box({{131, 131, 0, 0}, {131, 131, 0, 0}, {131, 131, 0, 0}}, 3, 1920, 1080),
              (CropBox{1920, 820, 0, 130}));
    // A video that changes shape (Everything Everywhere All at Once): four
    // pictures at the full frame, five letterboxed: no picture cut
    const BarEdges scope{233, 234, 0, 0};
    const BarEdges full{0, 0, 0, 0};
    EXPECT_EQ(handbrake_box({full, full, full, full, scope, scope, scope, scope, scope}, 10, 3840, 2076),
              (CropBox{3840, 2076, 0, 0}));
    // Three of nine at the full frame: the median's bars
    EXPECT_EQ(handbrake_box({full, full, full, scope, scope, scope, scope, scope, scope}, 10, 3840, 2076),
              (CropBox{3840, 2076 - 232 - 234, 0, 232}));
    // Fewer than three pictures said something: the whole frame
    EXPECT_EQ(handbrake_box({letterbox, letterbox}, 10, 1920, 1080), (CropBox{1920, 1080, 0, 0}));
    // ... but a video too short for three judged on the ones it has
    EXPECT_EQ(handbrake_box({letterbox}, 1, 1920, 1080), (CropBox{1920, 804, 0, 138}));
    EXPECT_EQ(handbrake_box({}, 0, 1920, 1080), (CropBox{1920, 1080, 0, 0}));
}

// A Y4M stream of two 4x2 pictures of 16-bit luma, as FFmpeg's yuv4mpegpipe
// writes it: each picture's samples are their high bytes.
TEST(BlackBarsRule, ReadsFfmpegsLumaStream) {
    std::string stream = "YUV4MPEG2 W4 H2 F24:1 Ip A1:1 Cmono16 XCOLORRANGE=FULL\n";
    for (int frame = 0; frame < 2; ++frame) {
        stream += "FRAME\n";
        for (int i = 0; i < 8; ++i) {
            const int value = (16 + frame * 100 + i) << 2;  // a 10-bit sample, as gray16le holds it
            stream.push_back(static_cast<char>((value << 6) & 0xff));
            stream.push_back(static_cast<char>((value << 6) >> 8));
        }
    }
    std::size_t at = 0;
    Y4mLumaReader reader([&](void* into, std::size_t bytes) {
        const std::size_t n = std::min(bytes, stream.size() - at);
        std::memcpy(into, stream.data() + at, n);
        at += n;
        return n;
    });
    const std::optional<LumaPicture> first = reader.next();
    ASSERT_TRUE(first);
    EXPECT_EQ(first->width, 4);
    EXPECT_EQ(first->height, 2);
    EXPECT_EQ(first->luma, (std::vector<std::uint8_t>{16, 17, 18, 19, 20, 21, 22, 23}));
    EXPECT_EQ(reader.next()->luma[0], 116);
    EXPECT_FALSE(reader.next());
    // A stream that is not 16-bit luma, or ends inside a picture, is refused
    stream = "YUV4MPEG2 W4 H2 F24:1 C420p10\n";
    at = 0;
    Y4mLumaReader colour([&](void* into, std::size_t bytes) {
        const std::size_t n = std::min(bytes, stream.size() - at);
        std::memcpy(into, stream.data() + at, n);
        at += n;
        return n;
    });
    EXPECT_THROW(colour.next(), std::runtime_error);
}

}  // namespace
}  // namespace vml
