// HDR VMAF's reading of a pair's colours and the sizes it converts pictures
// to: libvmaf's own choices (conversion_policy.c), so that it takes the
// pictures HdrScorer converts as they are.
#include "vml/vmaf_hdr.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <variant>

namespace vml {
namespace {

VideoInfo video(const std::string& transfer, const std::string& matrix = "", const std::string& primaries = "") {
    VideoInfo info;
    info.color_range = "tv";
    info.color_transfer = transfer;
    info.color_space = matrix;
    info.color_primaries = primaries;
    return info;
}

// The 4K models' target: ICtCp, limited range, 4:2:0, 16 bits, 2160 lines.
HdrModelTarget model_at(unsigned display) {
    HdrModelTarget model;
    model.colour = {VMAF_COLOR_RANGE_LIMITED, VMAF_COLOR_PRIMARIES_BT2020, VMAF_COLOR_TRC_SMPTE2084,
                    VMAF_COLOR_MATRIX_ICTCP};
    model.pixel_format = VMAF_PIX_FMT_YUV420P;
    model.bit_depth = 16;
    model.display = display;
    return model;
}

TEST(HdrVmaf, ScoresPqVideosOnlyAndReadsTheirUntaggedColoursAsBt2020) {
    // A PQ source with P3 primaries; the untagged test video takes its tags.
    const auto p3 = hdr_colours(video("smpte2084", "bt2020nc", "smpte432"), video(""));
    ASSERT_TRUE(std::holds_alternative<HdrColours>(p3));
    EXPECT_EQ(std::get<HdrColours>(p3).distorted.primaries, VMAF_COLOR_PRIMARIES_SMPTE432);
    EXPECT_EQ(std::get<HdrColours>(p3).source.range, VMAF_COLOR_RANGE_LIMITED);

    // PQ with neither matrix nor primaries tagged: BT.2020's.
    const auto untagged = hdr_colours(video("smpte2084"), video("smpte2084"));
    ASSERT_TRUE(std::holds_alternative<HdrColours>(untagged));
    EXPECT_EQ(std::get<HdrColours>(untagged).source.matrix, VMAF_COLOR_MATRIX_BT2020_NCL);
    EXPECT_EQ(std::get<HdrColours>(untagged).source.primaries, VMAF_COLOR_PRIMARIES_BT2020);

    // SDR, HLG and untagged transfers are refused, and so is a matrix libvmaf
    // does not convert from.
    for (const char* transfer : {"bt709", "arib-std-b67", ""}) {
        const auto refused = hdr_colours(video(transfer), video(transfer));
        ASSERT_TRUE(std::holds_alternative<std::string>(refused)) << transfer;
        EXPECT_NE(std::get<std::string>(refused).find("PQ (HDR10)"), std::string::npos);
    }
    EXPECT_TRUE(std::holds_alternative<std::string>(hdr_colours(video("smpte2084", "smpte170m"), video(""))));
}

TEST(HdrVmaf, PicturesAreConvertedToTheSizesLibvmafChooses) {
    // 1080p for a 4K model: both upscaled with bicubic.
    auto targets = hdr_targets(1920, 1080, 1920, 1080, model_at(2160));
    EXPECT_EQ(targets[0].w, 3840U);
    EXPECT_EQ(targets[0].h, 2160U);
    EXPECT_EQ(targets[0].resample_filter, VMAF_RESAMPLE_BICUBIC);
    EXPECT_EQ(targets[1].resample_filter, VMAF_RESAMPLE_BICUBIC);
    EXPECT_EQ(targets[1].bpc, 16U);

    // A 4K source keeps its size; only the 1080p test video is scaled.
    targets = hdr_targets(3840, 2160, 1920, 1080, model_at(2160));
    EXPECT_EQ(targets[0].resample_filter, VMAF_RESAMPLE_DEFAULT);
    EXPECT_EQ(targets[1].resample_filter, VMAF_RESAMPLE_BICUBIC);
    EXPECT_EQ(targets[1].w, 3840U);

    // Down to a 1080p model with lanczos; sizes 2 pixels apart from rounding
    // (854x480 scales to 1922x1080) still pair, at the source's size.
    EXPECT_EQ(hdr_targets(3840, 2160, 3840, 2160, model_at(1080))[1].resample_filter, VMAF_RESAMPLE_LANCZOS);
    targets = hdr_targets(854, 480, 1280, 720, model_at(1080));
    EXPECT_EQ(targets[1].w, 1922U);

    // Refused as libvmaf refuses: other shapes, other orientations, and a long
    // side over 4x the display size.
    EXPECT_THROW(hdr_targets(1920, 1080, 1440, 1080, model_at(2160)), std::invalid_argument);
    EXPECT_THROW(hdr_targets(1920, 1080, 1080, 1920, model_at(2160)), std::invalid_argument);
    EXPECT_THROW(hdr_targets(4000, 400, 4000, 400, model_at(1080)), std::invalid_argument);
}

TEST(HdrVmaf, APictureAlreadyInTheModelsFormIsTakenAsItIs) {
    const VmafPictureConvertTarget target = hdr_targets(3840, 2160, 3840, 2160, model_at(2160))[0];
    VmafPicture picture{};
    picture.pix_fmt = VMAF_PIX_FMT_YUV420P;
    picture.bpc = 16;
    picture.w[0] = 3840;
    picture.h[0] = 2160;
    picture.color = target.color;
    EXPECT_TRUE(picture_matches(picture, target));
    picture.bpc = 10;
    EXPECT_FALSE(picture_matches(picture, target));
    picture.bpc = 16;
    picture.color.matrix = VMAF_COLOR_MATRIX_BT2020_NCL;
    EXPECT_FALSE(picture_matches(picture, target));

    EXPECT_EQ(hdr_conversion_threads(2, false), 1);
    EXPECT_EQ(hdr_conversion_threads(24, false), 8);
    EXPECT_EQ(hdr_conversion_threads(64, false), 12);
    EXPECT_EQ(hdr_conversion_threads(1, true), 1);
    EXPECT_EQ(hdr_conversion_threads(12, true), 8);
    EXPECT_EQ(hdr_conversion_threads(64, true), 16);
}

}  // namespace
}  // namespace vml
