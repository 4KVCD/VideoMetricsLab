// Vship's view of a video, held to the Python app's (vship.json, from
// perceptual_vship.py and colour.py): the layouts FFmpeg sends frames in and
// their sizes, the colour spaces Vship is given, the filter chains and
// outputs, the sizes compared at, and the GPUs and sizes a metric is refused
// at. On a PC with a GPU Vship can use, the probe finds it.
#include "vml/vship.h"

#include <gtest/gtest.h>

#include <string>

#include "reference.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

using Json = nlohmann::json;

const Json& vship_reference() {
    static const Json reference = test::load_reference("vship.json");
    return reference;
}

std::optional<std::string> text_or_none(const Json& value) {
    return value.is_null() ? std::nullopt : std::optional(value.get<std::string>());
}

VideoInfo video_from(const Json& info) {
    VideoInfo video;
    video.path = "video.mkv";
    video.width = info["width"].get<int>();
    video.height = info["height"].get<int>();
    video.fps = 24.0;
    video.duration = 10.0;
    video.nb_frames = 240;
    video.codec_name = "hevc";
    video.pix_fmt = info["pix_fmt"].get<std::string>();
    video.color_range = info["color_range"].get<std::string>();
    video.color_space = info["color_space"].get<std::string>();
    video.color_transfer = info["color_transfer"].get<std::string>();
    video.color_primaries = info["color_primaries"].get<std::string>();
    video.chroma_location = info["chroma_location"].get<std::string>();
    return video;
}

VideoInfo plain_video(const std::string& pix_fmt, const std::string& color_range = "") {
    VideoInfo video;
    video.path = "video.mkv";
    video.width = 1920;
    video.height = 1080;
    video.pix_fmt = pix_fmt;
    video.color_range = color_range;
    return video;
}

std::optional<CropBox> crop_from(const Json& crop) {
    if (crop.is_null()) {
        return std::nullopt;
    }
    return CropBox{crop[0].get<int>(), crop[1].get<int>(), crop[2].get<int>(), crop[3].get<int>()};
}

void expect_image(const Json& expected, const VshipImageFormat& image) {
    EXPECT_EQ(image.pixel_format, expected["pixel_format"].get<std::string>());
    EXPECT_EQ(image.family, expected["family"].get<int>());
    EXPECT_EQ(image.sample, expected["sample"].get<int>());
    EXPECT_EQ(image.subw, expected["subw"].get<int>());
    EXPECT_EQ(image.subh, expected["subh"].get<int>());
    EXPECT_EQ(image.full_range, expected["full_range"].get<bool>());
    EXPECT_EQ((std::vector<int>(image.plane_order.begin(), image.plane_order.end())),
              expected["plane_order"].get<std::vector<int>>());
}

// Each decoded pixel format becomes the Python app's layout, with its
// frames' sizes, or the same refusal.
TEST(Vship, LaysFramesOutAsThePythonApp) {
    for (const Json& expected : vship_reference()["formats"]) {
        SCOPED_TRACE(expected["pix_fmt"].get<std::string>());
        if (expected.contains("error")) {
            try {
                vship_image_format(plain_video(expected["pix_fmt"].get<std::string>()));
                ADD_FAILURE() << "no error";
            } catch (const VshipUnavailableError& error) {
                EXPECT_EQ(error.what(), expected["error"].get<std::string>());
            }
            continue;
        }
        const VshipImageFormat image = vship_image_format(plain_video(expected["pix_fmt"].get<std::string>()));
        expect_image(expected["image"], image);
        for (const Json& layout : expected["layouts"]) {
            const VshipImageFormat::Layout got = image.frame_layout(layout["width"].get<int>(), layout["height"].get<int>());
            EXPECT_EQ(got.frame_bytes, layout["frame_bytes"].get<std::size_t>());
            EXPECT_EQ((std::vector<std::size_t>(got.plane_bytes.begin(), got.plane_bytes.end())),
                      layout["plane_bytes"].get<std::vector<std::size_t>>());
            EXPECT_EQ((std::vector<std::int64_t>(got.strides.begin(), got.strides.end())),
                      layout["strides"].get<std::vector<std::int64_t>>());
        }
    }
    for (const Json& expected : vship_reference()["passthrough"]) {
        SCOPED_TRACE(expected.dump());
        const VideoInfo info = plain_video(expected["pix_fmt"].get<std::string>(),
                                           expected["color_range"].get<std::string>());
        if (expected.contains("error")) {
            EXPECT_THROW(passthrough_format(info, text_or_none(expected["hwaccel"])), VshipUnavailableError);
            continue;
        }
        const auto found = passthrough_format(info, text_or_none(expected["hwaccel"]));
        ASSERT_EQ(found.has_value(), !expected["passthrough"].is_null());
        if (found) {
            expect_image(expected["passthrough"]["image"], found->first);
            EXPECT_EQ(found->second, expected["passthrough"]["piped"].get<std::string>());
        }
    }
}

// The colour space Vship is given, from the video's tags and the layout, or
// the same refusal.
TEST(Vship, DescribesColoursAsThePythonApp) {
    for (const Json& expected : vship_reference()["colours"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        const VideoInfo info = video_from(expected["info"]);
        try {
            const vship_api::Colorspace space = vship_colorspace(info, vship_image_format(info), 1280, 720);
            ASSERT_TRUE(expected.contains("colorspace")) << "no error";
            const Json& wanted = expected["colorspace"];
            EXPECT_EQ(space.width, wanted["width"].get<std::int64_t>());
            EXPECT_EQ(space.height, wanted["height"].get<std::int64_t>());
            EXPECT_EQ(space.target_width, wanted["target_width"].get<std::int64_t>());
            EXPECT_EQ(space.target_height, wanted["target_height"].get<std::int64_t>());
            EXPECT_EQ(space.sample, wanted["sample"].get<int>());
            EXPECT_EQ(space.range, wanted["range"].get<int>());
            EXPECT_EQ(space.subsampling.subw, wanted["subsampling"][0].get<int>());
            EXPECT_EQ(space.subsampling.subh, wanted["subsampling"][1].get<int>());
            EXPECT_EQ(space.chromaLocation, wanted["chroma_location"].get<int>());
            EXPECT_EQ(space.colorFamily, wanted["family"].get<int>());
            EXPECT_EQ(space.YUVMatrix, wanted["matrix"].get<int>());
            EXPECT_EQ(space.transferFunction, wanted["transfer"].get<int>());
            EXPECT_EQ(space.primaries, wanted["primaries"].get<int>());
            EXPECT_EQ((std::vector<int>{space.crop.top, space.crop.bottom, space.crop.left, space.crop.right}),
                      wanted["crop"].get<std::vector<int>>());
        } catch (const VshipUnavailableError& error) {
            ASSERT_TRUE(expected.contains("error")) << error.what();
            EXPECT_EQ(error.what(), expected["error"].get<std::string>());
        }
    }
}

// A video that gives no chroma siting is read with the other's: a y4m
// file's "center" (as FFmpeg writes it for a source without one) against a
// test video without a siting was read half a pixel apart from it.
TEST(Vship, AVideoWithoutChromaSitingIsReadWithTheOthers) {
    using Sitings = std::pair<std::string, std::string>;
    EXPECT_EQ(shared_chroma_location("center", "unspecified"), Sitings("center", "center"));
    EXPECT_EQ(shared_chroma_location("", "topleft"), Sitings("topleft", "topleft"));
    EXPECT_EQ(shared_chroma_location("left", "center"), Sitings("left", "center"));  // both given
    EXPECT_EQ(shared_chroma_location("", "unknown"), Sitings("", "unknown"));        // neither given
    EXPECT_TRUE(chroma_location_shared("center", ""));
    EXPECT_FALSE(chroma_location_shared("", "left"));  // read as left before too
    EXPECT_FALSE(chroma_location_shared("left", "center"));
}

// The filters and outputs FFmpeg is given, the sizes compared at, and the
// sizes a metric is refused at on a GPU.
TEST(Vship, PlansFfmpegAndSizesAsThePythonApp) {
    for (const Json& expected : vship_reference()["chains"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        std::optional<Rational> numbered_at;
        if (!expected["numbered_at"].is_null()) {
            numbered_at = Rational{expected["numbered_at"][0].get<long long>(), expected["numbered_at"][1].get<long long>()};
        }
        EXPECT_EQ(vship_filter_chain(video_from(expected["info"]), crop_from(expected["crop"]),
                                     std::pair(expected["target"][0].get<int>(), expected["target"][1].get<int>()),
                                     expected["pixel_format"].get<std::string>(), expected["step"].get<int>(),
                                     expected["algorithm"].get<std::string>(), text_or_none(expected["hwaccel"]),
                                     numbered_at),
                  expected["chain"].get<std::string>());
    }
    for (const auto& [pixel_format, arguments] : vship_reference()["piped"].items()) {
        EXPECT_EQ(vship_piped_as(pixel_format), arguments.get<std::vector<std::string>>()) << pixel_format;
    }
    for (const Json& expected : vship_reference()["sizes"]) {
        SCOPED_TRACE(expected.dump());
        VideoInfo source = plain_video("yuv420p");
        source.width = expected["source"][0].get<int>();
        source.height = expected["source"][1].get<int>();
        VideoInfo test = plain_video("yuv420p");
        test.width = expected["test"][0].get<int>();
        test.height = expected["test"][1].get<int>();
        const auto [source_size, test_size] = vship_scaled_sizes(
            source, test,
            expected["to_source"].get<bool>() ? ScaleDirection::distorted_to_source : ScaleDirection::source_to_distorted,
            crop_from(expected["source_crop"]), crop_from(expected["test_crop"]));
        EXPECT_EQ((std::vector<int>{source_size.first, source_size.second}), expected["sizes"][0].get<std::vector<int>>());
        EXPECT_EQ((std::vector<int>{test_size.first, test_size.second}), expected["sizes"][1].get<std::vector<int>>());
    }
    for (const Json& expected : vship_reference()["refusals"]) {
        SCOPED_TRACE(expected.dump());
        VshipDevice device{"vulkan", "a GPU", 0, "5.1.2", std::nullopt};
        const std::string vendor = expected["vendor"].get<std::string>();
        device.vendor = vendor == "intel"    ? std::optional(GpuVendor::intel)
                        : vendor == "nvidia" ? std::optional(GpuVendor::nvidia)
                        : vendor == "amd"    ? std::optional(GpuVendor::amd)
                        : vendor == "none"   ? std::optional(GpuVendor::none)
                                             : std::nullopt;
        std::optional<std::pair<int, int>> computed;
        if (!expected["computed"].is_null()) {
            computed = std::pair(expected["computed"][0].get<int>(), expected["computed"][1].get<int>());
        }
        EXPECT_EQ(size_refusal(device, expected["key"].get<std::string>(),
                               std::pair(expected["size"][0].get<int>(), expected["size"][1].get<int>()), computed),
                  text_or_none(expected["refusal"]));
    }
}

}  // namespace
}  // namespace vml
