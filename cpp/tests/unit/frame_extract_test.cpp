// Video Compare's still frames held to the Python app's (reference/
// frame_extract.json, from its frame_extract.py): each side's filter chain
// and FFmpeg command in every colour mode, the seek times, the compared size,
// the HDR family of a video and the SDR white level; and, with FFmpeg, a
// real frame of a clip.
#include "vml/frame_extract.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/platform/system.h"
#include "vml/video_playback.h"

namespace vml {
namespace {

using Json = nlohmann::json;

VideoInfo video_from(const Json& data) {
    VideoInfo info;
    info.path = platform::path_from_text(data.at("path").get<std::string>());
    info.width = data.at("width");
    info.height = data.at("height");
    info.fps = *test::reference_number(data.at("fps"));
    info.sar = data.at("sar");
    info.pix_fmt = data.at("pix_fmt");
    info.color_primaries = data.at("color_primaries");
    info.color_space = data.at("color_space");
    info.color_transfer = data.at("color_transfer");
    info.color_range = data.at("color_range");
    info.start_offset = *test::reference_number(data.at("start_offset"));
    return info;
}

std::optional<CropBox> crop_from(const Json& data) {
    return data.is_null() ? std::nullopt : std::optional(CropBox{data[0], data[1], data[2], data[3]});
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

TEST(FrameExtract, PreparesFramesAsThePythonAppDoes) {
    const Json reference = test::load_reference("frame_extract.json");
    std::map<std::string, PreviewColorSettings> colours;
    const Json& colour_cases = reference.at("colours");
    ASSERT_FALSE(colour_cases.empty());
    for (const Json& item : colour_cases) {
        const PreviewColorSettings settings = colours_from(item);
        EXPECT_EQ(preview_color_mode_value(settings.mode), item.at("mode").get<std::string>());
        EXPECT_EQ(settings.target_nits(), *test::reference_number(item.at("target_nits"))) << item.at("name");
        colours[item.at("name")] = settings;
    }

    const Json& comparisons = reference.at("comparisons");
    ASSERT_FALSE(comparisons.empty());
    for (const Json& item : comparisons) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        FrameComparison comparison;
        comparison.source_info = video_from(item.at("source"));
        comparison.distorted_info = video_from(item.at("distorted"));
        comparison.source_crop = crop_from(item.at("source_crop"));
        comparison.distorted_crop = crop_from(item.at("distorted_crop"));
        comparison.scale_direction = item.at("scale_direction") == "distorted_to_source"
                                         ? ScaleDirection::distorted_to_source
                                         : ScaleDirection::source_to_distorted;
        comparison.scale_algorithm = item.at("scale_algorithm");
        if (!item.at("resample_width").is_null()) {
            comparison.resample_target = ResampleTarget{item.at("resample_width").get<int>(), "test"};
        }
        comparison.fps = *test::reference_number(item.at("fps"));
        comparison.frame_count = item.at("frame_count");
        comparison.auto_crop_pending = item.at("auto_crop_pending");
        EXPECT_EQ(comparison_dimensions(comparison), (Size{item.at("dimensions")[0], item.at("dimensions")[1]}));
        for (const Json& side_case : item.at("sides")) {
            const FrameSide side = side_case.at("side") == "source" ? FrameSide::source : FrameSide::distorted;
            SCOPED_TRACE(side_case.at("side").get<std::string>());
            for (const auto& [name, filter] : side_case.at("filters").items()) {
                EXPECT_EQ(frame_filter(comparison, side, colours.at(name)), filter.get<std::string>()) << name;
            }
            for (const auto& [frame, expected] : side_case.at("commands").items()) {
                std::vector<std::string> wanted = expected.get<std::vector<std::string>>();
                const std::vector<std::string> command =
                    build_frame_command("<ffmpeg>", comparison, side, std::stoll(frame), colours.at("display-aware, unknown"));
#ifndef _WIN32
                // The Python app's paths are Windows paths, which are no
                // absolute paths here: compared as this system completes them.
                const auto input = std::find(wanted.begin(), wanted.end(), "-i") + 1;
                *input = platform::path_text(std::filesystem::weakly_canonical(frame_input_path(comparison, side)));
#endif
                EXPECT_EQ(command, wanted) << "frame " << frame;
            }
        }
    }

    for (const Json& item : reference.at("hdr_kinds")) {
        const std::optional<std::string> kind = hdr_kind(video_from(item.at("video")));
        EXPECT_EQ(kind.value_or("none"), item.at("kind").is_null() ? "none" : item.at("kind").get<std::string>())
            << item.at("video").at("path");
    }
    for (const Json& item : reference.at("seeks")) {
        EXPECT_EQ(seek_seconds(video_from(item.at("video")), item.at("frame"), *test::reference_number(item.at("fps"))),
                  *test::reference_number(item.at("seconds")))
            << item.at("frame");
    }
    FrameComparison no_rate;
    no_rate.source_info.path = "source.mkv";
    try {
        build_frame_command("<ffmpeg>", no_rate, FrameSide::source, 0);
        ADD_FAILURE() << "no error";
    } catch (const FrameExtractError& error) {
        EXPECT_EQ(error.what(), reference.at("no_frame_rate").get<std::string>());
    }
}

// A video whose keyframes are decoded long before they are shown (VVenC's
// random access in an MP4: 1.2 s) is sought that much earlier, the frames
// before the one asked for trimmed by their times, before any filter that
// numbers them anew; near its start, from the start. A short lead, which
// FFmpeg's own allowance covers, leaves the command as it was.
TEST(FrameExtract, SeeksEarlierWhereKeyframesAreDecodedLongBeforeShown) {
    FrameComparison comparison;
    comparison.source_info.path = "source.mp4";
    comparison.source_info.width = 1920;
    comparison.source_info.height = 1080;
    comparison.distorted_info = comparison.source_info;
    comparison.fps = 30000.0 / 1001.0;
    const auto argument = [](const std::vector<std::string>& command, const std::string& name) {
        return *(std::find(command.begin(), command.end(), name) + 1);
    };
    const std::vector<std::string> plain = build_frame_command("<ffmpeg>", comparison, FrameSide::source, 460);
    EXPECT_EQ(build_frame_command("<ffmpeg>", comparison, FrameSide::source, 460, {}, 0.08), plain);
    const std::vector<std::string> early = build_frame_command("<ffmpeg>", comparison, FrameSide::source, 460, {}, 1.2012);
    EXPECT_EQ(argument(plain, "-ss"), "15.344495833");
    EXPECT_EQ(argument(early, "-ss"), "14.143295833");
    EXPECT_EQ(argument(early, "-vf"), "trim=start=1.201200000," + argument(plain, "-vf"));
    const std::vector<std::string> start = build_frame_command("<ffmpeg>", comparison, FrameSide::source, 10, {}, 1.2012);
    EXPECT_EQ(argument(start, "-ss"), "0.000000000");
    EXPECT_EQ(argument(start, "-vf").substr(0, 23), "trim=start=0.329495833,");
    const std::vector<std::string> playback = build_playback_command(
        "<ffmpeg>", comparison, FrameSide::source, 460, {}, {}, std::nullopt, PlaybackProcessing::cpu, 1.2012);
    EXPECT_EQ(argument(playback, "-ss"), "14.143295833");
    EXPECT_NE(argument(playback, "-filter_complex").find("[in0]trim=start=1.201200000,"), std::string::npos);
}

// A test video scored scaled to one size because its size changes (a 720p
// start, scored at the 1080p source's size) is shown as decoded: its crop,
// in the 1920x1080 pixels it was scored at, in its own 1280x720 ones.
TEST(FrameExtract, ShowsAVideoScoredAtAnotherSizeAsDecoded) {
    ComparisonResult result;
    result.source_info.width = 1920;
    result.source_info.height = 1080;
    result.distorted_info.width = 1280;
    result.distorted_info.height = 720;
    result.distorted_info.scaled_to = std::pair(1920, 1080);
    result.source_crop = CropBox{1920, 596, 0, 242};
    result.distorted_crop = CropBox{1920, 596, 0, 242};
    FrameComparison shown = FrameComparison::from_result(result);
    EXPECT_FALSE(shown.distorted_info.scaled_to.has_value());
    EXPECT_EQ(shown.distorted_crop, (CropBox{1280, 396, 0, 162}));
    EXPECT_EQ(shown.source_crop, result.source_crop);
    result.distorted_crop = CropBox{1920, 1080, 0, 0};
    shown = FrameComparison::from_result(result);
    EXPECT_TRUE(shown.distorted_crop->is_noop(1280, 720));
}

}  // namespace
}  // namespace vml
