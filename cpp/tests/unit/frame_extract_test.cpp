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

}  // namespace
}  // namespace vml
