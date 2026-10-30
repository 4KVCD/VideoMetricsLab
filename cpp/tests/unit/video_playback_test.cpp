// Video Compare's playback commands held to the Python app's (reference/
// playback.json, from its video_playback.py): each video's decoding command
// as a decoder worker runs it, for pairs with HDR, crops, black bars still to
// detect and untagged video, both sides and the source as it is played, in
// every processing and colour mode, fitted and at full size; the sizes
// decoded at; the test videos decoded beside the one shown; the soundtrack.
#include "vml/video_playback.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "reference.h"
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
    info.codec_name = data.at("codec_name");
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

FrameComparison comparison_from(const Json& data) {
    FrameComparison comparison;
    comparison.source_info = video_from(data.at("source"));
    comparison.distorted_info = video_from(data.at("distorted"));
    comparison.source_crop = crop_from(data.at("source_crop"));
    comparison.distorted_crop = crop_from(data.at("distorted_crop"));
    comparison.scale_direction = data.at("scale_direction") == "distorted_to_source"
                                     ? ScaleDirection::distorted_to_source
                                     : ScaleDirection::source_to_distorted;
    comparison.scale_algorithm = data.at("scale_algorithm");
    comparison.fps = *test::reference_number(data.at("fps"));
    comparison.frame_count = data.at("frame_count");
    comparison.auto_crop_pending = data.at("auto_crop_pending");
    return comparison;
}

std::optional<std::string> accel_from(const Json& data) {
    return data.is_null() ? std::nullopt : std::optional(data.get<std::string>());
}

// The Python app's command, its input as this system completes it: the
// Python app's paths are Windows paths, which are no absolute paths
// elsewhere (and which each standard library completes its own way).
std::vector<std::string> wanted_command(const Json& expected, [[maybe_unused]] const std::filesystem::path& input) {
    std::vector<std::string> wanted = expected.get<std::vector<std::string>>();
#ifndef _WIN32
    *(std::find(wanted.begin(), wanted.end(), "-i") + 1) = platform::path_text(std::filesystem::weakly_canonical(input));
#endif
    return wanted;
}

TEST(VideoPlayback, DecodesEachVideoAsThePythonAppDoes) {
    const Json reference = test::load_reference("playback.json");
    std::map<std::string, PreviewColorSettings> colours;
    for (const Json& item : reference.at("colours")) {
        PreviewColorSettings settings;
        settings.mode = preview_color_mode_from(item.at("mode").get<std::string>());
        if (!item.at("hdr").is_null()) {
            settings.display_hdr_enabled = item.at("hdr").get<bool>();
        }
        if (!item.at("white").is_null()) {
            settings.display_sdr_white_nits = item.at("white").get<double>();
        }
        colours[item.at("name")] = settings;
    }
    const std::map<std::string, PlaybackProcessing> modes{{"interop", PlaybackProcessing::interop},
                                                          {"transfer", PlaybackProcessing::transfer},
                                                          {"software", PlaybackProcessing::software},
                                                          {"cpu", PlaybackProcessing::cpu}};
    const Json& commands = reference.at("commands");
    ASSERT_FALSE(commands.empty());
    for (const Json& item : commands) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        // The source as the view plays it: native or scaled down.
        const Json& played = item.at("played");
        const FrameComparison comparison =
            played.is_null() ? comparison_from(item.at("comparison"))
                             : source_playback_comparison(comparison_from(item.at("comparison")), played == "native");
        const HardwareDecoding plan{accel_from(item.at("plan")[0]), accel_from(item.at("plan")[1])};
        const std::optional<Size> maximum =
            item.at("maximum").is_null() ? std::nullopt : std::optional(Size{item.at("maximum")[0], item.at("maximum")[1]});
        const FrameSide side = item.at("side") == "source" ? FrameSide::source : FrameSide::distorted;
        EXPECT_EQ(build_playback_command("<ffmpeg>", comparison, side, item.at("start_frame"),
                                         colours.at(item.at("colour")), plan, maximum, modes.at(item.at("processing"))),
                  wanted_command(item.at("command"), frame_input_path(comparison, side)));
        EXPECT_EQ(playback_dimensions(comparison, maximum), (Size{item.at("size")[0], item.at("size")[1]}));
    }

    const Json& audio = reference.at("audio");
    ASSERT_FALSE(audio.empty());
    for (const Json& item : audio) {
        const FrameComparison comparison = comparison_from(item.at("comparison"));
        EXPECT_EQ(build_audio_command("C:\\tools\\ffplay.exe", comparison, item.at("frame")),
                  wanted_command(item.at("command"), frame_input_path(comparison, FrameSide::source)));
    }

    const Json& neighbours = reference.at("neighbours");
    ASSERT_FALSE(neighbours.empty());
    for (const Json& item : neighbours) {
        EXPECT_EQ(neighbour_indices(item.at("count"), item.at("selected"), item.at("width")),
                  item.at("indices").get<std::vector<int>>())
            << item.dump();
    }
}

// The source is scaled down by its decoder only when it plays smaller than
// its crop: native, or beside a test video as large, it plays at its own
// size, whatever its crop (the presenter crops it).
TEST(VideoPlayback, OnlyASourcePlayedSmallerIsScaledByItsDecoder) {
    FrameComparison pair;
    pair.source_info.width = 3840;
    pair.source_info.height = 2160;
    pair.distorted_info = pair.source_info;
    pair.distorted_info.width = 1920;
    pair.distorted_info.height = 1080;
    pair.scale_direction = ScaleDirection::distorted_to_source;
    EXPECT_EQ(source_scaled_size(source_playback_comparison(pair, true)), std::nullopt);
    EXPECT_EQ(source_scaled_size(source_playback_comparison(pair, false)), (Size{1920, 1080}));
    // Letterboxed: cropped first, then scaled to fit the encode's crop.
    pair.source_crop = CropBox{3840, 1608, 0, 276};
    pair.distorted_crop = CropBox{1920, 804, 0, 138};
    EXPECT_EQ(source_scaled_size(source_playback_comparison(pair, false)), (Size{1920, 804}));
    pair.distorted_info = pair.source_info;  // as large as the source
    pair.distorted_crop = pair.source_crop;
    EXPECT_EQ(source_scaled_size(source_playback_comparison(pair, false)), std::nullopt);
}

}  // namespace
}  // namespace vml
