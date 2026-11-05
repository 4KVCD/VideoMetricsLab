#include "vml/ffprobe.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "reference.h"
#include "test_videos.h"
#include "vml/geometry.h"
#include "vml/platform/system.h"
#include "vml/run_json.h"

namespace vml {
namespace {

// ffprobe's output is read as the Python app reads it, field for field: on
// the golden clips' real output, and on written output for Matroska's tags,
// soundtracks, cover art, odd and missing values, no video and no JSON.
TEST(Ffprobe, ReadsTheVideosDetailsAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("probe.json");
    ASSERT_FALSE(reference["cases"].empty());
    for (const nlohmann::json& item : reference["cases"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        const std::filesystem::path path = platform::path_from_text(item["path"].get<std::string>());
        const nlohmann::json& expected = item["expected"];
        if (expected.contains("error")) {
            try {
                video_info_from_probe(path, item["output"].get<std::string>());
                ADD_FAILURE() << "no error";
            } catch (const ProbeError& error) {
                EXPECT_EQ(std::string(error.what()), expected["error"].get<std::string>());
            }
            continue;
        }
        const VideoInfo info = video_info_from_probe(path, item["output"].get<std::string>());
        EXPECT_EQ(platform::path_text(info.path), expected["path"].get<std::string>());
        EXPECT_EQ(info.width, expected["width"].get<int>());
        EXPECT_EQ(info.height, expected["height"].get<int>());
        EXPECT_EQ(info.fps, *test::reference_number(expected["fps"]));
        EXPECT_EQ(info.duration, *test::reference_number(expected["duration"]));
        EXPECT_EQ(info.nb_frames, expected["nb_frames"].get<long long>());
        EXPECT_EQ(info.codec_name, expected["codec_name"].get<std::string>());
        EXPECT_EQ(info.sar, expected["sar"].get<std::string>());
        EXPECT_EQ(info.pix_fmt, expected["pix_fmt"].get<std::string>());
        EXPECT_EQ(info.bit_rate, expected["bit_rate"].get<long long>());
        EXPECT_EQ(info.bit_rate_whole_file, expected["bit_rate_whole_file"].get<bool>());
        EXPECT_EQ(info.nominal_fps, *test::reference_number(expected["nominal_fps"]));
        EXPECT_EQ(info.average_fps, *test::reference_number(expected["average_fps"]));
        EXPECT_EQ(info.color_range, expected["color_range"].get<std::string>());
        EXPECT_EQ(info.color_space, expected["color_space"].get<std::string>());
        EXPECT_EQ(info.color_transfer, expected["color_transfer"].get<std::string>());
        EXPECT_EQ(info.color_primaries, expected["color_primaries"].get<std::string>());
        EXPECT_EQ(info.chroma_location, expected["chroma_location"].get<std::string>());
        EXPECT_EQ(info.format_name, expected["format_name"].get<std::string>());
        EXPECT_EQ(info.start_offset, *test::reference_number(expected["start_offset"]));
    }
}

// A rotation tag as ffprobe gives it, and a pair stored at different rotations
// refused: compared as stored, one would be on its side against the other.
// An HDR video's frames list a column for their side data after the size.
TEST(Ffprobe, FrameSizesAreReadWithoutTheirSideData) {
    EXPECT_FALSE(listed_sizes_differ("3840,2160,\n3840,2160,\n", 3840, 2160));
    EXPECT_FALSE(listed_sizes_differ("3840,2160\n", 3840, 2160));
    EXPECT_TRUE(listed_sizes_differ("3840,2160,\n2560,1440,\n", 3840, 2160));
    EXPECT_TRUE(listed_sizes_differ("1920,1080\n", 3840, 2160));
    EXPECT_FALSE(listed_sizes_differ("", 3840, 2160));
}

TEST(Ffprobe, ReadsARotationTagAndAPairMustShareIt) {
    const auto video_turned = [](const std::string& side_data) {
        return video_info_from_probe(
            "a.mp4", R"({"streams":[{"codec_type":"video","codec_name":"h264","width":640,"height":268,)"
                     R"("pix_fmt":"yuv420p","r_frame_rate":"24/1","avg_frame_rate":"24/1","nb_frames":"24",)"
                     R"("duration":"1.0")" + side_data + R"(}],"format":{"duration":"1.0"}})");
    };
    const VideoInfo upright = video_turned("");
    const VideoInfo upside_down = video_turned(R"(,"side_data_list":[{"side_data_type":"Display Matrix","rotation":-180}])");
    const VideoInfo sideways = video_turned(R"(,"side_data_list":[{"side_data_type":"Display Matrix","rotation":90}])");
    EXPECT_EQ(upright.rotation, 0);
    EXPECT_EQ(upside_down.rotation, 180);
    EXPECT_EQ(sideways.rotation, 90);
    EXPECT_EQ(pair_problem(upside_down, upside_down, 0, 0), std::nullopt);
    EXPECT_NE(pair_problem(upside_down, upright, 0, 0), std::nullopt);
}

// Matroska's statistics tags tell a stream's rate, unless FFmpeg wrote the
// file: then they are its input's, even through a re-encode.
TEST(Ffprobe, StatisticsTagsCountUnlessFfmpegCopiedThem) {
    const auto rate_when_written_by = [](const std::string& writer) {
        const std::string probe = R"({"streams":[{"codec_type":"video","codec_name":"hevc","width":1280,"height":720,)"
                                  R"("pix_fmt":"yuv420p","r_frame_rate":"24/1","avg_frame_rate":"24/1",)"
                                  R"("tags":{"BPS":"100000"}}],)"
                                  R"("format":{"duration":"10.0","size":"789000","bit_rate":"631200",)"
                                  R"("format_name":"matroska,webm","tags":{"ENCODER":")" +
                                  writer + R"("}}})";
        return video_info_from_probe("a.mkv", probe).bit_rate;
    };
    EXPECT_EQ(rate_when_written_by("libebml v1.4.5 + libmatroska v1.7.1"), 100000);
    EXPECT_EQ(rate_when_written_by("Lavf63.1.101"), 631200);
}

// As measured on streams made at known rates: what ffprobe reports of a raw
// stream told to read it at a rate no stream has.
TEST(Ffprobe, ARawStreamsFrameRateIsOnlyWhatItsHeadersStore) {
    EXPECT_DOUBLE_EQ(*stored_frame_rate("hevc", 24000.0 / 1001), 24000.0 / 1001);
    EXPECT_DOUBLE_EQ(*stored_frame_rate("vvc", 60), 60);
    EXPECT_DOUBLE_EQ(*stored_frame_rate("mpeg2video", 25), 25);  // a real 25 is not taken for FFmpeg's guess
    // Reported doubled.
    EXPECT_DOUBLE_EQ(*stored_frame_rate("h264", 60), 30);
    EXPECT_DOUBLE_EQ(*stored_frame_rate("mpeg1video", 48000.0 / 1001), 24000.0 / 1001);
    // Stored by none: the rate it was told to read at, or the demuxer's ticks.
    EXPECT_FALSE(stored_frame_rate("av1", 1200000));
    EXPECT_FALSE(stored_frame_rate("hevc", 1200000));
    EXPECT_FALSE(stored_frame_rate("hevc", 1999));
    EXPECT_FALSE(stored_frame_rate("hevc", 0));
    // Not the stream's: H.263's clock, whatever rate it was made at; or a codec not shown to store one.
    EXPECT_FALSE(stored_frame_rate("h263", 30000.0 / 1001));
    EXPECT_FALSE(stored_frame_rate("mjpeg", 25));
}

// A run saved before the rate was looked for holds FFmpeg's guess at 25, which
// must not come back as the video's rate. One whose rate was known keeps it.
TEST(Ffprobe, ASavedRawStreamKeepsItsRateButNotFfmpegsGuess) {
    VideoInfo raw;
    raw.path = "clip.hevc";
    raw.format_name = "hevc";
    raw.nb_frames = 100;
    raw.fps = raw.nominal_fps = raw.average_fps = 25;
    raw.duration = 4;
    const VideoInfo guessed = video_from(video_json(raw));
    EXPECT_TRUE(guessed.frame_rate_unknown());
    EXPECT_EQ(guessed.fps, 0);
    EXPECT_EQ(guessed.duration, 0);

    raw.set_frame_rate(50);
    const VideoInfo known = video_from(video_json(raw));
    EXPECT_FALSE(known.frame_rate_unknown());
    EXPECT_EQ(known.fps, 50);
    EXPECT_EQ(known.duration, 2);
}

}  // namespace
}  // namespace vml
