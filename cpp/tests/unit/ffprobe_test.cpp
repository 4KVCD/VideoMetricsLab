#include "vml/ffprobe.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "reference.h"
#include "test_videos.h"
#include "vml/platform/system.h"

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

// Reading a video is ended by Cancel: closing the window, or stopping the
// Bitrate tab's scan, ends an ffprobe that takes long (a large file on a
// slow disk) rather than waiting up to the 60 s it is given. The stand-in
// ffprobe never answers.
TEST(Ffprobe, ReadingAVideoIsEndedByCancel) {
    const test::Folder folder;
    const std::filesystem::path log = folder / "log";
    std::filesystem::create_directory(log);
    const test::ScopedVariable tool("VML_TEST_TOOL", "hang");
    const test::ScopedVariable logged_to("VML_TEST_TOOL_LOG", platform::path_text(log));
    CancelToken cancel;
    std::thread user([&log, &cancel] {
        // Once the stand-in is running.
        for (int wait = 0; wait < 6000 && std::filesystem::is_empty(log); ++wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        cancel.cancel();
    });
    EXPECT_THROW(probe_video(folder / "video.mkv", VML_TEST_CHILD, cancel), Cancelled);
    user.join();
}

}  // namespace
}  // namespace vml
