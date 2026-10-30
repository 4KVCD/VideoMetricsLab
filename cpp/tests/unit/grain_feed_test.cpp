// Pictures the app decodes itself, fed to FFmpeg as a rawvideo input
// (grain_feed.h), and the pipe they go through.
#include "vml/grain_feed.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <future>
#include <iterator>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/perceptual_cpu.h"
#include "vml/platform/pipe.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

using Arguments = std::vector<std::string>;

// A video at the root of the drive (input_path of a relative name would look
// through every folder above the tests').
VideoInfo pq_video(const char* name) {
    VideoInfo info;
    info.path = std::filesystem::path("/") / name;
    info.width = 3840;
    info.height = 2160;
    info.fps = 23.976023976023978;
    info.nominal_fps = 24000.0 / 1001.0;
    info.codec_name = "vvc";
    info.pix_fmt = "yuv420p10le";
    info.color_range = "tv";
    info.color_primaries = "bt2020";
    info.color_transfer = "smpte2084";
    info.color_space = "bt2020nc";
    info.chroma_location = "topleft";
    return info;
}

// `command` with the file input {"-i", the video's path} in it replaced by
// `fed`: what a fed video's command is to be, all else the same.
Arguments with_fed_input(Arguments command, const VideoInfo& video, const Arguments& fed) {
    const Arguments file{"-i", input_path(video.path)};
    const auto at = std::search(command.begin(), command.end(), file.begin(), file.end());
    EXPECT_NE(at, command.end());
    const auto index = at - command.begin();
    command.erase(at, at + 2);
    command.insert(command.begin() + index, fed.begin(), fed.end());
    return command;
}

// The pictures are rawvideo of the video's size, layout and nominal rate,
// tagged as FFmpeg's decoding of the file tags its frames; untagged stays
// untagged, and ffprobe's name of RGB's matrix is the option's "rgb".
TEST(GrainFeed, FedPicturesAreRawvideoTaggedAsTheFilesFramesAre) {
    EXPECT_EQ(feed_input_options(pq_video("a.mkv")),
              (Arguments{"-f", "rawvideo", "-video_size", "3840x2160", "-pixel_format", "yuv420p10le", "-framerate",
                         "23.976023976023978", "-color_range", "tv", "-color_primaries", "bt2020", "-color_trc",
                         "smpte2084", "-colorspace", "bt2020nc", "-chroma_sample_location", "topleft"}));
    VideoInfo untagged = pq_video("b.mkv");
    untagged.pix_fmt = "yuv420p";
    untagged.nominal_fps = 25;  // r_frame_rate: FFmpeg's rate for the file
    untagged.color_range = "unknown";
    untagged.color_primaries.clear();
    untagged.color_transfer = "unspecified";
    untagged.color_space = "gbr";
    untagged.chroma_location = "unspecified";
    EXPECT_EQ(feed_input_options(untagged), (Arguments{"-f", "rawvideo", "-video_size", "3840x2160", "-pixel_format",
                                                       "yuv420p", "-framerate", "25.0", "-colorspace", "rgb"}));
    VideoInfo raw = pq_video("c.266");
    raw.timed_raw = true;  // read at the rate given for it
    raw.fps = 50;
    EXPECT_EQ(feed_input_options(raw)->at(7), "50.0");
    // As FFmpeg's av_guess_frame_rate: the average where it is below 70 and
    // r_frame_rate above 210.
    VideoInfo rated = pq_video("g.mkv");
    for (const auto [nominal, average, expected] : {std::tuple(240.0, 60.0, "60.0"), std::tuple(200.0, 24.0, "200.0"),
                                                    std::tuple(1000.0, 0.0, "1000.0")}) {
        rated.nominal_fps = nominal;
        rated.average_fps = average;
        EXPECT_EQ(feed_input_options(rated)->at(7), expected) << nominal << " / " << average;
    }
}

// A layout the decoder would convert, a tag no decoder option names, or a
// size that changes on the way is not fed: the metric is refused instead.
TEST(GrainFeed, WhatRawvideoCannotCarryAsTheFileIsIsNotFed) {
    VideoInfo four_two_two = pq_video("d.mkv");
    four_two_two.pix_fmt = "yuv422p10le";
    EXPECT_EQ(feed_input_options(four_two_two), std::nullopt);
    EXPECT_TRUE(feed_refusal(four_two_two).has_value());
    VideoInfo gamut = pq_video("e.mkv");
    gamut.color_primaries = "vgamut";
    EXPECT_EQ(feed_input_options(gamut), std::nullopt);
    VideoInfo scaled = pq_video("f.mkv");
    scaled.scaled_to = std::pair(1920, 1080);
    EXPECT_TRUE(feed_refusal(scaled).has_value());
}

// Each route's command has a fed video's input in its file's place, and no
// GPU decoding for it; the other video's input and everything after are as
// for the files.
TEST(GrainFeed, AFedVideoTakesItsFilesPlaceInEveryCommand) {
    const VideoInfo test = pq_video("test.mkv");
    const VideoInfo source = pq_video("source.mkv");
    const Arguments fed{"-f", "rawvideo", "-i", R"(\\.\pipe\fed)"};
    const HardwareDecoding source_on_gpu{"cuda", std::nullopt};

    InputOptions files;
    InputOptions test_fed;
    test_fed.distorted_fed = fed;
    EXPECT_EQ(build_ffmpeg_command("ffmpeg", test.path, source.path, "[0:v]null[a]", source_on_gpu, 0, test_fed),
              with_fed_input(build_ffmpeg_command("ffmpeg", test.path, source.path, "[0:v]null[a]", source_on_gpu, 0,
                                                  files),
                             test, fed));

    const std::pair<std::string, std::string> graphs{"[0:v]null[vmaf_dist]", "[0:v]null[vmaf_ref]"};
    const std::pair<Arguments, Arguments> outputs{{"<test pipe>"}, {"<source pipe>"}};
    InputOptions source_fed;
    source_fed.source_fed = fed;
    std::vector<Arguments> streams =
        build_stream_commands("ffmpeg", test.path, source.path, graphs, HardwareDecoding{}, outputs, files);
    streams[1] = with_fed_input(streams[1], source, fed);
    EXPECT_EQ(build_stream_commands("ffmpeg", test.path, source.path, graphs, HardwareDecoding{}, outputs, source_fed),
              streams);

    EXPECT_EQ(build_resample_command("ffmpeg", source.path, "[0:v]null", std::nullopt, 0, {}, fed),
              with_fed_input(build_resample_command("ffmpeg", source.path, "[0:v]null", std::nullopt, 0), source, fed));

    VideoInfo plain_test = test;
    plain_test.pix_fmt = "yuv420p";
    const std::filesystem::path folder = std::filesystem::temp_directory_path();
    EXPECT_EQ(png_pairs_command("ffmpeg", source, plain_test, VmafOptions{}, std::nullopt, std::nullopt, 1, folder,
                                fed, std::nullopt),
              with_fed_input(png_pairs_command("ffmpeg", source, plain_test, VmafOptions{}, std::nullopt, std::nullopt,
                                               1, folder),
                             plain_test, fed));
}

// A write into a pipe whose reader closed its end ends the feeding without
// an error; one the system refused for want of memory is made again.
TEST(Pipes, AFailedWriteMeansTheReaderWentTryAgainOrAFailure) {
    using platform::PipeWriteError;
    EXPECT_EQ(platform::pipe_write_error(109), PipeWriteError::reader_gone);   // ERROR_BROKEN_PIPE
    EXPECT_EQ(platform::pipe_write_error(232), PipeWriteError::reader_gone);   // ERROR_NO_DATA
    EXPECT_EQ(platform::pipe_write_error(1450), PipeWriteError::try_again);    // ERROR_NO_SYSTEM_RESOURCES
    EXPECT_EQ(platform::pipe_write_error(1453), PipeWriteError::try_again);    // ERROR_WORKING_SET_QUOTA
    EXPECT_EQ(platform::pipe_write_error(1816), PipeWriteError::try_again);    // ERROR_NOT_ENOUGH_QUOTA
    EXPECT_EQ(platform::pipe_write_error(5), PipeWriteError::failed);          // ERROR_ACCESS_DENIED
}

#ifdef _WIN32
// Everything written, in pieces bigger and smaller than the pipe, is read
// whole before the reader finds the end.
TEST(Pipes, WhatIsWrittenIsReadWholeThenItsEnd) {
    const std::string path = platform::OutboundPipe::unique_path("test");
    platform::OutboundPipe pipe(path, 1 << 20);
    std::vector<char> written(3 * 1024 * 1024 + 17);
    for (std::size_t i = 0; i < written.size(); ++i) {
        written[i] = static_cast<char>(i * 31 + 7);
    }
    std::future<std::vector<char>> read = std::async(std::launch::async, [&path] {
        std::ifstream reader(platform::path_from_text(path), std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(reader), std::istreambuf_iterator<char>());
    });
    pipe.connect();
    EXPECT_TRUE(pipe.write(written.data(), 100));
    EXPECT_TRUE(pipe.write(written.data() + 100, written.size() - 100));
    EXPECT_TRUE(pipe.finish());
    EXPECT_EQ(read.get(), written);
}

// A reader that never came (touch: an FFmpeg that failed first) lets the
// writer's wait end, and its first write finds it gone.
TEST(Pipes, AReaderThatNeverReadIsGone) {
    const std::string path = platform::OutboundPipe::unique_path("test");
    platform::OutboundPipe pipe(path, 1 << 20);
    platform::OutboundPipe::touch(path);
    pipe.connect();
    const char picture[16] = {};
    EXPECT_FALSE(pipe.write(picture, sizeof(picture)));
}
#endif

}  // namespace
}  // namespace vml
