// A raw stream (a .hevc file with no container), timed at the frame rate
// the command line's --fps gives it, scored as the same stream in a
// container: the same frames, black bars and scores, the frames decoded by
// FFmpeg (the app's own decoders need packet timestamps, which a raw stream
// has none of).
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "test_videos.h"
#include "vml/app_log.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffprobe.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

// Rewrites the HEVC stream at `path` with only the parameter sets (VPS, SPS,
// PPS) from before its first picture, as x265's own command line writes one
// unless told to repeat them (--repeat-headers).
void keep_first_parameter_sets(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    const std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    constexpr std::string_view start_code("\0\0\1", 3);
    std::size_t unit = data.find(start_code);
    std::string kept = data.substr(0, unit);
    bool picture = false;
    while (unit != std::string::npos) {
        const std::size_t next = data.find(start_code, unit + start_code.size());
        const int type = (static_cast<unsigned char>(data[unit + 3]) >> 1) & 0x3F;
        picture = picture || type <= 31;
        if (!picture || type < 32 || type > 34) {
            kept.append(data, unit, (next == std::string::npos ? data.size() : next) - unit);
        }
        unit = next;
    }
    std::ofstream(path, std::ios::binary | std::ios::trunc) << kept;
}

TEST(RawStream, ScoresAsTheSameStreamInAContainer) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    // Letterboxed, 640x280 pictures in a 640x360 frame.
    const std::vector<std::string> bars{"-vf", "pad=640:360:0:40"};
    const auto with_bars = [&bars](std::vector<std::string> options) {
        options.insert(options.end(), bars.begin(), bars.end());
        return options;
    };
    const VideoInfo short_source = probe_video(
        test::make_clip(*found, folder / "short.mkv", "libx264", "yuv420p", 4.0, with_bars({"-crf", "10"}), "640x280"),
        found->ffprobe);
    const VideoInfo long_source = probe_video(
        test::make_clip(*found, folder / "long.mkv", "libx264", "yuv420p", 8.0, with_bars({"-crf", "10"}), "640x280"),
        found->ffprobe);
    const RunEnvironment environment{found->ffmpeg, found->version, found->ffprobe,
                                     DecodingCapabilities::detect(found->ffmpeg), platform::logical_cores()};

    // The app's log lines, for how the black bars were sampled.
    std::mutex lines_mutex;
    std::vector<std::string> lines;
    app_log::forward_lines([&](int, std::string_view, std::string_view text) {
        const std::lock_guard lock(lines_mutex);
        lines.emplace_back(text);
    });
    struct StopForwarding {
        ~StopForwarding() { app_log::forward_lines({}); }
    } stop_forwarding;

    // B-frames, whose order a raw stream's reader must work out, and black
    // bars found two ways: in 4 s whose only keyframe is the first, in one
    // pass from the start (its windows overlap, as a short video's do); in
    // 8 s with a keyframe every half second and its parameter sets only at
    // its start, from bytes partway through.
    const struct {
        const char* name;
        const VideoInfo* source;
        double seconds;
        std::size_t keyframe_interval;
    } streams[] = {
        {"one keyframe", &short_source, 4.0, 1000},
        {"a keyframe every 12 frames", &long_source, 8.0, 12},
    };
    for (const auto& stream : streams) {
        SCOPED_TRACE(stream.name);
        const auto frames = static_cast<std::size_t>(std::llround(stream.seconds * 24000 / 1001));
        const bool one_keyframe = stream.keyframe_interval > frames;
        const std::string name = one_keyframe ? "one" : "many";
        const std::filesystem::path contained_path =
            test::make_clip(*found, folder / (name + ".mkv"), "libx265", "yuv420p10le", stream.seconds,
                            with_bars({"-crf", "30", "-g", std::to_string(stream.keyframe_interval)}), "640x280");
        test::run_ffmpeg(*found, {"-i", platform::path_text(contained_path), "-c", "copy", "-f", "hevc",
                                  platform::path_text(folder / (name + ".hevc"))});
        if (!one_keyframe) {
            keep_first_parameter_sets(folder / (name + ".hevc"));
        }
        const VideoInfo contained = probe_video(contained_path, found->ffprobe);
        const VideoInfo raw =
            timed_raw_stream(probe_video(folder / (name + ".hevc"), found->ffprobe), 24000.0 / 1001, found->ffprobe);
        ASSERT_TRUE(raw.is_raw_stream());
        EXPECT_EQ(raw.nb_frames, static_cast<long long>(frames));
        EXPECT_EQ(raw.input_options(), (std::vector<std::string>{"-framerate", "23.976023976023978"}));

        // VMAF on the CPU (FFmpeg's libvmaf filter), and PSNR and SSIM alone,
        // which the app scores itself: from its own decoders for the
        // container, through FFmpeg's pipes for the raw stream.
        for (const bool vmaf : {true, false}) {
            SCOPED_TRACE(vmaf ? "VMAF and PSNR" : "PSNR and SSIM");
            std::vector<std::string> raw_statuses;
            const auto score = [&](const VideoInfo& test_video, std::vector<std::string>* statuses) {
                VmafOptions run_options;
                run_options.compute_vmaf = vmaf;
                run_options.vmaf_on_gpu = false;
                run_options.extra_features = vmaf ? std::vector<std::string>{"name=psnr"}
                                                  : std::vector<std::string>{"name=psnr", "name=float_ssim"};
                run_options.crop_mode = CropMode::automatic;
                return run_vmaf(
                    environment, *stream.source, test_video, run_options, nullptr,
                    [statuses](const RunStatus& status) {
                        if (statuses != nullptr) {
                            statuses->push_back(status.text);
                        }
                    },
                    CancelToken());
            };
            const ComparisonResult from_container = score(contained, nullptr);
            const ComparisonResult from_raw = score(raw, &raw_statuses);

            ASSERT_TRUE(from_container.distorted_crop.has_value());
            EXPECT_EQ(from_container.distorted_crop->h, 280);
            EXPECT_EQ(from_raw.distorted_crop, from_container.distorted_crop);
            for (const std::string key : vmaf ? std::vector<std::string>{"vmaf", "psnr"}
                                              : std::vector<std::string>{"psnr", "ssim"}) {
                const FrameMetricResult* expected = from_container.metric_results.frame(key);
                const FrameMetricResult* from_stream = from_raw.metric_results.frame(key);
                ASSERT_NE(expected, nullptr) << key;
                ASSERT_NE(from_stream, nullptr) << key;
                EXPECT_EQ(from_stream->values.size(), frames) << key;
                EXPECT_EQ(from_stream->values, expected->values) << key;
            }
            // Straight to FFmpeg: no attempt at decoding in the app first.
            EXPECT_TRUE(std::none_of(raw_statuses.begin(), raw_statuses.end(), [](const std::string& status) {
                return status.find("in the app failed") != std::string::npos;
            }));
        }
        // The black bars sampled from bytes partway through where the stream
        // has random access pictures there (the parameter sets put before
        // them), and else in one pass from its start.
        const std::lock_guard lock(lines_mutex);
        const bool one_pass = std::any_of(lines.begin(), lines.end(), [&name](const std::string& line) {
            return line.starts_with("Black bars in " + name + ".hevc: sampling in one pass");
        });
        EXPECT_EQ(one_pass, one_keyframe);
    }
}

// A raw YUV file (pictures and nothing else), read at the size, pixel
// format and rate the command line gives, scored as the same pictures in a
// container: the same frames, black bars and scores.
TEST(RawStream, AYuvFileScoresAsTheSamePicturesInAContainer) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    // Letterboxed, 640x280 pictures in a 640x360 frame.
    const VideoInfo source = probe_video(
        test::make_clip(*found, folder / "source.mkv", "libx264", "yuv420p", 4.0, {"-crf", "10", "-vf",
                        "pad=640:360:0:40"}, "640x280"),
        found->ffprobe);
    const std::filesystem::path contained_path = test::make_clip(
        *found, folder / "encode.mkv", "libx264", "yuv420p10le", 4.0, {"-crf", "30", "-vf", "pad=640:360:0:40"},
        "640x280");
    test::run_ffmpeg(*found, {"-i", platform::path_text(contained_path), "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
                              platform::path_text(folder / "encode.yuv")});
    const VideoInfo contained = probe_video(contained_path, found->ffprobe);
    const VideoInfo raw = probe_raw_yuv(folder / "encode.yuv", "640x360", "yuv420p10le", 24000.0 / 1001, found->ffprobe);
    const auto frames = static_cast<std::size_t>(std::llround(4.0 * 24000 / 1001));
    ASSERT_TRUE(raw.is_raw_stream());
    EXPECT_EQ(raw.nb_frames, static_cast<long long>(frames));
    EXPECT_EQ(raw.input_options(), (std::vector<std::string>{"-f", "rawvideo", "-video_size", "640x360",
                                                             "-pixel_format", "yuv420p10le", "-framerate",
                                                             "23.976023976023978"}));
    const RunEnvironment environment{found->ffmpeg, found->version, found->ffprobe,
                                     DecodingCapabilities::detect(found->ffmpeg), platform::logical_cores()};
    const auto score = [&](const VideoInfo& test_video) {
        VmafOptions options;
        options.compute_vmaf = false;
        options.extra_features = {"name=psnr", "name=float_ssim"};
        options.crop_mode = CropMode::automatic;
        return run_vmaf(environment, source, test_video, options, nullptr, {}, CancelToken());
    };
    const ComparisonResult from_container = score(contained);
    const ComparisonResult from_raw = score(raw);
    ASSERT_TRUE(from_container.distorted_crop.has_value());
    EXPECT_EQ(from_container.distorted_crop->h, 280);
    EXPECT_EQ(from_raw.distorted_crop, from_container.distorted_crop);
    for (const std::string key : {"psnr", "ssim"}) {
        const FrameMetricResult* expected = from_container.metric_results.frame(key);
        const FrameMetricResult* from_pictures = from_raw.metric_results.frame(key);
        ASSERT_NE(expected, nullptr) << key;
        ASSERT_NE(from_pictures, nullptr) << key;
        EXPECT_EQ(from_pictures->values.size(), frames) << key;
        EXPECT_EQ(from_pictures->values, expected->values) << key;
    }
}

}  // namespace
}  // namespace vml
