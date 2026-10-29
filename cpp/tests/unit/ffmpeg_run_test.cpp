// FFmpeg's filter graph and command, and the reading of its logs, held to
// the Python app's own (runs.json, from vmaf_runner.py).
#include "vml/ffmpeg_run.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "reference.h"
#include "vml/frame_sync.h"
#include "vml/gpu_vmaf.h"
#include "vml/platform/system.h"
#include "vml/size_changes.h"

namespace vml {
namespace {

using Json = nlohmann::json;

void set_pairing(const std::string& value) {
#ifdef _WIN32
    _putenv_s(std::string(pairing_variable).c_str(), value.c_str());
#else
    setenv(std::string(pairing_variable).c_str(), value.c_str(), 1);
#endif
}

// A video at the root of the drive: a command names its videos by their full
// paths (input_path), which for a name that does not exist checks every
// folder above it, so a relative name would make the tests as slow as the
// folder they run in is deep.
VideoInfo video(const char* name, int width, int height, const char* pix_fmt, double fps) {
    VideoInfo info;
    info.path = std::filesystem::path("/") / name;
    info.width = width;
    info.height = height;
    info.pix_fmt = pix_fmt;
    info.fps = fps;
    info.duration = 10.0;
    info.codec_name = "hevc";
    return info;
}

// The cases' parts, as runs.json names them.
std::pair<VideoInfo, VideoInfo> pair_named(const std::string& name) {
    if (name == "4k against 1080p") {
        return {video("source.mp4", 3840, 2160, "yuv420p10le", 120.0),
                video("test.mkv", 1920, 1080, "yuv420p10le", 120.0)};
    }
    return {video("source.mkv", 1920, 1080, "yuv420p", 24000.0 / 1001.0),
            video("test.mkv", 1920, 1080, "yuv420p10le", 24000.0 / 1001.0)};
}

std::pair<std::optional<CropBox>, std::optional<CropBox>> crops_named(const std::string& name) {
    if (name == "letterbox") {
        return {CropBox{3840, 1608, 0, 276}, CropBox{1920, 804, 0, 138}};
    }
    return {std::nullopt, std::nullopt};
}

HardwareDecoding plan_named(const std::string& name) {
    if (name == "cuda") {
        return {"cuda", "cuda"};
    }
    if (name == "cuda source") {
        return {"cuda", std::nullopt};
    }
    if (name == "qsv") {
        return {"qsv", "qsv"};
    }
    if (name == "d3d11va test") {
        return {std::nullopt, "d3d11va"};
    }
    return {};
}

VmafOptions options_named(const std::string& choice, const std::string& setting) {
    VmafOptions options;
    options.compute_vmaf = choice != "psnr, xpsnr" && choice != "xpsnr" && choice != "neg alone";
    options.compute_vmaf_neg = choice == "vmaf, neg" || choice == "all" || choice == "neg alone";
    options.compute_xpsnr = choice == "vmaf, xpsnr" || choice == "all" || choice == "psnr, xpsnr" || choice == "xpsnr";
    if (choice == "vmaf, v1") {
        options.compute_vmaf_v1 = true;
        options.model_v1 = "path=models/vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json";  // only its file name is used
    }
    if (choice == "vmaf, psnr, ssim" || choice == "all") {
        options.extra_features = {"name=psnr", "name=float_ssim"};
    } else if (choice == "psnr, xpsnr") {
        options.extra_features = {"name=psnr"};
    }
    if (setting == "plain") {
        options.n_threads = 12;
    } else if (setting == "subsample, lanczos, upscale") {
        options.n_threads = 8;
        options.n_subsample = 3;
        options.scale_algorithm = "lanczos";
        options.scale_direction = ScaleDirection::distorted_to_source;
    } else if (setting == "limit") {
        options.n_threads = 24;
        options.duration_limit = 1.0;
    } else {
        options.n_threads = 0;  // "Auto": the PC's cores
    }
    return options;
}

// A resolution test: its one input, the source, split into an untouched
// reference and a test video scaled down to the target and back up.
TEST(FfmpegRun, BuildsThePythonAppsResolutionTestCommands) {
    const Json reference = test::load_reference("runs.json");
    ASSERT_FALSE(reference["resample_commands"].empty());
    const std::map<std::string, ResampleTarget> targets{
        {"1080p", {1920, "1080p"}}, {"720p", {1280, "720p"}}, {"480p", {854, "480p"}}};
    for (const Json& item : reference["resample_commands"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        set_pairing(item["pairing"].get<std::string>());
        const VideoInfo source = pair_named(item["pair"].get<std::string>()).first;
        const std::optional<CropBox> source_crop = crops_named(item["crop"].get<std::string>()).first;
        const std::optional<std::string> hwaccel = plan_named(item["plan"].get<std::string>()).source;
        VmafOptions options = options_named(item["choice"].get<std::string>(), item["setting"].get<std::string>());
        options.resample_test = targets.at(item["target"].get<std::string>());
        const std::optional<std::string> xpsnr_log =
            options.compute_xpsnr ? std::optional<std::string>("xpsnr_log.txt") : std::nullopt;
        const std::string graph =
            build_resample_filtergraph(source, options, source_crop, hwaccel, "vmaf_log.json",
                                       item["model"].get<std::string>(), xpsnr_log, item["cores"].get<unsigned>());
        std::vector<std::string> command =
            build_resample_command("<ffmpeg>", source.path, graph, hwaccel, options.duration_limit);
        for (std::string& part : command) {
            if (part == platform::path_text(std::filesystem::absolute(source.path))) {
                part = "<source>";  // an absolute path, which differs between PCs
            }
        }
        EXPECT_EQ(command, item["command"].get<std::vector<std::string>>());
    }
    set_pairing("");
}

// VMAF on the GPU: each video's own FFmpeg, decoding it, giving its frames
// as the app pairs them, and writing them into the app's pipe.
TEST(FfmpegRun, BuildsThePythonAppsCommandsForVmafOnTheGpu) {
    const Json reference = test::load_reference("runs.json");
    ASSERT_FALSE(reference["gpu_commands"].empty());
    for (const Json& item : reference["gpu_commands"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        set_pairing(item["pairing"].get<std::string>());
        const auto [source, test] = pair_named(item["pair"].get<std::string>());
        const auto [source_crop, test_crop] = crops_named(item["crop"].get<std::string>());
        const HardwareDecoding plan = plan_named(item["plan"].get<std::string>());
        const VmafOptions options = options_named("vmaf, neg", item["setting"].get<std::string>());
        const double limit = test::reference_number(item["limit"]).value();
        std::vector<std::vector<std::string>> commands = build_stream_commands(
            "<ffmpeg>", test.path, source.path,
            build_stream_graphs(source, test, options, source_crop, test_crop, plan), plan,
            by_timestamp()
                ? timed_gpu_output_args(limit, "<test pipe>", "<test pipe>-times", "<source pipe>", "<source pipe>-times")
                : gpu_output_args(limit, "<test pipe>", "<source pipe>"));
        for (std::vector<std::string>& command : commands) {
            for (std::string& part : command) {
                if (part == platform::path_text(std::filesystem::absolute(test.path))) {
                    part = "<test>";
                } else if (part == platform::path_text(std::filesystem::absolute(source.path))) {
                    part = "<source>";
                }
            }
        }
        EXPECT_EQ(commands, item["commands"].get<std::vector<std::vector<std::string>>>());
    }
    set_pairing("");
}

// A Mac's GPU decoder hands its frames over in system memory itself, so a
// run with several graphs decodes each picture once and leaves the download
// to it; one graph keeps the frames on the GPU for its own hwdownload.
TEST(FfmpegRun, AMacsGpuDecoderDownloadsItselfForSeveralGraphs) {
    const HardwareDecoding mac{"videotoolbox", "videotoolbox"};
    const std::vector<std::string> several =
        build_ffmpeg_command("ffmpeg", "test.mp4", "ref.mp4", "[0:v]null[a]\n[1:v]null[b]", mac, 0);
    EXPECT_EQ(std::count(several.begin(), several.end(), "-hwaccel_output_format"), 0);
    EXPECT_EQ(std::count(several.begin(), several.end(), "videotoolbox"), 2);
    const std::vector<std::string> one = build_ffmpeg_command("ffmpeg", "test.mp4", "ref.mp4", "[0:v]null", mac, 0);
    EXPECT_EQ(std::count(one.begin(), one.end(), "videotoolbox_vld"), 2);
}

TEST(FfmpegRun, ReadsLibvmafsLogAsThePythonAppDoes) {
    const Json reference = test::load_reference("runs.json");
    const std::filesystem::path folder = std::filesystem::temp_directory_path();
    const double fps = *test::reference_number(reference["log_fps"]);
    ASSERT_FALSE(reference["logs"].empty());
    for (const Json& item : reference["logs"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        const std::filesystem::path log = folder / "vml_test_vmaf_log.json";
        std::ofstream(log, std::ios::binary) << item["log"].get<std::string>();
        std::optional<std::filesystem::path> xpsnr_log;
        if (!item["xpsnr_log"].is_null()) {
            xpsnr_log = folder / "vml_test_xpsnr_log.txt";
            std::ofstream(*xpsnr_log, std::ios::binary) << item["xpsnr_log"].get<std::string>();
        }
        const FrameScores scores = parse_libvmaf_log(log, fps, xpsnr_log);
        EXPECT_EQ(scores.frame, item["frames"].get<std::vector<int>>());
        EXPECT_EQ(scores.time, test::reference_numbers(item["times"]));
        ASSERT_EQ(scores.metrics.size(), item["metrics"].size());
        for (const auto& [key, values] : item["metrics"].items()) {
            ASSERT_TRUE(scores.metrics.contains(key)) << key;
            const std::vector<double> expected = test::reference_numbers(values);
            const std::vector<float>& actual = scores.metrics.at(key);
            ASSERT_EQ(actual.size(), expected.size()) << key;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_TRUE(test::same_number(expected[i], static_cast<double>(actual[i]))) << key << " " << i;
            }
        }
    }
}

TEST(FfmpegRun, ARunCutShortIsRefused) {
    EXPECT_EQ(short_comparison(1000, 990, 24.0, 1), std::nullopt);
    EXPECT_EQ(short_comparison(1000, 500, 24.0, 1).value_or(""),
              "Only 500 of the 1000 frames expected could be compared: one of the videos ends after 0:00:20.8, "
              "though its file says it is longer. It may have been cut short: an encode that stopped early, or a "
              "copy that did not finish.");
}

// FFmpeg 9.0.1's word that it rebuilt its filters as a video's size changed,
// and its error output at watched_log_level cut back to the lines
// "-loglevel error" shows for the same failure, without their levels; a line
// with no level goes with the message before it, and is kept before any.
TEST(SizeChanges, ReadsFfmpegsLines) {
    EXPECT_TRUE(filters_rebuilt("[vf#0:0 @ 0000017c52a6b5c0] [info] Reconfiguring filter graph because video "
                                "parameters changed to yuv420p10le(tv, unknown), 1280x720, unspecified alpha"));
    EXPECT_FALSE(filters_rebuilt("[info] Stream mapping:"));
    const std::string written =
        "no level first\n"
        "[info] Input #0, lavfi, from 'testsrc2=size=64x64:rate=1:duration=1':\n"
        "[info]     description     : a value of two\n"
        "                     lines\n"
        "[info]   Duration: N/A, start: 0.000000, bitrate: N/A\n"
        "[in#0 @ 000002755dd61c00] [error] Error opening input: No such file or directory\n"
        "[swscaler @ 000002755dd62000] [warning] deprecated pixel format used, make sure you did set range correctly\n"
        "[error] Error opening input file E:\\nonexistent.mkv.\n"
        "[fatal] Error opening input files: No such file or directory\n"
        "no level\n"
        "[info] frame=    1 fps=0.0 q=-0.0 Lsize=N/A time=00:00:01.00 bitrate=N/A speed= 235x\n";
    EXPECT_EQ(error_lines_of(written), "no level first\n"
                                       "[in#0 @ 000002755dd61c00] Error opening input: No such file or directory\n"
                                       "Error opening input file E:\\nonexistent.mkv.\n"
                                       "Error opening input files: No such file or directory\n"
                                       "no level\n");
}

}  // namespace
}  // namespace vml
