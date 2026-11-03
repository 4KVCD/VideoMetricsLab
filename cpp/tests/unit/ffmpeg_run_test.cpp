// FFmpeg's filter graph and command, and the reading of its logs, held to
// the Python app's own (runs.json, from vmaf_runner.py).
#include "vml/ffmpeg_run.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "reference.h"
#include "vml/crop_detect.h"
#include "vml/gpu_frames.h"
#include "vml/gpu_vmaf.h"
#include "vml/platform/system.h"
#include "vml/size_changes.h"

namespace vml {
namespace {

using Json = nlohmann::json;

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
        options.range_end = 1;  // the first second
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
            build_resample_command("<ffmpeg>", source.path, graph, hwaccel);
        for (std::string& part : command) {
            if (part == platform::path_text(std::filesystem::absolute(source.path))) {
                part = "<source>";  // an absolute path, which differs between PCs
            }
        }
        EXPECT_EQ(command, item["command"].get<std::vector<std::string>>());
    }
}

// VMAF on the GPU: each video's own FFmpeg, decoding it, giving its frames
// as the app pairs them, and writing them into the app's pipe.
TEST(FfmpegRun, BuildsThePythonAppsCommandsForVmafOnTheGpu) {
    const Json reference = test::load_reference("runs.json");
    ASSERT_FALSE(reference["gpu_commands"].empty());
    for (const Json& item : reference["gpu_commands"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        const auto [source, test] = pair_named(item["pair"].get<std::string>());
        const auto [source_crop, test_crop] = crops_named(item["crop"].get<std::string>());
        const HardwareDecoding plan = plan_named(item["plan"].get<std::string>());
        const VmafOptions options = options_named("vmaf, neg", item["setting"].get<std::string>());
        std::vector<std::vector<std::string>> commands = build_stream_commands(
            "<ffmpeg>", test.path, source.path,
            build_stream_graphs(source, test, options, source_crop, test_crop, plan), plan,
            gpu_output_args("<test pipe>", "<source pipe>"));
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
}

// A range late in the videos: both videos' chains first keep the frames from
// a second before it (VMAF's motion looks back) to a frame past it, counted
// as decoded, and the outputs have no duration of their own. At 120 fps,
// seconds 5 to 6 are frames 600 to 719.
TEST(FfmpegRun, ARangesFramesAreKeptAtTheHeadOfBothChains) {
    const auto [source, test] = pair_named("4k against 1080p");
    VmafOptions options = options_named("vmaf", "plain");
    options.range_start = 5;
    options.range_end = 6;
    const std::string graph = build_filtergraph(source, test, options, std::nullopt, std::nullopt, HardwareDecoding{},
                                                "vmaf_log.json", "version=vmaf_v0.6.1", std::nullopt, 12);
    EXPECT_NE(graph.find("[0:V:0]trim=start_frame=480:end_frame=721,"), std::string::npos) << graph;
    EXPECT_NE(graph.find("[1:V:0]trim=start_frame=480:end_frame=721,"), std::string::npos) << graph;
    const std::vector<std::string> command =
        build_ffmpeg_command("ffmpeg", test.path, source.path, graph, HardwareDecoding{});
    EXPECT_EQ(std::count(command.begin(), command.end(), "-t"), 0);
}

// A Mac's GPU decoder hands its frames over in system memory itself, so a
// run with several graphs decodes each picture once and leaves the download
// to it; one graph keeps the frames on the GPU for its own hwdownload.
TEST(FfmpegRun, AMacsGpuDecoderDownloadsItselfForSeveralGraphs) {
    const HardwareDecoding mac{"videotoolbox", "videotoolbox"};
    const std::vector<std::string> several =
        build_ffmpeg_command("ffmpeg", "test.mp4", "ref.mp4", "[0:v]null[a]\n[1:v]null[b]", mac);
    EXPECT_EQ(std::count(several.begin(), several.end(), "-hwaccel_output_format"), 0);
    EXPECT_EQ(std::count(several.begin(), several.end(), "videotoolbox"), 2);
    const std::vector<std::string> one = build_ffmpeg_command("ffmpeg", "test.mp4", "ref.mp4", "[0:v]null", mac);
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
    EXPECT_EQ(short_comparison(1000, 990, 0, 24.0, 1), std::nullopt);
    EXPECT_EQ(short_comparison(1000, 500, 0, 24.0, 1).value_or(""),
              "Only 500 of the 1000 frames expected could be compared: one of the videos ends after 0:00:20.8, "
              "though its file says it is longer. It may have been cut short: an encode that stopped early, or a "
              "copy that did not finish.");
    // In a range from 1:00, the time is the video's, not the range's.
    EXPECT_NE(short_comparison(1000, 500, 1440, 24.0, 1).value_or("").find("ends after 0:01:20.8,"),
              std::string::npos);
}

// An FFmpeg whose write into the app's frame pipe Windows refused (its lines
// as FFmpeg 9.0.1 wrote them under load), told from FFmpeg failing otherwise
// with the same code, or with the muxer's lines alone, and from a crash.
TEST(FfmpegRun, TellsARefusedFrameWriteFromOtherFailures) {
    const std::string refused =
        "[vost#0:0/wrapped_avframe @ 000002bb317dc040] Error submitting a packet to the muxer: Invalid argument\n"
        "    Last message repeated 1 times\n"
        "[out#0/yuv4mpegpipe @ 000002bb246d0380] Error muxing a packet\n"
        "[out#0/yuv4mpegpipe @ 000002bb246d0380] Task finished with error code: -22 (Invalid argument)\n"
        "[out#0/yuv4mpegpipe @ 000002bb246d0380] Terminating thread with return code -22 (Invalid argument)\n"
        "[out#0/yuv4mpegpipe @ 000002bb246d0380] Error writing trailer: Invalid argument\n"
        "[out#0/yuv4mpegpipe @ 000002bb246d0380] Error closing file: Invalid argument\n";
    EXPECT_TRUE(refused_frame_write(-22, refused));
    EXPECT_FALSE(refused_frame_write(static_cast<int>(0xC0000005), refused));
    EXPECT_FALSE(refused_frame_write(1, refused));
    EXPECT_FALSE(refused_frame_write(
        -22, "[vost#0:0/wrapped_avframe @ 000002bb317dc040] Error submitting a packet to the muxer: Invalid argument\n"
             "[out#0/yuv4mpegpipe @ 000002bb246d0380] Error muxing a packet\n"
             "[out#0/yuv4mpegpipe @ 000002bb246d0380] Task finished with error code: -22 (Invalid argument)\n"));
    EXPECT_FALSE(refused_frame_write(-22, "[Parsed_scale_0 @ 0000021c] Impossible to convert between the formats "
                                          "supported by the filter 'graph 0 input from stream 0:0' and the filter "
                                          "'auto_scale_0'\n"));
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

// A GPU decoder's failure trips the run without a note of its own (each
// process's would name its own error); the note is the re-run's, one text for
// the whole run, and goes between processes with its kind.
TEST(FfmpegRun, AFailedGpuDecoderTripsTheRunAndOnlyItsReRunIsNoted) {
    trust_gpu_decoding();
    struct TrustedAgain {  // the process-wide latch left as it was, however the test ends
        ~TrustedAgain() { trust_gpu_decoding(); }
    } trusted_again;
    std::vector<RunStatus> sent;
    const StatusCallback keep = [&sent](const RunStatus& status) { sent.push_back(status); };
    distrust_failed_decoder(GpuDecodeFailedError("a packet has no timestamp"), keep);
    distrust_failed_decoder(GpuDecodeFailedError("VVdeC failed", std::string(software_decoder), false), keep);
    EXPECT_TRUE(sent.empty());
    EXPECT_FALSE(gpu_decoding_distrusted());
    distrust_failed_decoder(GpuDecodeFailedError("the GPU's decoder found an error in the video", "nvidia", true),
                            keep);
    ASSERT_EQ(sent.size(), 1u);
    EXPECT_EQ(sent[0].kind, StatusKind::gpu_decoder_failed);
    EXPECT_FALSE(is_gpu_fallback(sent[0].kind));
    EXPECT_EQ(gpu_decoding_distrusted(), "the GPU's decoder found an error in the video");

    const RunStatus note = gpu_decoding_off_status(*gpu_decoding_distrusted());
    EXPECT_TRUE(is_gpu_fallback(note.kind));
    EXPECT_EQ(fallback_note(note.text), "The GPU's video decoding failed (the GPU's decoder found an error in the "
                                        "video); the videos were decoded on the CPU");
    for (const RunStatus& status : {sent[0], note}) {
        const RunStatus back = status_from_json(status_json(status));
        EXPECT_EQ(back.kind, status.kind);
        EXPECT_EQ(back.text, status.text);
    }
}

// A child's statuses taken in by the run's process: vvdec-vulkan opened is noted for the pass, vvdec-vulkan turned
// off there is off here too and in the next child's options (the other GPU decoders stay trusted), and both are
// passed on; a new run turns it on again.
TEST(FfmpegRun, VvdecVulkanTurnedOffInAChildIsOffForTheRun) {
    trust_gpu_decoding();
    struct TrustedAgain {  // the process-wide latches left as they were, however the test ends
        ~TrustedAgain() { trust_gpu_decoding(); }
    } trusted_again;
    std::vector<RunStatus> sent;
    const StatusCallback keep = [&sent](const RunStatus& status) { sent.push_back(status); };
    VmafOptions options;
    EXPECT_FALSE(vvdec_wanted(options.gpu_decode, options.vvc_on_gpu));  // the CPU's VVdeC by default
    options.vvc_on_gpu = true;
    EXPECT_TRUE(vvdec_wanted(options.gpu_decode, options.vvc_on_gpu));
    EXPECT_FALSE(vvdec_wanted(false, options.vvc_on_gpu));
    bool vvdec_used = false;
    take_child_status({"opened", StatusKind::vvdec_used}, keep, vvdec_used);
    EXPECT_TRUE(vvdec_used);
    EXPECT_TRUE(child_options(options).vvc_on_gpu);
    take_child_status(status_from_json(status_json({"its device was lost", StatusKind::vvdec_off})), keep, vvdec_used);
    EXPECT_TRUE(vvdec_turned_off());
    EXPECT_FALSE(vvdec_wanted(options.gpu_decode, options.vvc_on_gpu));
    EXPECT_FALSE(child_options(options).vvc_on_gpu);
    EXPECT_FALSE(gpu_decoding_distrusted());
    EXPECT_EQ(sent.size(), 2u);
    trust_gpu_decoding();
    EXPECT_FALSE(vvdec_turned_off());
}

// Forgetting a video's black bars takes its answer out of the store and keeps
// the store's others (it is read first); with none for it, nothing is
// written.
TEST(FfmpegRun, ForgettingAVideosBlackBarsKeepsTheOthers) {
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() / ("vml_crop_store_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(folder);
    const std::filesystem::path store = folder / "black_bars.json";
    const std::filesystem::path video = folder / "video.mkv";
    std::ofstream(video, std::ios::binary) << "not a video";
    // The keys as crop_detect.cpp's answer_key makes them: the file, its size and time, the duration sampled and
    // the dark limit (32 / 255), rounded
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(std::filesystem::absolute(video));
    const auto size = static_cast<long long>(std::filesystem::file_size(resolved));
    const Json ours = {{platform::path_text(resolved), size, *platform::modified_ns(resolved), 10.0, 0.12549},
                       {1920, 800, 0, 140}};
    const Json other = {{"E:\\other.mkv", 10, 20, 30.0, 0.12549}, {1920, 816, 0, 132}};
    std::ofstream(store, std::ios::binary) << Json{{"version", 4}, {"boxes", {other, ours}}}.dump();
    const auto read_store = [&store] {
        std::ifstream file(store, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    };
    forget_crop_answers();  // nothing of other tests' in this process's answers
    set_crop_store([store] { return store; });
    VideoInfo info;
    info.path = video;
    info.duration = 10.0;
    forget_crop_answer(info);
    const std::string after = read_store();
    EXPECT_EQ(Json::parse(after)["boxes"], Json::array({other}));
    forget_crop_answer(info);  // none left for it
    EXPECT_EQ(read_store(), after);
    set_crop_store({});
    forget_crop_answers();
    std::filesystem::remove_all(folder);
}

}  // namespace
}  // namespace vml
