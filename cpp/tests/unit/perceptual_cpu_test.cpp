// SSIMULACRA2 and Butteraugli on the CPU (perceptual_cpu.h), held to the
// Python app's (perceptual_cpu.json, from perceptual_cpu.py and colour.py):
// the filter graphs and FFmpeg commands that make the pictures, the scores
// read from the tools' output, how many pairs are scored at once,
// Butteraugli's 3-norm, and the pictures' colour chunks. Where FFmpeg runs:
// the pictures are each video's frame N, a task scores every pair with
// stand-ins for the tools, and Cancel ends the tools at once. And each
// metric goes where the Python app sends it (run_perceptual_task).
#include "vml/perceptual_cpu.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <thread>

#include "reference.h"
#include "test_videos.h"
#include "vml/colour.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/frame_sync.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/vship_run.h"

namespace vml {
namespace {

using Json = nlohmann::json;

const Json& reference() {
    static const Json loaded = test::load_reference("perceptual_cpu.json");
    return loaded;
}

VideoInfo video_from(const Json& info) {
    VideoInfo video;
    video.path = "video.mkv";
    video.width = info["width"].get<int>();
    video.height = info["height"].get<int>();
    video.fps = *test::reference_number(info["fps"]);
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

std::optional<CropBox> crop_from(const Json& crop) {
    if (crop.is_null()) {
        return std::nullopt;
    }
    return CropBox{crop[0].get<int>(), crop[1].get<int>(), crop[2].get<int>(), crop[3].get<int>()};
}

std::string bytes_from_hex(const std::string& hex) {
    std::string bytes;
    for (std::size_t at = 0; at + 1 < hex.size(); at += 2) {
        bytes.push_back(static_cast<char>(std::stoi(hex.substr(at, 2), nullptr, 16)));
    }
    return bytes;
}

std::string hex_of(const std::string& bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string hex;
    for (const char byte : bytes) {
        hex.push_back(digits[static_cast<unsigned char>(byte) >> 4]);
        hex.push_back(digits[static_cast<unsigned char>(byte) & 0xF]);
    }
    return hex;
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream(path, std::ios::binary) << bytes;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// The name FFmpeg gives pair `number`'s pictures after their side.
std::string picture_name(int number) {
    const std::string digits = std::to_string(number);
    return std::string(8 - digits.size(), '0') + digits + ".png";
}

// A video as the Python app's reference cases make one by default.
VideoInfo plain_video(const std::filesystem::path& path) {
    VideoInfo video;
    video.path = path;
    video.width = 1920;
    video.height = 1080;
    video.fps = 24.0;
    video.duration = 10.0;
    video.nb_frames = 240;
    video.codec_name = "hevc";
    video.pix_fmt = "yuv420p";
    return video;
}

// The filter graph and the command FFmpeg is given, by frame number and by
// timestamp.
TEST(PerceptualCpu, MakesThePicturesAsThePythonApp) {
    for (const Json& expected : reference()["graphs"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        const test::ScopedVariable pairing(std::string(pairing_variable),
                                           expected["by_time"].get<bool>() ? "timestamp" : "");
        VmafOptions options;
        options.scale_direction = expected["to_source"].get<bool>() ? ScaleDirection::distorted_to_source
                                                                     : ScaleDirection::source_to_distorted;
        options.scale_algorithm = expected["algorithm"].get<std::string>();
        EXPECT_EQ(image_filtergraph(video_from(expected["source"]), video_from(expected["test"]), options,
                                    crop_from(expected["source_crop"]), crop_from(expected["test_crop"]),
                                    expected["step"].get<int>()),
                  expected["graph"].get<std::string>());
    }
    const std::filesystem::path folder = std::filesystem::temp_directory_path() / "pictures";
    const std::string separator(1, static_cast<char>(std::filesystem::path::preferred_separator));
    for (const Json& expected : reference()["commands"]) {
        SCOPED_TRACE(expected.dump());
        VmafOptions options;
        options.duration_limit = *test::reference_number(expected["limit"]);
        std::vector<std::string> command;
        for (std::string argument : expected["command"].get<std::vector<std::string>>()) {
            if (argument == "<source>" || argument == "<test>") {
                argument = input_path(argument == "<source>" ? "source.mkv" : "test.mkv");
            } else if (argument.starts_with("<folder>/")) {
                argument = platform::path_text(folder) + separator + argument.substr(9);
            }
            command.push_back(argument);
        }
        EXPECT_EQ(png_pairs_command("ffmpeg", plain_video("source.mkv"), plain_video("test.mkv"), options,
                                    std::nullopt, std::nullopt, expected["step"].get<int>(), folder),
                  command);
    }
}

// The score a tool printed, or the same refusal; and how many pairs are
// scored at once for a size, the metrics, the CPU's tasks, cores and memory.
TEST(PerceptualCpu, ReadsScoresAndSharesTheCpuAsThePythonApp) {
    for (const Json& expected : reference()["scores"]) {
        const std::string text = expected["text"].get<std::string>();
        SCOPED_TRACE(text);
        if (expected.contains("error")) {
            try {
                parse_score("ssimulacra2", text);
                ADD_FAILURE() << "no error";
            } catch (const PerceptualRunError& error) {
                EXPECT_EQ(std::string(error.what()), expected["error"].get<std::string>());
            }
            continue;
        }
        const double wanted = *test::reference_number(expected["score"]);
        const double score = parse_score("ssimulacra2", text);
        EXPECT_EQ(score, wanted);
        EXPECT_EQ(std::signbit(score), std::signbit(wanted));
    }
    for (const Json& expected : reference()["workers"]) {
        SCOPED_TRACE(expected.dump());
        EXPECT_EQ(scoring_workers(expected["pixels"].get<long long>(),
                                  expected["metrics"].get<std::vector<std::string>>(),
                                  expected["concurrent"].get<int>(), expected["cores"].get<unsigned>(),
                                  expected["free_memory"].get<unsigned long long>()),
                  expected["workers"].get<int>());
    }
}

// Butteraugli's 3-norm of a distortion map is numpy's, to the last digit on
// Windows; a colour map and a file that is not a map are refused as the
// Python app refuses them.
TEST(PerceptualCpu, TakesButteraugli3NormAsNumpy) {
    const test::Folder folder;
    for (const Json& expected : reference()["norms"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        const std::filesystem::path map = folder / "map.pfm";
        write_file(map, bytes_from_hex(expected["bytes"].get<std::string>()));
        if (expected.contains("norm")) {
            EXPECT_TRUE(test::same_number(*test::reference_number(expected["norm"]), butteraugli_norm3(map)));
            continue;
        }
        const std::string error = expected["error"].get<std::string>();
        try {
            butteraugli_norm3(map);
            ADD_FAILURE() << "no error";
        } catch (const PerceptualRunError& refused) {
            EXPECT_EQ(std::string(refused.what()), error);
        } catch (const std::runtime_error&) {
            EXPECT_EQ(error, "unreadable");
        }
    }
}

// The chunks that tell libjxl's tools how to read a picture's values, and
// FFmpeg's pictures with them in place of their own.
TEST(PerceptualCpu, DescribesThePicturesColoursAsThePythonApp) {
    for (const Json& expected : reference()["chunks"]) {
        const VideoColour colour{1, expected["transfer"].get<int>(), expected["primaries"].get<int>(), false};
        EXPECT_EQ(hex_of(png_colour_chunks(colour)), expected["chunks"].get<std::string>()) << expected.dump();
    }
    const test::Folder folder;
    for (const Json& expected : reference()["pictures"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        const std::filesystem::path picture = folder / "picture.png";
        write_file(picture, bytes_from_hex(expected["bytes"].get<std::string>()));
        describe_png(picture, VideoColour{1, expected["transfer"].get<int>(), expected["primaries"].get<int>(), false});
        EXPECT_EQ(hex_of(read_file(picture)), expected["described"].get<std::string>());
    }
}

// ---------------------------------------------------------------- with FFmpeg

// Two videos whose frame N is the same picture (a grey of its own): the
// source at 20 fps, the test video at 23.976 with a millisecond of jitter.
std::pair<VideoInfo, VideoInfo> same_pictures(const test::Tools& found, const test::Folder& folder) {
    const std::string pictures = "geq=lum='16+4*N':cb=128:cr=128,format=yuv420p";
    test::run_ffmpeg(found, {"-f", "lavfi", "-i", "nullsrc=s=64x48:r=20:d=3", "-vf", pictures, "-c:v", "ffv1",
                             platform::path_text(folder / "source.mkv")});
    test::run_ffmpeg(found, {"-f", "lavfi", "-i", "nullsrc=s=64x48:r=24000/1001:d=2", "-vf",
                             pictures + ",settb=1/1000,setpts='round(N*1001/24)+mod(N\\,2)'", "-fps_mode",
                             "passthrough", "-enc_time_base", "filter", "-c:v", "ffv1",
                             platform::path_text(folder / "test.mkv")});
    return {probe_video(folder / "source.mkv", found.ffprobe), probe_video(folder / "test.mkv", found.ffprobe)};
}

// Each pair is the two videos' frame N (frame_sync), every step-th, and the
// 1.5 s limit keeps the same frames of both: the test video's 36, not the
// source's 30.
TEST(PerceptualCpu, ThePicturesAreEachVideosFrameNCutAtTheLimit) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    const auto [source, test_video] = same_pictures(*found, folder);
    ASSERT_EQ(frame_rate(test_video), (Rational{24000, 1001}));
    ASSERT_EQ(frames_within("1.500", frame_rate(test_video)), 36);
    for (const int step : {1, 3}) {
        SCOPED_TRACE(step);
        const std::filesystem::path pictures = folder / ("pictures-" + std::to_string(step));
        std::filesystem::create_directory(pictures);
        VmafOptions options;
        options.duration_limit = 1.5;
        const platform::ProcessOutput output = platform::run_program(
            png_pairs_command(found->ffmpeg, source, test_video, options, std::nullopt, std::nullopt, step, pictures),
            std::chrono::minutes(2));
        ASSERT_EQ(output.exit_code, 0) << output.standard_error;
        std::vector<std::string> seen;
        for (int number = 1;; ++number) {
            const std::string name = picture_name(number);
            const std::filesystem::path reference_picture = pictures / ("reference-" + name);
            const std::filesystem::path test_picture = pictures / ("test-" + name);
            if (!std::filesystem::exists(reference_picture) || !std::filesystem::exists(test_picture)) {
                break;
            }
            EXPECT_EQ(read_file(reference_picture), read_file(test_picture)) << name;
            seen.push_back(read_file(reference_picture));
        }
        EXPECT_EQ(seen.size(), static_cast<std::size_t>((36 + step - 1) / step));
        std::sort(seen.begin(), seen.end());
        EXPECT_EQ(std::unique(seen.begin(), seen.end()), seen.end());  // each pair a different picture
    }
}

// With VML_FRAME_PAIRING=timestamp, as before: the test video is the source
// with its sixth frame dropped, the others' times kept. Paired by position,
// every frame after the gap was compared with the source's next one; by
// time, as libvmaf pairs them, each frame is compared with itself.
TEST(PerceptualCpu, PairedByTimestampAFrameDroppedFromTheTestVideoPairsTheRestByTime) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::ScopedVariable pairing(std::string(pairing_variable), "timestamp");
    const test::Folder folder;
    test::run_ffmpeg(*found, {"-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24:duration=1", "-vf",
                              "format=yuv420p10le", "-c:v", "ffv1", platform::path_text(folder / "source.mkv")});
    test::run_ffmpeg(*found, {"-i", platform::path_text(folder / "source.mkv"), "-vf", "select='not(eq(n,5))'",
                              "-fps_mode", "passthrough", "-c:v", "ffv1", platform::path_text(folder / "test.mkv")});
    const VideoInfo source = probe_video(folder / "source.mkv", found->ffprobe);
    const VideoInfo test_video = probe_video(folder / "test.mkv", found->ffprobe);
    for (const auto& [step, pairs] : {std::pair{1, 23}, std::pair{3, 8}}) {
        SCOPED_TRACE(step);
        const std::string graph = image_filtergraph(source, test_video, VmafOptions{}, std::nullopt, std::nullopt, step);
        ASSERT_NE(graph.find("blend=all_mode=or:shortest=1:repeatlast=0:ts_sync_mode=nearest"), std::string::npos);
        // Each picture's MD5, as FFmpeg's framemd5 lists it, in files of
        // this step's own.
        const auto listing_of = [&folder, step](const std::string& label) {
            return folder / (label + "-" + std::to_string(step) + ".txt");
        };
        std::vector<std::string> command{found->ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                                         "-i", platform::path_text(folder / "test.mkv"),
                                         "-i", platform::path_text(folder / "source.mkv"), "-filter_complex", graph};
        for (const char* label : {"distorted", "reference"}) {
            command.insert(command.end(), {"-map", "[" + std::string(label) + "]", "-fps_mode", "passthrough", "-f",
                                           "framemd5", platform::path_text(listing_of(label))});
        }
        const platform::ProcessOutput output = platform::run_program(command, std::chrono::minutes(2));
        ASSERT_EQ(output.exit_code, 0) << output.standard_error;
        const auto hashes = [&listing_of](const char* label) {
            std::vector<std::string> found_hashes;
            std::ifstream listing(listing_of(label));
            for (std::string line; std::getline(listing, line);) {
                if (!line.empty() && line.front() != '#') {
                    found_hashes.push_back(line.substr(line.rfind(',') + 1));
                }
            }
            return found_hashes;
        };
        EXPECT_EQ(hashes("distorted").size(), static_cast<std::size_t>(pairs));
        EXPECT_EQ(hashes("distorted"), hashes("reference"));
    }
}

// A request for a task over the same picture video twice, its black bars
// kept.
PerceptualRequest request_for(const test::Tools& found, const VideoInfo& video, int n_subsample) {
    PerceptualRequest request;
    request.environment.ffmpeg = found.ffmpeg;
    request.environment.ffprobe = found.ffprobe;
    request.source = video;
    request.distorted = video;
    request.options.crop_mode = CropMode::none;
    request.options.n_subsample = n_subsample;
    return request;
}

// The files the stand-in tools wrote into `folder`, by the start of their
// names.
std::vector<std::string> logged(const std::filesystem::path& folder, const std::string& kind) {
    std::vector<std::string> found;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
        if (platform::path_text(entry.path().filename()).starts_with(kind + "-")) {
            found.push_back(read_file(entry.path()));
        }
    }
    return found;
}

// Every pair is scored, every other frame here, each by both tools (the
// stand-ins score a pair by its number), several pairs at a time; the
// scores keep the pairs' order and the tool's version; the progress ends at
// the frames compared; and the pictures' folder is gone after the task.
TEST(PerceptualCpu, ATaskScoresEveryPairInOrder) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    const VideoInfo video = probe_video(
        test::make_clip(*found, folder / "clip.mkv", "libx264", "yuv420p", 1.0, {}, "64x48"), found->ffprobe);
    const std::filesystem::path log = folder / "log";
    std::filesystem::create_directory(log);
    const test::ScopedVariable tool("VML_TEST_TOOL", "score");
    const test::ScopedVariable logged_to("VML_TEST_TOOL_LOG", platform::path_text(log));
    const test::ScopedVariable ssimulacra2("SSIMULACRA2_PATH", VML_TEST_CHILD);
    const test::ScopedVariable butteraugli("BUTTERAUGLI_PATH", VML_TEST_CHILD);
    std::vector<std::array<long long, 2>> progress;
    const PerceptualTaskOutput output = run_cpu_perceptual_task(
        request_for(*found, video, 2), {"ssimulacra2", "butteraugli"}, std::nullopt,
        [&progress](long long current, long long total, double) { progress.push_back({current, total}); }, {},
        CancelToken());

    const FrameMetricResult* ssimulacra2_scores = output.metrics.frame("ssimulacra2");
    const FrameMetricResult* butteraugli_scores = output.metrics.frame("butteraugli");
    ASSERT_NE(ssimulacra2_scores, nullptr);
    ASSERT_NE(butteraugli_scores, nullptr);
    EXPECT_EQ(output.metrics.keys(), (std::vector<std::string>{"ssimulacra2", "butteraugli"}));
    ASSERT_EQ(ssimulacra2_scores->values.size(), 12u);  // 24 frames, every other one
    for (std::size_t pair = 0; pair < 12; ++pair) {
        EXPECT_EQ(ssimulacra2_scores->frame[pair], static_cast<int>(pair * 2));
        EXPECT_EQ(ssimulacra2_scores->values[pair], static_cast<float>(pair + 1));
        EXPECT_NEAR(butteraugli_scores->values[pair], static_cast<float>(pair + 1), 1e-4);
    }
    EXPECT_EQ(butteraugli_scores->frame, ssimulacra2_scores->frame);
    EXPECT_EQ(output.compared_frame_count, 24);
    for (const FrameMetricResult* scores : {ssimulacra2_scores, butteraugli_scores}) {
        EXPECT_EQ(scores->provenance.implementation, scores->key);
        EXPECT_EQ(scores->provenance.implementation_version, "test tool 1.0");
        EXPECT_EQ(scores->provenance.compute_backend, "cpu");
        EXPECT_EQ(scores->provenance.implementation_compatibility_id, scores->key + "-libjxl-cpu-v1");
    }
    ASSERT_FALSE(progress.empty());
    EXPECT_EQ(progress.back(), (std::array<long long, 2>{24, 24}));
    const std::vector<std::string> folders = logged(log, "folder");
    ASSERT_FALSE(folders.empty());
    EXPECT_FALSE(std::filesystem::exists(platform::path_from_text(folders.front())));
}

// Cancel ends the tools of every pair being scored at once, and the task
// says it was cancelled; the stand-ins would otherwise sleep for ten
// minutes.
TEST(PerceptualCpu, CancelEndsEveryPairBeingScored) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    const VideoInfo video = probe_video(
        test::make_clip(*found, folder / "clip.mkv", "libx264", "yuv420p", 1.0, {}, "64x48"), found->ffprobe);
    const std::filesystem::path log = folder / "log";
    std::filesystem::create_directory(log);
    const test::ScopedVariable tool("VML_TEST_TOOL", "hang");
    const test::ScopedVariable logged_to("VML_TEST_TOOL_LOG", platform::path_text(log));
    const test::ScopedVariable ssimulacra2("SSIMULACRA2_PATH", VML_TEST_CHILD);
    // As many pairs as are scored at once, two at most, are being scored
    // when the user cancels.
    const auto at_once = static_cast<std::size_t>(
        std::min(2, scoring_workers(64 * 48, {"ssimulacra2"}, 1, platform::logical_cores(), std::nullopt)));
    CancelToken cancel;
    std::thread user([&log, &cancel, at_once] {
        for (int wait = 0; wait < 6000 && logged(log, "started").size() < at_once; ++wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        cancel.cancel();
    });
    EXPECT_THROW(run_cpu_perceptual_task(request_for(*found, video, 1), {"ssimulacra2"}, std::nullopt, {}, {},
                                         cancel),
                 Cancelled);
    user.join();
    EXPECT_GE(logged(log, "started").size(), at_once);
}

// A pair is scored once FFmpeg has let go of its pictures. On Windows,
// FFmpeg can still hold a picture for a moment after giving it its name; a
// picture opened then could not be read, and the task failed. The stand-in
// FFmpeg holds the first pair's second picture for a second, while the
// scoring looks for the pair every 20 ms.
TEST(PerceptualCpu, APairIsScoredOnceFFmpegHasLetGoOfItsPictures) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const test::Folder folder;
    const VideoInfo video = probe_video(
        test::make_clip(*found, folder / "clip.mkv", "libx264", "yuv420p", 1.0, {}, "64x48"), found->ffprobe);
    const std::filesystem::path picture = folder / "picture.png";
    test::run_ffmpeg(*found, {"-i", platform::path_text(folder / "clip.mkv"), "-frames:v", "1", "-pix_fmt", "rgb48le",
                              platform::path_text(picture)});
    PerceptualRequest request = request_for(*found, video, 1);
    request.environment.ffmpeg = VML_TEST_CHILD;
    const test::ScopedVariable ffmpeg("VML_TEST_FFMPEG_PICTURE", platform::path_text(picture));
    const test::ScopedVariable pairs("VML_TEST_FFMPEG_PAIRS", "24");
    const test::ScopedVariable held("VML_TEST_FFMPEG_HOLD_MS", "1000");
    const test::ScopedVariable tool("VML_TEST_TOOL", "score");
    const test::ScopedVariable ssimulacra2("SSIMULACRA2_PATH", VML_TEST_CHILD);
    const PerceptualTaskOutput output =
        run_cpu_perceptual_task(request, {"ssimulacra2"}, std::nullopt, {}, {}, CancelToken());

    const FrameMetricResult* scores = output.metrics.frame("ssimulacra2");
    ASSERT_NE(scores, nullptr);
    EXPECT_EQ(scores->values.size(), 24u);
}

// ------------------------------------------------------------ where they run

// A task's output with `key`'s score, as one place or the other gives it.
PerceptualTaskOutput scored(const std::string& key, float value, const std::string& backend) {
    PerceptualTaskOutput output;
    const MetricProvenance provenance{key, "1", backend, key, {}};
    if (key == "cvvdp") {
        output.metrics.add(SequenceMetricResult{key, value, provenance, {}, {}, {}});
    } else {
        output.metrics.add(FrameMetricResult{key, {0}, {0.0}, {value}, provenance});
    }
    output.compared_frame_count = 1;
    return output;
}

// What the stand-in routes were asked to do, in order.
struct Routed {
    bool probed = false;
    std::vector<std::string> gpu;
    std::vector<std::string> cpu;
    std::optional<CropPair> cpu_crops;
    std::vector<std::string> events;  // progress reports and the CPU's announcements
    std::vector<std::string> statuses;
};

// Routes whose GPU is `device` (none: "no supported GPU"), and whose GPU and
// CPU work is `on_gpu` and `on_cpu`.
PerceptualRoutes stand_in_routes(
    Routed& routed, std::optional<VshipDevice> device,
    std::function<PerceptualTaskOutput(const std::vector<std::string>&, const PerceptualHooks&)> on_gpu,
    std::function<PerceptualTaskOutput(const std::vector<std::string>&, const ProgressCallback&)> on_cpu) {
    PerceptualRoutes routes;
    routes.find_gpu = [&routed, device] {
        routed.probed = true;
        return std::pair(device, std::string(device ? "" : "no supported GPU"));
    };
    routes.find_crops = [](const PerceptualRequest&, const StatusCallback&, const CancelToken&) { return CropPair{}; };
    routes.run_gpu = [&routed, on_gpu](const PerceptualRequest&, const std::vector<std::string>& keys,
                                       const VshipDevice&, const CropPair&, bool, const PerceptualHooks& hooks,
                                       const CancelToken&) {
        routed.gpu = keys;
        return on_gpu(keys, hooks);
    };
    routes.run_cpu = [&routed, on_cpu](const PerceptualRequest&, const std::vector<std::string>& keys,
                                       const std::optional<CropPair>& crops, const ProgressCallback& on_progress,
                                       const StatusCallback&, const CancelToken&) {
        routed.cpu = keys;
        routed.cpu_crops = crops;
        return on_cpu(keys, on_progress);
    };
    return routes;
}

PerceptualHooks recording_hooks(Routed& routed) {
    PerceptualHooks hooks;
    hooks.on_progress = [&routed](long long current, long long total, double fps) {
        routed.events.push_back(std::to_string(current) + "/" + std::to_string(total) + " at " +
                                std::to_string(static_cast<int>(fps)));
    };
    hooks.on_status = [&routed](const RunStatus& status) { routed.statuses.push_back(status.text); };
    hooks.on_cpu = [&routed](const std::vector<std::string>& keys) {
        std::string joined;
        for (const std::string& key : keys) {
            joined += (joined.empty() ? "" : ", ") + key;
        }
        routed.events.push_back("cpu: " + joined);
    };
    return hooks;
}

PerceptualRequest routed_request(double seconds = 60.0) {
    PerceptualRequest request;
    request.source = plain_video("source.mkv");
    request.distorted = plain_video("test.mkv");
    for (VideoInfo* video : {&request.source, &request.distorted}) {
        video->duration = seconds;
        video->nb_frames = static_cast<long long>(seconds * 24);
    }
    request.options.crop_mode = CropMode::none;
    return request;
}

const VshipDevice test_gpu{"cuda", "test GPU", 0, "5.1.2", GpuVendor::nvidia};

// Without a GPU Vship can use, SSIMULACRA2 goes to the CPU, and CVVDP,
// which runs on the GPU only, fails with the reason; alone, CVVDP fails the
// task.
TEST(PerceptualRouting, WithoutAGpuCvvdpFailsAndTheOthersRunOnTheCpu) {
    Routed routed;
    const PerceptualRoutes routes = stand_in_routes(
        routed, std::nullopt, [](const auto&, const auto&) -> PerceptualTaskOutput { throw std::logic_error("GPU"); },
        [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); });
    const PerceptualRequest request = routed_request();
    const PerceptualTaskOutput output =
        run_perceptual_task(request, {"ssimulacra2", "cvvdp"}, false, recording_hooks(routed), CancelToken(), routes);
    EXPECT_EQ(routed.cpu, (std::vector<std::string>{"ssimulacra2"}));
    EXPECT_TRUE(output.metrics.has("ssimulacra2"));
    ASSERT_NE(failure_of(output.failures, "cvvdp"), nullptr);
    EXPECT_EQ(*failure_of(output.failures, "cvvdp"),
              "CVVDP needs a GPU that Vship can use, and could not use one: no supported GPU");
    EXPECT_EQ(routed.statuses,
              (std::vector<std::string>{"Vship GPU unavailable (no supported GPU); using CPU reference metrics\xE2\x80\xA6"}));
    EXPECT_EQ(routed.events, (std::vector<std::string>{"cpu: ssimulacra2"}));

    try {
        run_perceptual_task(request, {"cvvdp"}, false, recording_hooks(routed), CancelToken(), routes);
        ADD_FAILURE() << "CVVDP alone did not fail";
    } catch (const PerceptualRunError& error) {
        EXPECT_EQ(std::string(error.what()),
                  "CVVDP needs a GPU that Vship can use, and could not use one: no supported GPU");
    }
}

// Each metric runs where it is set to, the CPU's with the black bars the GPU
// task found; each stage's progress is its own, the CPU's from 0 once it is
// told which metrics it takes; and a choice of the CPU alone does not probe
// the GPU.
TEST(PerceptualRouting, EachMetricRunsWhereItIsSet) {
    Routed routed;
    const PerceptualRoutes routes = stand_in_routes(
        routed, test_gpu,
        [](const auto&, const PerceptualHooks& hooks) {
            hooks.on_progress(1, 1, 10.0);
            return scored("ssimulacra2", 91.0f, "gpu");
        },
        [](const auto&, const ProgressCallback& on_progress) {
            on_progress(1, 1, 8.0);
            return scored("butteraugli", 0.2f, "cpu");
        });
    PerceptualRequest request = routed_request();
    request.backends = {{"ssimulacra2", "gpu"}, {"butteraugli", "cpu"}};
    const PerceptualTaskOutput output = run_perceptual_task(request, {"ssimulacra2", "butteraugli"}, false,
                                                            recording_hooks(routed), CancelToken(), routes);
    EXPECT_EQ(routed.gpu, (std::vector<std::string>{"ssimulacra2"}));
    EXPECT_EQ(routed.cpu, (std::vector<std::string>{"butteraugli"}));
    EXPECT_TRUE(routed.cpu_crops.has_value());
    EXPECT_EQ(output.metrics.keys(), (std::vector<std::string>{"ssimulacra2", "butteraugli"}));
    EXPECT_EQ(output.metrics.frame("ssimulacra2")->provenance.compute_backend, "gpu");
    EXPECT_EQ(output.metrics.frame("butteraugli")->provenance.compute_backend, "cpu");
    EXPECT_EQ(routed.events, (std::vector<std::string>{"1/1 at 10", "cpu: butteraugli", "1/1 at 8"}));

    Routed cpu_only;
    const PerceptualRoutes cpu_routes = stand_in_routes(
        cpu_only, test_gpu, [](const auto&, const auto&) -> PerceptualTaskOutput { throw std::logic_error("GPU"); },
        [](const auto&, const auto&) { return scored("butteraugli", 0.2f, "cpu"); });
    run_perceptual_task(request, {"butteraugli"}, false, recording_hooks(cpu_only), CancelToken(), cpu_routes);
    EXPECT_FALSE(cpu_only.probed);
    EXPECT_FALSE(cpu_only.cpu_crops.has_value());  // the CPU task finds the black bars itself
}

// A metric that failed on the GPU while CVVDP finished is calculated on the
// CPU for a video up to ten minutes long; for a longer one, nobody agreed
// to the CPU's days, so it fails with the advice.
TEST(PerceptualRouting, AMetricThatFailedOnTheGpuMovesToTheCpuOnlyForShortVideos) {
    for (const double seconds : {60.0, 7200.0}) {
        SCOPED_TRACE(seconds);
        Routed routed;
        const PerceptualRoutes routes = stand_in_routes(
            routed, test_gpu,
            [](const auto&, const auto&) {
                PerceptualTaskOutput output = scored("cvvdp", 9.4f, "gpu");
                output.failures = {{"ssimulacra2", "Vship SSIMULACRA2 failed: out of memory"}};
                return output;
            },
            [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); });
        const PerceptualTaskOutput output = run_perceptual_task(
            routed_request(seconds), {"ssimulacra2", "cvvdp"}, false, recording_hooks(routed), CancelToken(), routes);
        EXPECT_TRUE(output.metrics.has("cvvdp"));
        if (seconds < long_cpu_run_seconds) {
            EXPECT_EQ(routed.cpu, (std::vector<std::string>{"ssimulacra2"}));
            EXPECT_TRUE(output.failures.empty());
            EXPECT_EQ(output.metrics.frame("ssimulacra2")->provenance.compute_backend, "cpu");
            EXPECT_EQ(routed.statuses.back(), "SSIMULACRA2 failed on the GPU; calculating it on the CPU\xE2\x80\xA6");
        } else {
            EXPECT_TRUE(routed.cpu.empty());
            EXPECT_FALSE(output.metrics.has("ssimulacra2"));
            ASSERT_NE(failure_of(output.failures, "ssimulacra2"), nullptr);
            EXPECT_EQ(*failure_of(output.failures, "ssimulacra2"),
                      "GPU scoring failed (Vship SSIMULACRA2 failed: out of memory). It was not retried on the CPU, "
                      "which would take hours to days for a video over 10 minutes. Choose CPU for SSIMULACRA2 to "
                      "calculate it on the CPU anyway.");
        }
    }
}

// When the GPU task fails, the CPU calculates what it can, for a video up to
// ten minutes long; CVVDP fails with the GPU's reason, which does not blame
// the GPU. For a longer video the task fails, with CVVDP's reason too.
TEST(PerceptualRouting, AFailedGpuTaskMovesToTheCpuOnlyForShortVideos) {
    for (const double seconds : {60.0, 7200.0}) {
        SCOPED_TRACE(seconds);
        Routed routed;
        const PerceptualRoutes routes = stand_in_routes(
            routed, test_gpu,
            [](const auto&, const auto&) -> PerceptualTaskOutput {
                throw PerceptualRunError("the frame rates differ");
            },
            [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); });
        const PerceptualRequest request = routed_request(seconds);
        if (seconds < long_cpu_run_seconds) {
            const PerceptualTaskOutput output = run_perceptual_task(request, {"ssimulacra2", "cvvdp"}, false,
                                                                    recording_hooks(routed), CancelToken(), routes);
            EXPECT_EQ(routed.cpu, (std::vector<std::string>{"ssimulacra2"}));
            EXPECT_TRUE(output.metrics.has("ssimulacra2"));
            ASSERT_NE(failure_of(output.failures, "cvvdp"), nullptr);
            EXPECT_EQ(*failure_of(output.failures, "cvvdp"), "CVVDP could not be calculated: the frame rates differ");
            EXPECT_EQ(routed.statuses.back(),
                      "Vship GPU compute failed (the frame rates differ); using CPU reference metrics\xE2\x80\xA6");
            continue;
        }
        try {
            run_perceptual_task(request, {"ssimulacra2", "cvvdp"}, false, recording_hooks(routed), CancelToken(),
                                routes);
            ADD_FAILURE() << "no error";
        } catch (const PerceptualRunError& error) {
            EXPECT_EQ(std::string(error.what()),
                      "GPU scoring failed (the frame rates differ). It was not retried on the CPU, which would take "
                      "hours to days for a video over 10 minutes. Choose CPU for SSIMULACRA2 to calculate it on the "
                      "CPU anyway.\nCVVDP could not be calculated: the frame rates differ");
        }
        EXPECT_TRUE(routed.cpu.empty());
    }
}

// A cancelled GPU task does not start the CPU's.
TEST(PerceptualRouting, CancellingDoesNotStartTheCpu) {
    Routed routed;
    const PerceptualRoutes routes = stand_in_routes(
        routed, test_gpu, [](const auto&, const auto&) -> PerceptualTaskOutput { throw Cancelled(); },
        [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); });
    EXPECT_THROW(run_perceptual_task(routed_request(), {"ssimulacra2"}, false, recording_hooks(routed), CancelToken(),
                                     routes),
                 Cancelled);
    EXPECT_TRUE(routed.cpu.empty());
}

}  // namespace
}  // namespace vml
