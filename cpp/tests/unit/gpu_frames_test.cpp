// The app's own decoding of videos (gpu_frames.h) held to the Python app's
// (decoding.json, from gpu_frames.py and vmaf_runner.py): the plans, FFmpeg's
// command and the choice of decoders on every system; and, where the PC has
// a decoder, its pictures against FFmpeg's decoding, and VMAF on the GPU
// from them against VMAF from FFmpeg's frames.
#include "vml/gpu_frames.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/gpu_vmaf.h"
#include "vml/model_select.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/vmaf_v1_gpu.h"

namespace vml {
namespace {

using Json = nlohmann::json;
using test::Folder;
using test::make_clip;
using test::Tools;
using test::tools;

// VML_SOFTWARE_DECODE as a test sets it, put back as it was.
class SoftwareDecode : public test::ScopedVariable {
public:
    explicit SoftwareDecode(const std::string& value)
        : test::ScopedVariable(std::string(software_decode_variable), value) {}
};

TEST(GpuFrames, PlansAsThePythonAppDoes) {
    const Json reference = test::load_reference("decoding.json");
    ASSERT_FALSE(reference["plans"].empty());
    for (const Json& item : reference["plans"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        VideoInfo info;
        info.path = "video.mkv";
        info.codec_name = item["codec_name"].get<std::string>();
        info.pix_fmt = item["pix_fmt"].get<std::string>();
        info.width = item["width"].get<int>();
        info.height = item["height"].get<int>();
        std::optional<CropBox> crop;
        if (!item["crop"].is_null()) {
            const std::vector<int> box = item["crop"].get<std::vector<int>>();
            crop = CropBox{box[0], box[1], box[2], box[3]};
        }
        std::optional<std::pair<int, int>> size;
        if (!item["size"].is_null()) {
            size = std::pair(item["size"][0].get<int>(), item["size"][1].get<int>());
        }
        const auto plan = [&] {
            return plan_decode(info, crop, item["shift"].get<int>(), item["luma_only"].get<bool>(), size,
                               item["algorithm"].get<std::string>());
        };
        if (item.contains("error")) {
            try {
                plan();
                ADD_FAILURE() << "planned to decode it in the app";
            } catch (const GpuDecodeUnavailableError& error) {
                EXPECT_EQ(error.what(), item["error"].get<std::string>());
            }
            continue;
        }
        const DecodePlan made = plan();
        const Json& expected = item["plan"];
        EXPECT_EQ(made.codec, expected["codec"].get<std::string>());
        EXPECT_EQ(made.bit_depth, expected["bit_depth"].get<int>());
        EXPECT_EQ(made.width, expected["width"].get<int>());
        EXPECT_EQ(made.height, expected["height"].get<int>());
        EXPECT_EQ((std::vector<int>{made.crop_x, made.crop_y, made.crop_w, made.crop_h}),
                  expected["crop"].get<std::vector<int>>());
        EXPECT_EQ(made.shift, expected["shift"].get<int>());
        EXPECT_EQ(made.luma_only, expected["luma_only"].get<bool>());
        EXPECT_EQ((std::vector<int>{made.out_w, made.out_h}), expected["out"].get<std::vector<int>>());
        EXPECT_EQ(made.scaler, expected["scaler"].get<std::string>());
        EXPECT_EQ(made.scaled(), expected["scaled"].get<bool>());
        EXPECT_EQ(made.frame_bytes(), expected["frame_bytes"].get<std::size_t>());
    }
}

TEST(GpuFrames, CopiesThePacketsWithThePythonAppsCommand) {
    const Json reference = test::load_reference("decoding.json");
    ASSERT_FALSE(reference["packet_commands"].empty());
    for (const auto& [codec, expected] : reference["packet_commands"].items()) {
        SCOPED_TRACE(codec);
        std::vector<std::string> command = packet_reader_command("<ffmpeg>", "video.mkv", codec, "<listing>");
        for (std::string& part : command) {
            if (part == input_path("video.mkv")) {
                part = "<video>";
            }
        }
        EXPECT_EQ(command, expected.get<std::vector<std::string>>());
    }
}

// Each video by its GPU's decoder or the software decoder, as the Python app
// chooses. With the software decoder not bundled (or VML_SOFTWARE_DECODE
// set to ffmpeg) everywhere, and bundled where it is.
TEST(GpuFrames, DecodesTheVideosThePythonAppDecodesItself) {
    const Json reference = test::load_reference("decoding.json");
    int checked = 0;
    for (const Json& item : reference["decoders"]) {
        const SoftwareDecode software(item["software_decode"].get<std::string>());
        if (software_decoder_bundled() != item["bundled"].get<bool>()) {
            continue;  // the software decoder is not bundled for this system
        }
        const auto accel = [](const Json& value) {
            return value.is_null() ? std::nullopt : std::optional(value.get<std::string>());
        };
        const HardwareDecoding decoding{accel(item["source"]), accel(item["test"])};
        SCOPED_TRACE(decoding.describe() + ", " + item["software_decode"].get<std::string>());
        const std::optional<std::pair<std::string, std::string>> decoders = decoders_here(decoding);
        if (item["decoders"].is_null()) {
            EXPECT_FALSE(decoders.has_value());
        } else {
            ASSERT_TRUE(decoders.has_value());
            EXPECT_EQ((std::vector<std::string>{decoders->first, decoders->second}),
                      item["decoders"].get<std::vector<std::string>>());
        }
        ++checked;
    }
    EXPECT_GE(checked, 20);  // at least the cases without the software decoder
}

// The decoder's pictures must match the packets fed: in timestamp order,
// each the next packet's, and one for every packet by the end. Those an MP4
// edit list cuts off are dropped.
TEST(GpuFrames, PicturesThatDoNotMatchThePacketsFailTheRun) {
    PictureChecks checks("the GPU's decoder");
    for (const long long pts : {0, 3000, 1000, 2000}) {  // in decode order
        checks.fed(pts, pts == 0);
    }
    EXPECT_TRUE(checks.take(0));  // discarded
    EXPECT_FALSE(checks.take(1000));
    EXPECT_FALSE(checks.take(2000));
    const auto message = [](const auto& call) {
        try {
            call();
        } catch (const GpuDecodeFailedError& error) {
            return std::string(error.what());
        }
        return std::string();
    };
    EXPECT_EQ(message([&] { checks.check_end(); }), "the GPU's decoder gave 3 pictures for 4 packets");
    checks.take(3000);
    EXPECT_EQ(message([&] { checks.check_end(); }), "");
    EXPECT_EQ(message([&] { checks.take(2500); }), "the GPU's decoder gave pictures out of order");

    PictureChecks skipped("the software decoder");
    for (const long long pts : {0, 1000, 2000}) {
        skipped.fed(pts, false);
    }
    skipped.take(0);
    EXPECT_EQ(message([&] { skipped.take(2000); }), "the software decoder gave 1 pictures for 2 packets");
    PictureChecks made_up("the GPU's decoder");
    made_up.fed(1000, false);
    EXPECT_EQ(message([&] { made_up.take(500); }), "the GPU's decoder gave pictures the packets do not have");
}

// ----------------------------------------------------- on the PC's decoders

// FFmpeg's decoding of the clip, cropped as the plan says, as raw pictures:
// its standard output read as data (run_program reads it as text).
std::string ffmpeg_pictures(const Tools& found, const std::filesystem::path& path, const DecodePlan& plan) {
    const std::string crop = "crop=" + std::to_string(plan.crop_w) + ":" + std::to_string(plan.crop_h) + ":" +
                             std::to_string(plan.crop_x) + ":" + std::to_string(plan.crop_y);
    platform::ChildProcess ffmpeg = platform::ChildProcess::start(
        {found.ffmpeg, "-nostdin", "-v", "error", "-i", platform::path_text(path), "-map", "0:v:0", "-vf",
         crop + ",format=" + (plan.bit_depth > 8 ? "yuv420p10le" : "yuv420p"), "-fps_mode", "passthrough", "-f",
         "rawvideo", "pipe:1"});
    std::string pictures;
    std::vector<char> buffer(1 << 20);
    while (const std::size_t count = ffmpeg.read(buffer.data(), buffer.size())) {
        pictures.append(buffer.data(), count);
    }
    EXPECT_EQ(ffmpeg.wait(), 0) << ffmpeg.standard_error();
    return pictures;
}

// The decoder's pictures, packed, one after another.
std::string decoded_pictures(const Tools& found, const VideoInfo& info, const DecodePlan& plan,
                             const std::string& backend) {
    GpuFrameStream stream(info, plan, 0, 4, backend, found.ffmpeg, found.ffprobe);
    stream.start();
    std::string pictures;
    std::vector<char> picture(plan.frame_bytes());
    std::optional<long long> last;
    while (const std::optional<GpuFrameStream::Picture> next = stream.next(CancelToken())) {
        stream.download(next->slot, picture.data());
        stream.release(next->slot);
        pictures.append(picture.data(), picture.size());
        EXPECT_TRUE(!last || next->pts > *last);
        last = next->pts;
    }
    return pictures;
}

// The decoders' pictures are FFmpeg's, sample for sample: whole and cropped,
// at 8 and 10 bits, and without the frames an MP4's edit list cuts off,
// which FFmpeg decodes and drops. With each decoder this PC has.
TEST(GpuFrames, PicturesAreFfmpegsDecoding) {
    const std::optional<Tools> found = tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const Folder folder;
    const std::filesystem::path h264 = make_clip(*found, folder / "h264.mkv", "libx264", "yuv420p", 2.0);
    const std::filesystem::path hevc = make_clip(*found, folder / "hevc.mkv", "libx265", "yuv420p10le", 2.0);
    // A copy cut out of an MP4 starts at a keyframe before the cut, and its
    // edit list marks the frames before the cut to be dropped.
    const std::filesystem::path whole =
        make_clip(*found, folder / "whole.mp4", "libx264", "yuv420p", 4.0, {"-g", "48"});
    const std::filesystem::path cut = folder / "cut.mp4";
    test::run_ffmpeg(*found, {"-ss", "1.3", "-i", platform::path_text(whole), "-c", "copy", platform::path_text(cut)});
    const std::vector<std::pair<std::filesystem::path, std::optional<CropBox>>> clips{
        {h264, std::nullopt}, {h264, CropBox{600, 300, 20, 30}}, {hevc, CropBox{638, 358, 2, 2}}, {cut, std::nullopt}};
    int decoded = 0;
    for (const auto& [path, crop] : clips) {
        const VideoInfo info = probe_video(path, found->ffprobe);
        const DecodePlan plan = plan_decode(info, crop, 6);
        const std::string expected = ffmpeg_pictures(*found, path, plan);
        for (const std::string backend : {"nvidia", "intel", "amd", "software"}) {
            if (!decoder_supports(0, plan, backend).first) {
                continue;  // no such decoder on this PC
            }
            SCOPED_TRACE(platform::path_text(path.filename()) + (crop ? ", cropped" : "") + ", " + backend);
            const std::string pictures = decoded_pictures(*found, info, plan, backend);
            EXPECT_EQ(pictures.size() / plan.frame_bytes(), expected.size() / plan.frame_bytes());
            EXPECT_TRUE(pictures == expected) << "the pictures differ from FFmpeg's";
            ++decoded;
        }
    }
    if (decoded == 0) {
        GTEST_SKIP() << "no decoder of the app's on this PC";
    }
}

// Scores clips of `size` with each route the PC has: FFmpeg's frames, the
// GPU's decoder and the software decoder. Holds each route's scores to
// FFmpeg's, and counts the app's routes it compared in `routes`.
void compare_routes(const Tools& found, const RunEnvironment& environment, const std::string& size, int& routes) {
    const Folder folder;
    const std::filesystem::path source_path =
        make_clip(found, folder / "source.mkv", "libx264", "yuv420p", 2.0, {"-crf", "10"}, size);
    const std::filesystem::path test_path =
        make_clip(found, folder / "test.mkv", "libx265", "yuv420p", 2.0, {"-crf", "32"}, size);
    const VideoInfo source = probe_video(source_path, found.ffprobe);
    const VideoInfo test = probe_video(test_path, found.ffprobe);

    // Each route's scores, and the statuses its run sent.
    struct Route {
        std::vector<float> vmaf;
        std::vector<float> neg;
        std::vector<std::string> statuses;
    };
    const auto run = [&](bool gpu_decode, const std::string& software_decode) {
        const SoftwareDecode software(software_decode);
        VmafOptions options;
        options.compute_vmaf_neg = true;
        options.gpu_decode = gpu_decode;
        options.crop_mode = CropMode::none;
        Route route;
        const ComparisonResult result = run_vmaf(
            environment, source, test, options, nullptr,
            [&](const RunStatus& status) { route.statuses.push_back(status.text); }, CancelToken());
        route.vmaf = result.metric_results.frame("vmaf")->values;
        route.neg = result.metric_results.frame("vmaf_neg")->values;
        return route;
    };
    // Whether the app decoded the videos: it said it would, and its decoding
    // did not fail (FFmpeg would then have decoded them after all).
    const auto decoded_here = [](const Route& route) {
        const auto said = [&](std::string_view text) {
            return std::any_of(route.statuses.begin(), route.statuses.end(),
                               [&](const std::string& status) { return status.find(text) != std::string::npos; });
        };
        EXPECT_FALSE(said("decoding through FFmpeg instead")) << "the app's decoding failed";
        return said("Running VMAF on the GPU (GPU decode:") && !said("decoding through FFmpeg instead");
    };
    const Route ffmpeg = run(false, "ffmpeg");
    ASSERT_FALSE(decoded_here(ffmpeg));
    ASSERT_EQ(ffmpeg.vmaf.size(), 48u);
    for (const bool gpu_decode : {true, false}) {
        const Route app = run(gpu_decode, "");
        if (!decoded_here(app)) {
            continue;  // no decoder for it on this PC: FFmpeg decoded it
        }
        SCOPED_TRACE(gpu_decode ? "the GPU's decoder" : "the software decoder");
        EXPECT_EQ(app.vmaf, ffmpeg.vmaf);
        EXPECT_EQ(app.neg, ffmpeg.neg);
        ++routes;
    }
}

// VMAF and NEG on the GPU from the pictures the app decodes itself, with
// NVIDIA's decoder (whose pictures stay on the GPU) and with the software
// decoder, are exactly the scores from FFmpeg's frames: also at a width
// whose rows the Vulkan engine pads.
TEST(GpuFrames, VmafOnTheGpuScoresTheAppsPicturesAsFfmpegsFrames) {
    const std::optional<Tools> found = tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    if (gpu_vmaf_backend().backend != "vulkan") {
        GTEST_SKIP() << "no GPU that Vulkan scores VMAF on";
    }
    const RunEnvironment environment{found->ffmpeg, found->version, found->ffprobe,
                                     DecodingCapabilities::detect(found->ffmpeg), platform::logical_cores()};
    std::vector<int> routes;
    for (const std::string size : {"640x360", "638x360"}) {
        SCOPED_TRACE(size);
        compare_routes(*found, environment, size, routes.emplace_back(0));
    }
    EXPECT_EQ(routes[1], routes[0]) << "the padded width took other routes";
    if (routes[0] == 0) {
        GTEST_SKIP() << "no decoder of the app's on this PC";
    }
}

// VMAF v1 on the GPU from the pictures a decoder keeps on the GPU (NVIDIA's,
// and AMD's handed over to Vulkan) scores what it scores from FFmpeg's
// frames: with each picture copied on the GPU, and with Windows' decoder's
// pictures read where it left them (pictures mode, their slots given back
// once the GPU has read them), HEVC's and H.264's (layers of a texture
// array there, copied into textures of their own); every frame and every
// other, and with VMAF v0.6.1 beside it.
TEST(GpuFrames, VmafV1ScoresThePicturesKeptOnTheGpuAsFfmpegsFrames) {
    const std::optional<Tools> found = tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    if (!v1_available().available) {
        GTEST_SKIP() << "VMAF v1 is not calculated with this PC's GPU";
    }
    const std::string v1 = "path=" + platform::path_text(resource_folder() / "models" / "vmaf_v1.0.16" /
                                                         "vmaf_v1.0.16_3d0h.json");
    const struct {
        const char* codec;
        GpuModels models;
        int step;
    } cases[] = {{"hevc", {{"vmaf_v1", v1}}, 1},
                 {"hevc", {{"vmaf_v1", v1}}, 2},
                 {"hevc", {{"vmaf", "vmaf_v0.6.1"}, {"vmaf_v1", v1}}, 1},
                 {"h264", {{"vmaf_v1", v1}}, 1}};
    int routes = 0;
    for (const auto& checked : cases) {
        SCOPED_TRACE(std::string(checked.codec) + ", " + std::to_string(checked.models.size()) + " models, every " +
                     std::to_string(checked.step));
        const bool h264 = std::string_view(checked.codec) == "h264";
        const int bits = h264 ? 8 : 10;
        const Folder folder;
        const auto clip = [&](const char* name, const char* quality) {
            return make_clip(*found, folder / name, h264 ? "libx264" : "libx265", h264 ? "yuv420p" : "yuv420p10le",
                             2.0, {"-crf", quality});
        };
        const VideoInfo source = probe_video(clip("source.mkv", "10"), found->ffprobe);
        const VideoInfo test = probe_video(clip("test.mkv", "32"), found->ffprobe);
        const DecodePlan source_plan = plan_decode(source, std::nullopt, 6);
        const DecodePlan test_plan = plan_decode(test, std::nullopt, 6);
        const std::size_t frame_bytes = source_plan.frame_bytes();

        // FFmpeg's frames, from system memory.
        const std::string source_frames = ffmpeg_pictures(*found, source.path, source_plan);
        const std::string test_frames = ffmpeg_pictures(*found, test.path, test_plan);
        ASSERT_EQ(source_frames.size(), test_frames.size());
        ASSERT_GT(source_frames.size() / frame_bytes, 20u);
        MultiScorer from_ffmpeg(640, 360, bits, checked.models, checked.step, "vulkan", std::nullopt);
        for (std::size_t at = 0; at + frame_bytes <= source_frames.size(); at += frame_bytes) {
            const auto frame = [&](const std::string& frames) {
                return std::span(reinterpret_cast<const std::uint8_t*>(frames.data() + at), frame_bytes);
            };
            from_ffmpeg.add(frame(source_frames), frame(test_frames));
        }
        const PairScores expected = from_ffmpeg.finish();

        for (const std::string backend : {"nvidia", "amd"}) {
            for (const bool pictures : {false, true}) {
                if (!decoder_supports(0, test_plan, backend).first || (pictures && !hands_over_textures(backend))) {
                    continue;  // no such decoder on this PC
                }
                SCOPED_TRACE(backend + (pictures ? ", pictures read in place" : ", pictures copied"));
                const int pool = pictures ? 6 : 4;
                GpuFrameStream test_stream(test, test_plan, 0, pool, backend, found->ffmpeg, found->ffprobe, true);
                GpuFrameStream source_stream(source, source_plan, 0, pool, backend, found->ffmpeg, found->ffprobe,
                                             true);
                ASSERT_TRUE(test_stream.handover() && source_stream.handover());
                // Windows' decoder gives its pictures as textures whichever way
                // they are scored; reading them in place is the scorer's choice.
                if (pictures) {
                    ASSERT_TRUE(test_stream.gives_textures() && source_stream.gives_textures());
                }
                std::optional<MultiScorer> scorer;
                scorer.emplace(640, 360, bits, checked.models, checked.step, "vulkan", std::nullopt, &test_stream,
                               pictures);
                ASSERT_EQ(scorer->pictures(), pictures);
                test_stream.start();
                source_stream.start();
                SlotReleases releases;
                if (pictures) {
                    releases.test = [&](int slot) { scorer->release_later(test_stream, slot, false); };
                    releases.reference = [&](int slot) { scorer->release_later(source_stream, slot, true); };
                }
                score_frame_pairs(
                    test_stream, source_stream, test, std::nullopt, 0,
                    [&](int test_slot, int source_slot) {
                        scorer->add_decoded(source_stream, source_slot, test_stream, test_slot);
                    },
                    {}, CancelToken(), releases);
                const PairScores got = scorer->finish();
                EXPECT_EQ(got.frames, expected.frames);
                EXPECT_EQ(got.by_metric, expected.by_metric);
                ++routes;
            }
        }
    }
    if (routes == 0) {
        GTEST_SKIP() << "no decoder on this PC that keeps its pictures on the GPU";
    }
}

// Decoding in the app that fails once it has started (here a 10-bit video
// whose details say 8-bit, which the decoder finds in its first picture)
// hands the run to FFmpeg, which finishes it, and the run says so.
TEST(GpuFrames, DecodingThatFailsHandsTheRunToFfmpeg) {
    const std::optional<Tools> found = tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const SoftwareDecode software("");
    if (!software_decoder_bundled() || gpu_vmaf_backend().backend != "vulkan") {
        GTEST_SKIP() << "no software decoder, or no GPU that Vulkan scores VMAF on";
    }
    const Folder folder;
    const VideoInfo source = probe_video(
        make_clip(*found, folder / "source.mkv", "libx264", "yuv420p", 2.0, {"-crf", "10"}), found->ffprobe);
    VideoInfo test =
        probe_video(make_clip(*found, folder / "test.mkv", "libx265", "yuv420p10le", 2.0), found->ffprobe);
    test.pix_fmt = "yuv420p";
    const RunEnvironment environment{found->ffmpeg, found->version, found->ffprobe, DecodingCapabilities{},
                                     platform::logical_cores()};
    VmafOptions options;
    options.gpu_decode = false;  // the software decoder, which checks each picture's depth
    options.crop_mode = CropMode::none;
    std::vector<std::string> statuses;
    const ComparisonResult result = run_vmaf(
        environment, source, test, options, nullptr,
        [&](const RunStatus& status) { statuses.push_back(status.text); }, CancelToken());
    EXPECT_TRUE(std::any_of(statuses.begin(), statuses.end(), [](const std::string& status) {
        return status.starts_with("Decoding in the app failed (") &&
               status.ends_with("); decoding through FFmpeg instead…");
    }));
    ASSERT_NE(result.metric_results.frame("vmaf"), nullptr);
    EXPECT_EQ(result.metric_results.frame("vmaf")->values.size(), 48u);
    EXPECT_EQ(result.metric_results.frame("vmaf")->provenance.compute_backend, "gpu");
}

// An AVI with B-frames gives its packets no presentation time (FFmpeg lists
// the lowest 64-bit number for it), so the app's decoding cannot put its
// pictures in order: it is refused with the reason the Python app gives,
// which the run's status and the log show, and FFmpeg decodes the video.
TEST(GpuFrames, PacketsWithoutTimestampsAreRefusedForTheReasonThePythonAppGives) {
    const std::optional<Tools> found = tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    const SoftwareDecode software("");
    if (!software_decoder_bundled()) {
        GTEST_SKIP() << "no software decoder";
    }
    const Folder folder;
    const VideoInfo info = probe_video(
        make_clip(*found, folder / "b-frames.avi", "libx264", "yuv420p", 1.0, {"-bf", "2"}), found->ffprobe);
    GpuFrameStream stream(info, plan_decode(info, std::nullopt, 6), 0, 4, "software", found->ffmpeg, found->ffprobe);
    try {
        stream.start();
        while (const std::optional<GpuFrameStream::Picture> picture = stream.next(CancelToken())) {
            stream.release(picture->slot);
        }
        ADD_FAILURE() << "decoded";
    } catch (const GpuDecodeFailedError& error) {
        EXPECT_STREQ(error.what(), "a packet has no timestamp");
    }
}

}  // namespace
}  // namespace vml
