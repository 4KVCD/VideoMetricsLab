// PSNR, SSIM and XPSNR calculated in the app (libvmaf_cpu.h) held to the
// scores of FFmpeg's own filters, through the run (run_vmaf), on each route
// the frames take to the app on this PC: the GPU's decoder, the software
// decoder, and FFmpeg's pipes.
#include "vml/libvmaf_cpu.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "test_videos.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffprobe.h"
#include "vml/gpu_frames.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

// The scores are FFmpeg's to the decimals the app keeps (its log's six,
// XPSNR's stats file's four), at a depth the source is widened to, with
// every n_subsample-th frame scored.
TEST(LibvmafCpu, PsnrSsimAndXpsnrInTheAppAreFfmpegsOwnScores) {
    const std::optional<test::Tools> found = test::tools();
    if (!found) {
        GTEST_SKIP() << "no FFmpeg";
    }
    if (!cpu_scores_xpsnr()) {
        GTEST_SKIP() << "the app ships no libvmaf-fast with XPSNR for this system";
    }
    const test::Folder folder;
    // An 8-bit source against a 10-bit test video: compared at 10 bits.
    const VideoInfo source = probe_video(
        test::make_clip(*found, folder / "source.mkv", "libx264", "yuv420p", 2.0, {"-crf", "10"}), found->ffprobe);
    const VideoInfo test_video = probe_video(
        test::make_clip(*found, folder / "test.mkv", "libx265", "yuv420p10le", 2.0, {"-crf", "32"}), found->ffprobe);
    const RunEnvironment environment{found->ffmpeg, found->version, found->ffprobe,
                                     DecodingCapabilities::detect(found->ffmpeg), platform::logical_cores()};

    struct Route {
        std::map<std::string, std::vector<float>> scores;
        std::map<std::string, std::string> implementations;
        std::vector<std::string> statuses;
    };
    const auto run = [&](bool gpu_decode, const std::string& cpu_metrics, const std::string& software_decode) {
        const test::ScopedVariable metrics_variable("VML_CPU_METRICS", cpu_metrics);
        const test::ScopedVariable decode_variable(std::string(software_decode_variable), software_decode);
        VmafOptions options;
        options.compute_vmaf = false;
        options.extra_features = {"name=psnr", "name=float_ssim"};
        options.compute_xpsnr = true;
        options.n_subsample = 2;
        options.gpu_decode = gpu_decode;
        options.crop_mode = CropMode::none;
        Route route;
        const ComparisonResult result = run_vmaf(
            environment, source, test_video, options, nullptr,
            [&](const RunStatus& status) { route.statuses.push_back(status.text); }, CancelToken());
        for (const std::string key : {"psnr", "ssim", "xpsnr"}) {
            const FrameMetricResult* metric = result.metric_results.frame(key);
            EXPECT_NE(metric, nullptr) << key;
            if (metric != nullptr) {
                route.scores[key] = metric->values;
                route.implementations[key] = metric->provenance.implementation;
            }
        }
        return route;
    };
    const auto said = [](const Route& route, const std::string& start) {
        return std::any_of(route.statuses.begin(), route.statuses.end(),
                           [&](const std::string& status) { return status.starts_with(start); });
    };

    const Route ffmpeg = run(true, "ffmpeg", "");
    ASSERT_EQ(ffmpeg.scores.at("psnr").size(), 24u);  // every second frame of 48
    ASSERT_EQ(ffmpeg.scores.at("xpsnr").size(), 24u);
    EXPECT_NE(ffmpeg.implementations.at("psnr"), "libvmaf");

    // In the app: the frames from the GPU's decoder, from the software
    // decoder, or through FFmpeg's pipes.
    const struct {
        const char* name;
        bool gpu_decode;
        const char* software_decode;
        const char* status;
    } routes[] = {
        {"the GPU's decoder", true, "", "Running PSNR, SSIM and XPSNR in the app (GPU decode: source cuda"},
        {"the software decoder", false, "", "Running PSNR, SSIM and XPSNR in the app (GPU decode: off)"},
        {"FFmpeg's pipes", false, "ffmpeg", "Running ffmpeg, PSNR, SSIM and XPSNR in the app"},
    };
    int compared = 0;
    for (const auto& way : routes) {
        const Route app = run(way.gpu_decode, "", way.software_decode);
        if (!said(app, way.status)) {
            continue;  // this PC has no such route (no NVIDIA decoder, no software decoder)
        }
        SCOPED_TRACE(way.name);
        EXPECT_FALSE(said(app, "Decoding in the app failed") || said(app, "GPU decoding failed"));
        for (const std::string key : {"psnr", "ssim", "xpsnr"}) {
            EXPECT_EQ(app.implementations.at(key), "libvmaf") << key;
            EXPECT_EQ(app.scores.at(key), ffmpeg.scores.at(key)) << key;
        }
        ++compared;
    }
    EXPECT_GE(compared, 1);  // FFmpeg's pipes are there wherever FFmpeg is
}

}  // namespace
}  // namespace vml
