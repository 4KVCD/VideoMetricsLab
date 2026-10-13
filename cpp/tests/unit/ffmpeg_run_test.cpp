// FFmpeg's filter graph and command, and the reading of its logs, held to
// the Python app's own (runs.json, from vmaf_runner.py).
#include "vml/ffmpeg_run.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "reference.h"
#include "vml/frame_sync.h"
#include "vml/platform/system.h"

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

VideoInfo video(const char* name, int width, int height, const char* pix_fmt, double fps) {
    VideoInfo info;
    info.path = name;
    info.width = width;
    info.height = height;
    info.pix_fmt = pix_fmt;
    info.fps = fps;
    info.duration = 10.0;
    info.codec_name = "hevc";
    return info;
}

// The cases' parts, as make_reference.py's run_cases names them.
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

TEST(FfmpegRun, BuildsThePythonAppsFilterGraphAndCommand) {
    const Json reference = test::load_reference("runs.json");
    ASSERT_FALSE(reference["commands"].empty());
    for (const Json& item : reference["commands"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        set_pairing(item["pairing"].get<std::string>());
        const auto [source, test] = pair_named(item["pair"].get<std::string>());
        const auto [source_crop, test_crop] = crops_named(item["crop"].get<std::string>());
        const HardwareDecoding plan = plan_named(item["plan"].get<std::string>());
        const VmafOptions options = options_named(item["choice"].get<std::string>(), item["setting"].get<std::string>());
        const std::optional<std::string> xpsnr_log =
            options.compute_xpsnr ? std::optional<std::string>("xpsnr_log.txt") : std::nullopt;
        const std::string graph = build_filtergraph(source, test, options, source_crop, test_crop, plan,
                                                    "vmaf_log.json", item["model"].get<std::string>(), xpsnr_log,
                                                    item["cores"].get<unsigned>());
        std::vector<std::string> command =
            build_ffmpeg_command("<ffmpeg>", test.path, source.path, graph, plan, options.duration_limit);
        // The inputs as absolute paths, which differ between PCs.
        for (std::string& part : command) {
            if (part == platform::path_text(std::filesystem::absolute(test.path))) {
                part = "<test>";
            } else if (part == platform::path_text(std::filesystem::absolute(source.path))) {
                part = "<source>";
            }
        }
        EXPECT_EQ(command, item["command"].get<std::vector<std::string>>());
    }
    set_pairing("");
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

}  // namespace
}  // namespace vml
