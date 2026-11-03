// The command line reads its arguments, finds its test videos and says where
// each metric is calculated as the Python app's does (reference/cli.json,
// made by its argparse parser and its expand_tests, and devices.json, by its
// devices command).
#include "arguments.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "bitrate.h"
#include "compare.h"
#include "devices.h"
#include "reference.h"
#include "test_videos.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"

namespace vml::cli {
namespace {

using Json = nlohmann::json;

// What a command line read, as the Python app's argparse namespace holds
// it.
Json values_of(const Arguments& parsed) {
    if (parsed.command == Command::devices) {
        return Json{{"command", "devices"}};
    }
    const CompareArguments& a = parsed.compare;
    const auto either = [](const auto& value) { return value ? Json(*value) : Json(nullptr); };
    const auto path = [](const std::optional<std::string>& text) {
        return text ? Json(platform::path_text(python_path(*text))) : Json(nullptr);
    };
    return Json{
        {"command", "compare"},
        {"reference", platform::path_text(python_path(a.reference))},
        {"tests", a.tests},
        {"metrics", either(a.metrics)},
        {"model", a.model},
        {"model_v1", a.model_v1},
        {"black_bars", a.black_bars},
        {"scale_to", a.scale_to},
        {"scaler", a.scaler},
        {"duration", either(a.duration)},
        {"range", a.range ? Json::array({a.range->first, a.range->second}) : Json(nullptr)},
        {"colour", either(a.colour)},
        {"film_grain", either(a.film_grain)},
        {"fps", either(a.fps)},
        {"size", either(a.size)},
        {"pix_fmt", either(a.pix_fmt)},
        {"cvvdp_displays", a.cvvdp_displays.empty() ? Json(nullptr) : Json(a.cvvdp_displays)},
        {"cvvdp_fill", a.cvvdp_fill},
        {"subsample", a.subsample},
        {"threads", either(a.threads)},
        {"vmaf_on", either(a.vmaf_on)},
        {"vship", either(a.vship)},
        {"vship_pass", either(a.vship_pass)},
        {"cpu", a.cpu},
        {"no_gpu_decode", a.no_gpu_decode},
        {"gpu_decode", either(a.gpu_decode)},
        {"parallel", either(a.parallel)},
        {"recalculate", a.recalculate},
        {"csv", path(a.csv)},
        {"json_path", path(a.json_path)},
        {"quiet", a.quiet},
        {"verbose", a.verbose},
    };
}

// Python's text, with the C++ app's name and version for its own.
std::string with_version(std::string text) {
    const std::string version = std::string("VideoMetricsLab ") + VML_VERSION;
    for (std::size_t at = text.find("{version}"); at != std::string::npos; at = text.find("{version}", at)) {
        text.replace(at, 9, version);
    }
    return text;
}

TEST(CommandLine, ReadsArgumentsAsThePythonApp) {
    const Json reference = test::load_reference("cli.json");
    for (const Json& expected : reference["arguments"]) {
        SCOPED_TRACE(expected["argv"].dump());
        const int exit_code = expected["exit"];
        try {
            const Arguments parsed = parse_arguments(expected["argv"].get<std::vector<std::string>>());
            if (parsed.print_and_exit) {
                EXPECT_EQ(exit_code, 0);
                EXPECT_EQ(*parsed.print_and_exit, with_version(expected["stdout"]));
            } else if (parsed.command == Command::none) {
                // main() prints the help, and exits with 2.
                EXPECT_EQ(exit_code, 2);
                EXPECT_EQ(top_help(), with_version(expected["stdout"]));
            } else {
                Json values = expected["values"];
                for (const char* number : {"fps"}) {
                    if (values.is_object() && values.contains(number)) {
                        const std::optional<double> value = test::reference_number(values[number]);
                        values[number] = value ? Json(*value) : Json(nullptr);
                    }
                }
                EXPECT_EQ(values_of(parsed), values);
            }
        } catch (const UsageError& error) {
            EXPECT_EQ(exit_code, 2);
            EXPECT_EQ(std::string(error.what()) + "\n", with_version(expected["stderr"]));
        }
    }
}

TEST(CommandLine, FindsTestVideosAsThePythonApp) {
    const Json reference = test::load_reference("cli.json")["patterns"];
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() / ("vml_patterns_test_" + std::to_string(std::random_device{}()));
    for (const Json& file : reference["files"]) {
        const std::string name = file.get<std::string>();
        const std::filesystem::path path = folder / platform::path_from_text(name);
        std::filesystem::create_directories(name.ends_with("/") ? path : path.parent_path());
        if (!name.ends_with("/")) {
            std::ofstream(path).close();
        }
    }
    for (const Json& expected : reference["cases"]) {
        SCOPED_TRACE(expected["tests"].dump());
#ifndef _WIN32
        if (expected.value("windows", false)) {
            continue;  // names ignore case on Windows only
        }
#endif
        std::vector<std::string> tests;
        for (const Json& test : expected["tests"]) {
            tests.push_back(platform::path_text(folder / platform::path_from_text(test.get<std::string>())));
        }
        // Each path found, within the folder, with "/" as Python's as_posix() writes it.
        std::vector<std::string> found;
        for (const std::filesystem::path& path : expand_tests(tests, folder / "ref.mkv")) {
            std::string text = platform::path_text(path.lexically_relative(folder));
#ifdef _WIN32
            std::replace(text.begin(), text.end(), '\\', '/');
#endif
            found.push_back(text);
        }
        EXPECT_EQ(found, expected["found"].get<std::vector<std::string>>());
    }
    std::error_code error;
    std::filesystem::remove_all(folder, error);
}

// The devices command says where each metric is calculated as the Python
// app's says it (devices.json, made by its devices command with stand-ins
// for what it checks), after the app's name and version, each app's own.
TEST(CommandLine, SaysWhereEachMetricIsCalculatedAsThePythonApp) {
    const auto tool = [](const Json& status) {
        ToolStatus found{status["name"].get<std::string>(), status["path"].get<std::string>(),
                         status["runnable"].get<bool>(), std::nullopt, status["error"].get<std::string>()};
        if (!status["version"].is_null()) {
            found.version = status["version"].get<std::vector<int>>();
        }
        return found;
    };
    for (const Json& expected : test::load_reference("devices.json")) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        DeviceFindings found;
        found.tools = ToolsStatus{tool(expected["tools"][0]), tool(expected["tools"][1])};
        if (const Json& vship = expected["vship"]; !vship.is_null()) {
            found.vship = VshipDevice{vship["backend"].get<std::string>(), vship["name"].get<std::string>(),
                                      vship["gpu_id"].get<int>(), vship["version"].get<std::string>(), std::nullopt};
        }
        found.no_vship = expected["no_vship"].get<std::string>();
        found.vmaf.backend = expected["vmaf"]["backend"].get<std::string>();
        if (!expected["vmaf"]["device"].is_null()) {
            found.vmaf.device = expected["vmaf"]["device"].get<int>();
        }
        found.vmaf.text = expected["vmaf"]["text"].get<std::string>();
        found.v1 = V1Probe{expected["v1"]["available"].get<bool>(), expected["v1"]["text"].get<std::string>()};
        found.cpu_metrics_in_app = expected["cpu_metrics_in_app"].get<bool>();
        found.libvmaf_fast_version = expected["libvmaf_fast_version"].get<std::string>();
        // The reference was made on Windows, where FFmpeg is ffmpeg.exe.
        std::vector<std::string> lines;
        for (std::string line : expected["lines"].get<std::vector<std::string>>()) {
            if (const std::size_t at = line.find("ffmpeg.exe"); at != std::string::npos) {
                line.replace(at, 10, platform::executable_name("ffmpeg"));
            }
            lines.push_back(line);
        }
        EXPECT_EQ(device_lines(found), lines);
    }
}

// The command's options over the app's settings, as the window's controls
// set them: --gpu-decode turns GPU decoding on where Settings have it off,
// --model-v1 picks the window's choice, --cvvdp-fill fills every display
// (the Automatic one too), and without --film-grain Settings' default holds.
TEST(CommandLine, OptionsTakeThePlaceOfTheAppsSettings) {
    Settings settings;
    settings.default_gpu_decode = false;
    settings.film_grain_default = "without";
    CompareArguments arguments;
    EXPECT_FALSE(row_options(arguments, settings, {"vmaf"}).gpu_decode);
    EXPECT_EQ(film_grain_choice(arguments, settings), false);
    arguments.gpu_decode = "intel";
    arguments.model_v1 = "vmaf_v1_3d0h_2160";
    arguments.film_grain = "with";
    const VmafOptions options = row_options(arguments, settings, {"vmaf_v1"});
    EXPECT_TRUE(options.gpu_decode);
    EXPECT_EQ(options.gpu_vendor, GpuVendor::intel);
    EXPECT_EQ(options.model_choice_v1, builtin_choice("vmaf_v1_3d0h_2160"));
    EXPECT_EQ(film_grain_choice(arguments, settings), true);
    settings.film_grain_default = "ask";
    EXPECT_EQ(film_grain_choice(CompareArguments{}, settings), std::nullopt);

    arguments.cvvdp_fill = true;
    arguments.cvvdp_displays = {"6-inch phone"};
    EXPECT_TRUE(cvvdp_displays(arguments, settings).front().second.resize_to_display);
    EXPECT_TRUE(display_for({}, VideoInfo{}, true).second.resize_to_display);
    EXPECT_FALSE(display_for({}, VideoInfo{}, false).second.resize_to_display);
}

// --gpu-decode naming a maker whose GPU this PC lacks is refused where
// Windows lists the GPUs, and not on a Mac or Linux PC.
TEST(CommandLine, RefusesAGpuMakerThePcLacks) {
    const std::vector<GpuVendor> found{GpuVendor::nvidia, GpuVendor::intel};
    EXPECT_EQ(missing_gpu_maker("amd", found, true), "--gpu-decode amd: this PC has no AMD GPU (it has NVIDIA, Intel).");
    EXPECT_EQ(missing_gpu_maker("intel", found, true), std::nullopt);
    EXPECT_EQ(missing_gpu_maker("auto", {}, true), std::nullopt);
    EXPECT_EQ(missing_gpu_maker(std::nullopt, {}, true), std::nullopt);
    EXPECT_EQ(missing_gpu_maker("amd", {}, false), std::nullopt);
}

// The bitrate command: its options (--no-adjust-start only with --csv, whose
// times it sets), its table of the videos read, and each video's three CSV
// files under one name free for all three, a second run's beside them.
TEST(CommandLine, ReadsTheBitrateCommandAndWritesItsFiles) {
    const Arguments parsed =
        parse_arguments({"bitrate", "a.mkv", "b*.266", "--fps", "24000/1001", "--csv", "out", "--no-adjust-start"});
    ASSERT_EQ(parsed.command, Command::bitrate);
    EXPECT_EQ(parsed.bitrate.videos, (std::vector<std::string>{"a.mkv", "b*.266"}));
    EXPECT_DOUBLE_EQ(*parsed.bitrate.fps, 24000.0 / 1001);
    EXPECT_EQ(parsed.bitrate.csv, "out");
    EXPECT_TRUE(parsed.bitrate.no_adjust_start);
    EXPECT_FALSE(parsed.bitrate.json_path || parsed.bitrate.quiet);
    EXPECT_THROW(parse_arguments({"bitrate", "a.mkv", "--no-adjust-start"}), UsageError);
    EXPECT_THROW(parse_arguments({"bitrate"}), UsageError);
    EXPECT_THROW(parse_arguments({"bitrate", "a.mkv", "--metrics", "vmaf"}), UsageError);

    // 4,000 bytes in 2 s: 1,500 in the first second, 2,500 in the second
    BitrateVideo read{"clip.mkv", VideoInfo{},
                      bitrate_data("clip.mkv", {0.0, 0.5, 1.0, 1.5}, {0.5, 0.5, 0.5, 0.5}, {1000, 500, 2000, 500},
                                   {true, false, true, false}),
                      {}};
    read.info->codec_name = "hevc";
    const BitrateVideo missing{"gone.mkv", std::nullopt, std::nullopt, "not found"};
    EXPECT_EQ(bitrate_table({read, missing}),
              "  File      Codec    Duration  Frames  Average Mb/s  Minimum Mb/s  Maximum Mb/s\n"
              "  clip.mkv   HEVC  0:00:02.00       4         0.016         0.012         0.020\n");
    const nlohmann::ordered_json json = bitrate_json({read, missing});
    EXPECT_EQ(json["format"], "bitrate-1");
    EXPECT_EQ(json["videos"][0]["bytes"], 4000);
    EXPECT_EQ(json["videos"][0]["average_kbps"], 16.0);
    EXPECT_EQ(json["videos"][1]["error"], "not found");

    const test::Folder folder;
    const std::filesystem::path out = folder / "out";
    std::filesystem::create_directories(out);
    for (const std::string stem : {"clip", "clip_2"}) {
        std::set<std::filesystem::path> reserved;  // a run of its own: the first run's files are on disk
        const std::vector<std::filesystem::path> written = write_bitrate_csvs(read, out, true, reserved);
        ASSERT_EQ(written.size(), 3U);
        EXPECT_EQ(written[0], out / (stem + "_bitrate_frames.csv"));
        EXPECT_EQ(written[1], out / (stem + "_bitrate_seconds.csv"));
        EXPECT_EQ(written[2], out / (stem + "_bitrate_gops.csv"));
    }
    std::ifstream seconds(out / "clip_bitrate_seconds.csv", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(seconds)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text, "start_s,end_s,kbps\r\n0.000000,1.000000,12.0\r\n1.000000,2.000000,20.0\r\n");
}

#ifdef __APPLE__
// A Mac decodes with its own media engine: a maker named is refused, auto
// and none are not.
TEST(CommandLine, AMacRefusesAGpuMaker) {
    EXPECT_EQ(mac_gpu_decode_refusal("nvidia"),
              "--gpu-decode nvidia: a Mac decodes with its own GPU (VideoToolbox); use --gpu-decode auto.");
    EXPECT_TRUE(mac_gpu_decode_refusal("intel"));
    EXPECT_TRUE(mac_gpu_decode_refusal("amd"));
    EXPECT_FALSE(mac_gpu_decode_refusal("auto"));
    EXPECT_FALSE(mac_gpu_decode_refusal(std::nullopt));
}
#endif

}  // namespace
}  // namespace vml::cli
