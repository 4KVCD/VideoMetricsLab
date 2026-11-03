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

#include "bdrate.h"
#include "bitrate.h"
#include "compare.h"
#include "devices.h"
#include "reference.h"
#include "report.h"
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
        {"vvc_decoder", either(a.vvc_decoder)},
        {"parallel", either(a.parallel)},
        {"recalculate", a.recalculate},
        {"difference", a.difference},
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
// --vvc-decoder takes the place of Settings' VVC decoder, --model-v1 picks
// the window's choice, --cvvdp-fill fills every display (the Automatic one
// too), and without --film-grain Settings' default holds.
TEST(CommandLine, OptionsTakeThePlaceOfTheAppsSettings) {
    Settings settings;
    settings.default_gpu_decode = false;
    settings.film_grain_default = "without";
    settings.vvc_decode_on_gpu = false;
    CompareArguments arguments;
    EXPECT_FALSE(row_options(arguments, settings, {"vmaf"}).gpu_decode);
    EXPECT_FALSE(row_options(arguments, settings, {"vmaf"}).vvc_on_gpu);
    arguments.vvc_decoder = "gpu";
    EXPECT_TRUE(row_options(arguments, settings, {"vmaf"}).vvc_on_gpu);
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

// --difference: exactly two test videos, and the first minus the second in
// the table and the JSON, frame by frame and CVVDP second by second, with a
// metric only one video has noted.
TEST(CommandLine, GivesTheDifferenceOfTwoVideos) {
    EXPECT_EQ(difference_refusal(3), "argument --difference: needs exactly two test videos, got 3");
    EXPECT_TRUE(difference_refusal(1));
    EXPECT_FALSE(difference_refusal(2));

    std::vector<TestVideo> videos(2);
    for (TestVideo& video : videos) {
        video.metrics = {"vmaf", "psnr", "cvvdp"};
        video.result.emplace().compared_frame_count = 3;
    }
    videos[0].path = "a.mkv";
    videos[0].result->metric_results.add(FrameMetricResult{"vmaf", {0, 1, 2}, {0.0, 0.5, 1.0}, {90, 80, 70}, {}});
    videos[0].result->metric_results.add(FrameMetricResult{"psnr", {0, 1, 2}, {0.0, 0.5, 1.0}, {40, 41, 42}, {}});
    videos[0].result->metric_results.add(SequenceMetricResult{"cvvdp", 8.5, {}, {0, 2}, {0.0, 1.0}, {8.0f, 9.0f}});
    videos[1].path = "b.mkv";  // no PSNR; VMAF from frame 1 on
    videos[1].result->metric_results.add(FrameMetricResult{"vmaf", {1, 2, 3}, {0.5, 1.0, 1.5}, {75, 72, 60}, {}});
    videos[1].result->metric_results.add(SequenceMetricResult{"cvvdp", 8.0, {}, {0, 2}, {0.0, 1.0}, {7.5f, 9.5f}});
    const std::vector<Summaries> summaries = summarize(videos);
    const std::optional<nlohmann::ordered_json> difference = difference_metrics(videos, summaries);
    ASSERT_TRUE(difference);
    EXPECT_EQ(difference->dump(),
              R"({"vmaf":{"score":11.0,"mean":1.5,"minimum":-2.0,"maximum":5.0,"paired":2},)"
              R"("cvvdp":{"score":0.5,"mean":0.0,"minimum":-0.5,"maximum":0.5,"paired":2}})");
    VideoInfo source;
    source.path = "ref.mkv";
    const std::string table = results_table(source, videos, summaries, false, difference);
    const std::string section = table.substr(table.find("Difference"));
    EXPECT_EQ(section,
              "Difference: a.mkv minus b.mkv, per frame (CVVDP per second)\n"
              "  Metric        Score    Mean     Min     Max  Paired\n"
              "  VMAF v0.6.1  +11.00   +1.50   -2.00   +5.00       2\n"
              "  CVVDP        +0.500  +0.000  -0.500  +0.500       2\n"
              "  note: scored for one video only: PSNR (a.mkv)");
    const nlohmann::ordered_json document = results_document(source, videos, summaries, false, difference);
    EXPECT_EQ(document["difference"]["first"], "a.mkv");
    EXPECT_EQ(document["difference"]["second"], "b.mkv");
    EXPECT_EQ(document["difference"]["metrics"], *difference);
    videos[1].result.reset();
    EXPECT_FALSE(difference_metrics(videos, summarize(videos)));
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

// bitrate --difference: the first video's bitrate minus the second's, from
// each stream's start: the averages (and in %) and the seconds both have, in
// the table and the JSON, and as CSV files at the first's frames and over
// time; none when a video was not read.
TEST(CommandLine, GivesTheBitrateDifferenceOfTwoVideos) {
    EXPECT_TRUE(parse_arguments({"bitrate", "a.mkv", "b.mkv", "--difference"}).bitrate.difference);
    // a: 12 then 20 kb/s, 16 on average; b: 8 kb/s for one second, 10 s into its file
    BitrateVideo a{"a.mkv", VideoInfo{},
                   bitrate_data("a.mkv", {0.0, 0.5, 1.0, 1.5}, {0.5, 0.5, 0.5, 0.5}, {1000, 500, 2000, 500},
                                {true, false, true, false}),
                   {}};
    BitrateVideo b{"b.mkv", VideoInfo{}, bitrate_data("b.mkv", {10.0, 10.5}, {0.5, 0.5}, {500, 500}, {true, false}),
                   {}};
    a.info->codec_name = b.info->codec_name = "hevc";
    const std::optional<BitrateDifference> difference = bitrate_difference({a, b});
    ASSERT_TRUE(difference && difference->seconds);
    EXPECT_DOUBLE_EQ(difference->average_kbps, 8);
    EXPECT_DOUBLE_EQ(*difference->average_percent, 100);
    EXPECT_DOUBLE_EQ(difference->seconds->mean, 4);  // second 0 only: 12 - 8
    EXPECT_EQ(difference->seconds->paired, 1);

    const std::string table = bitrate_table({a, b}, difference);
    EXPECT_NE(table.find("Difference: a.mkv minus b.mkv, per second from each stream's start"), std::string::npos);
    EXPECT_NE(table.find("Average Mb/s  Average %  Mean Mb/s  Min Mb/s  Max Mb/s  Seconds"), std::string::npos);
    EXPECT_NE(table.find("\n  +0.008          +100.00     +0.004    +0.004    +0.004        1\n"), std::string::npos)
        << table;
    const nlohmann::ordered_json json = bitrate_json({a, b}, difference)["difference"];
    EXPECT_EQ(json["second"], "b.mkv");
    EXPECT_EQ(json["average_kbps"], 8.0);
    EXPECT_EQ(json["average_percent"], 100.0);
    EXPECT_EQ(json["seconds"]["maximum_kbps"], 4.0);
    EXPECT_EQ(json["seconds"]["paired"], 1);
    const BitrateVideo missing{"gone.mkv", std::nullopt, std::nullopt, "not found"};
    EXPECT_FALSE(bitrate_difference({a, missing}));
    EXPECT_FALSE(bitrate_difference({a}));

    const test::Folder folder;
    const std::filesystem::path out = folder / "out";
    std::filesystem::create_directories(out);
    std::set<std::filesystem::path> reserved;
    const std::vector<std::filesystem::path> written = write_difference_csvs(a, b, out, true, reserved);
    ASSERT_EQ(written.size(), 3U);
    EXPECT_EQ(written[0], out / "a_minus_b_bitrate_frames.csv");
    const auto text_of = [](const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    };
    EXPECT_EQ(text_of(written[0]),
              "frame,time_s,size_bytes\r\n0,0.000000,500\r\n1,0.500000,0\r\n2,1.000000,\r\n3,1.500000,\r\n");
    EXPECT_EQ(text_of(written[1]), "start_s,end_s,kbps\r\n0.000000,1.000000,4.0\r\n1.000000,2.000000,\r\n");
}

// bdrate: two CSV files with a row per encode (a BOM, a quoted cell, an
// empty score, columns in any case); s1's test at half the rate everywhere
// (-50 %), s2's the same as its anchor (0) but for a PSNR not scored (n/a,
// with a warning), s3 in the anchor's file only (left out); a mean and
// median per class; and the refusals.
TEST(CommandLine, CalculatesBdRatesFromTwoCsvFiles) {
    const Arguments parsed =
        parse_arguments({"bdrate", "--anchor", "a.csv", "--test", "b.csv", "--group-by", "Class"});
    ASSERT_EQ(parsed.command, Command::bdrate);
    BdrateArguments arguments = parsed.bdrate;
    try {
        parse_arguments({"bdrate", "--anchor", "a.csv"});
        ADD_FAILURE() << "no --test was taken";
    } catch (const UsageError& error) {
        EXPECT_NE(std::string(error.what()).find("the following arguments are required: --test"), std::string::npos);
    }
    EXPECT_THROW(parse_arguments({"bdrate", "--anchor", "a", "--test", "b", "--weights", "vmaf=1"}), UsageError);

    const RateTable anchor = read_rate_table("\xEF\xBB\xBFSequence,Class,KBPS,vmaf,psnr\r\n"
                                             "s1,\"A, B\",1000,30,30\r\ns1,\"A, B\",2000,33,33\r\n"
                                             "s1,\"A, B\",4000,36,36\r\ns1,\"A, B\",8000,39,39\r\n"
                                             "s2,C,1000,30,30\r\ns2,C,2000,33,33\r\ns2,C,4000,36,36\r\n"
                                             "s2,C,8000,39,39\r\ns3,C,1000,30,30\r\ns3,C,2000,33,33\r\n",
                                             "a.csv");
    const RateTable test = read_rate_table("sequence,kbps,vmaf,psnr\n"
                                           "s1,500,30,30\ns1,1000,33,33\ns1,2000,36,36\ns1,4000,39,39\n"
                                           "s2,1000,30,30\ns2,2000,33,\ns2,4000,36,36\ns2,8000,39,39\n",
                                           "b.csv");
    EXPECT_EQ(anchor.metric_keys, (std::vector<std::string>{"vmaf", "psnr"}));
    EXPECT_EQ(anchor.encodes[0].text.at("class"), "A, B");

    const BdRateReport report = bd_rate_report(anchor, test, bd_rate_options(arguments));
    EXPECT_EQ(std::ranges::count(report.warnings, "s3: only in the anchor, left out"), 1);
    EXPECT_EQ(std::ranges::count(report.warnings, "s2: psnr: not scored at every encode, n/a"), 1);
    const std::string table = bd_rate_table(report, "a.csv", "b.csv");
    for (const char* line : {"BD-rate of b.csv against a.csv, 2 sequences (- is less bitrate for the same score)\n",
                             "\n| | VMAF v0.6.1 | PSNR |\n|---|---|---|\n", "\n| mean | -25.0% | -50.0% |\n",
                             "\n| A, B mean | -50.0% | -50.0% |\n", "\n| C median | +0.0% | n/a |\n",
                             "\n| s2 | +0.0% | n/a |\n"}) {
        EXPECT_NE(table.find(line), std::string::npos) << line << table;
    }
    const std::string csv = bd_rate_csv(report);
    EXPECT_TRUE(csv.starts_with("\xEF\xBB\xBFrow,class,anchor_encodes,test_encodes,vmaf,psnr\r\nmean,,,,")) << csv;
    EXPECT_NE(csv.find("\r\ns2,C,4,4,0.0,\r\n"), std::string::npos) << csv;
    const nlohmann::ordered_json json = bd_rate_json(report, "a.csv", "b.csv");
    EXPECT_EQ(json["format"], "bdrate-1");
    EXPECT_NEAR(json["mean"]["bd_rate"]["vmaf"].get<double>(), -25, 1e-9);
    EXPECT_FALSE(json["mean"].contains("W"));
    EXPECT_NEAR(json["sequences"][0]["bd_rate"]["vmaf"].get<double>(), -50, 1e-9);
    EXPECT_TRUE(json["sequences"][1]["bd_rate"]["psnr"].is_null());
    EXPECT_EQ(json["sequences"][1]["why"]["psnr"], "not scored at every encode");
    EXPECT_EQ(json["groups"][0]["group"], "A, B");

    arguments.group_by = "vmaf";  // a score column, not a group
    EXPECT_THROW(bd_rate_report(anchor, test, bd_rate_options(arguments)), std::runtime_error);
    try {
        read_rate_table("sequence,kbps\r\ns1,fast\r\n", "a.csv");
        ADD_FAILURE() << "a rate that is not a number was taken";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "a.csv: line 2: kbps is not a number: 'fast'");
    }
    EXPECT_THROW(read_rate_table("sequence,vmaf\ns1,30\n", "a.csv"), std::runtime_error);

    // As a spreadsheet saves it: two unnamed empty columns, a row of empty
    // fields after the data
    const RateTable saved = read_rate_table("sequence,kbps,vmaf,,\r\ns9,1000,30,,\r\n,,,,\r\n", "a.csv");
    EXPECT_EQ(saved.columns, (std::vector<std::string>{"sequence", "kbps", "vmaf"}));
    EXPECT_EQ(saved.encodes.size(), 1u);
    // No sequence in both: no sequences, and the warnings name them
    const BdRateReport apart = bd_rate_report(saved, test, BdRateOptions{});
    EXPECT_TRUE(apart.sequences.empty());
    EXPECT_EQ(std::ranges::count(apart.warnings, "s9: only in the anchor, left out"), 1);
    EXPECT_EQ(std::ranges::count(apart.warnings, "s2: only in the test, left out"), 1);
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
