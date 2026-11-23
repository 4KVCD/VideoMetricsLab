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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "batch_file.h"
#include "bdrate.h"
#include "bitrate.h"
#include "common.h"
#include "compare.h"
#include "devices.h"
#include "file_names.h"
#include "reference.h"
#include "report.h"
#include "test_videos.h"
#include "vml/metric_cache.h"
#include "watch.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"
#include "vml/run_file.h"
#include "weak_report.h"

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
        {"reference", platform::path_text(python_path(a.reference.value_or("")))},
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

// Python's text, with the C++ app's name and version for its own
// ({app_version}: --name's help has a {version} of its own).
std::string with_version(std::string text) {
    const std::string placeholder = "{app_version}";
    const std::string version = std::string("VideoMetricsLab ") + VML_VERSION;
    for (std::size_t at = text.find(placeholder); at != std::string::npos; at = text.find(placeholder, at)) {
        text.replace(at, placeholder.size(), version);
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

// HDR VMAF: its model named by its word, and left out of "all", which would
// otherwise fail every SDR video's run.
TEST(CommandLine, ReadsHdrVmafAndLeavesItOutOfAll) {
    const CompareArguments all = parse_arguments({"compare", "ref.mkv", "a.mkv", "-m", "all"}).compare;
    ASSERT_TRUE(all.metrics);
    EXPECT_EQ(std::count(all.metrics->begin(), all.metrics->end(), "vmaf_hdr"), 0);
    EXPECT_EQ(std::count(all.metrics->begin(), all.metrics->end(), "vmaf_v1"), 1);
    const CompareArguments hdr =
        parse_arguments({"compare", "ref.mkv", "a.mkv", "-m", "vmaf_hdr", "--model-hdr", "hdr_vmaf_5d0h"}).compare;
    EXPECT_EQ(*hdr.metrics, std::vector<std::string>{"vmaf_hdr"});
    EXPECT_EQ(hdr.model_hdr, "hdr_vmaf_5d0h");
    EXPECT_THROW(parse_arguments({"compare", "ref.mkv", "a.mkv", "--model-hdr", "vmaf_v1_3d0h"}), UsageError);
}

// --worst and --segments (issue #24, which the Python app never had): the
// conditions read, each named metric checked, and the segments' options
// refused without --segments.
TEST(CommandLine, ReadsWorstFramesAndSegments) {
    const CompareArguments read =
        parse_arguments({"compare", "ref.mkv", "a.mkv", "--worst", "5", "--segments", " VMAF<90, butteraugli>2.5",
                         "--segments-all", "--segment-min-frames", "3", "--segment-merge-frames", "0",
                         "--segments-dir", "out"})
            .compare;
    EXPECT_EQ(read.worst, 5);
    ASSERT_EQ(read.segments.size(), 2U);
    EXPECT_EQ(read.segments[0].text(), "vmaf<90");
    EXPECT_EQ(read.segments[1].text(), "butteraugli>2.5");
    EXPECT_TRUE(read.segments_all);
    EXPECT_EQ(read.segment_min_frames, 3);
    EXPECT_EQ(read.segment_merge_frames, 0);
    EXPECT_EQ(read.segments_dir, "out");
    const auto error_of = [](const std::vector<std::string>& arguments) {
        try {
            parse_arguments(arguments);
        } catch (const UsageError& error) {
            const std::string text = error.what();
            return text.substr(text.find("error: "));
        }
        return std::string("no error");
    };
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--segments-dir", "out"}),
              "error: argument --segments-dir: only with --segments, whose segments it shapes");
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--segments", "vmaf=90"}),
              "error: argument --segments: invalid condition 'vmaf=90': write METRIC<VALUE or METRIC>VALUE, such as "
              "vmaf<90");
    EXPECT_TRUE(error_of({"compare", "ref.mkv", "a.mkv", "--segments", "vmav<90"})
                    .starts_with("error: argument --segments: unknown metric vmav; choose from: "));
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--segments-report", "r.json"}),
              "error: argument --segments-report: only with --segments, whose segments it shapes");
}

// --segments saved: the conditions last set in the window's Weak segments,
// those on metrics not compared left out (and said), its options where the
// command gives none; refused when none is left. --segments-report writes
// every video's segments in one file, JSON or CSV by its name.
TEST(CommandLine, SegmentsSavedTakesTheWindowsRulesAndReportsEveryVideo) {
    CompareArguments arguments = parse_arguments({"compare", "ref.mkv", "a.mkv", "--segments", " Saved",
                                                  "--segment-merge-frames", "1", "--segments-report", "r.csv"})
                                     .compare;
    EXPECT_TRUE(arguments.segments_saved);
    EXPECT_TRUE(arguments.segments.empty());
    EXPECT_EQ(arguments.segments_report, "r.csv");
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<80", "ssimulacra2<60"};
    settings.weak_segments_all = true;
    settings.weak_segment_min_frames = 2;
    settings.weak_segment_merge_frames = 5;
    EXPECT_EQ(take_saved_segments(arguments, settings, {"vmaf", "psnr"}), std::vector<std::string>{"ssimulacra2<60"});
    ASSERT_EQ(arguments.segments.size(), 1U);
    EXPECT_EQ(arguments.segments[0].text(), "vmaf<80");
    EXPECT_TRUE(arguments.segments_all);
    EXPECT_EQ(arguments.segment_min_frames, 2);
    EXPECT_EQ(arguments.segment_merge_frames, 1);  // given here
    // A profile with no condition on the metrics compared takes one on the
    // metric shown, as the Worst Frames tab starts: Metric Graphs' metric
    // where it is compared, else the first (default_weak_condition).
    const auto fallback = [&settings](const std::vector<std::string>& asked) {
        CompareArguments saved = parse_arguments({"compare", "ref.mkv", "a.mkv", "--segments", "saved"}).compare;
        std::optional<WeakCondition> taken;
        take_saved_segments(saved, settings, asked, &taken);
        EXPECT_EQ(saved.segments.size(), 1U);
        return taken ? taken->text() : std::string("none");
    };
    settings.graph_metric = "vmaf";
    EXPECT_EQ(fallback({"psnr"}), default_weak_condition(metric_definition("psnr")).text());
    settings.weak_segment_conditions.clear();
    EXPECT_EQ(fallback({"psnr", "vmaf"}), "vmaf<85");  // the metric shown, compared
    EXPECT_EQ(fallback({"psnr", "ssim"}), default_weak_condition(metric_definition("psnr")).text());

    // profile:NAME: that profile's (any case), not the one in use; a name no
    // profile has is refused with the app's profiles.
    settings.weak_segment_profiles_json = R"([{"name": "Anime", "conditions": ["vmaf<93", "psnr<38"]}])";
    CompareArguments named =
        parse_arguments({"compare", "ref.mkv", "a.mkv", "--segments", "profile: anime"}).compare;
    EXPECT_TRUE(named.segments_saved);
    EXPECT_EQ(named.segments_profile, "anime");
    EXPECT_EQ(take_saved_segments(named, settings, {"vmaf"}), std::vector<std::string>{"psnr<38"});
    EXPECT_EQ(named.segments[0].text(), "vmaf<93");
    // preset:KEY: the preset's, whatever the app's profiles.
    CompareArguments preset =
        parse_arguments({"compare", "ref.mkv", "a.mkv", "--segments", "preset:Grainy-Film"}).compare;
    EXPECT_EQ(preset.segments_preset, "Grainy-Film");
    EXPECT_EQ(take_saved_segments(preset, settings, {"vmaf"}), std::vector<std::string>{"ssimulacra2<55"});
    EXPECT_EQ(preset.segments[0].text(), "vmaf<85");
    EXPECT_EQ(preset.segment_min_frames, 6);
    CompareArguments no_preset =
        parse_arguments({"compare", "ref.mkv", "a.mkv", "--segments", "preset:anime"}).compare;
    try {
        take_saved_segments(no_preset, settings, {"vmaf"});
        ADD_FAILURE() << "taken";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), "--segments preset:anime: no weak segment preset has that name; choose "
                                             "from: animation, cgi, digital, grainy-film, hdr, preview");
    }
    CompareArguments hdr = parse_arguments({"compare", "ref.mkv", "a.mkv", "--segments", "preset:hdr"}).compare;
    try {
        take_saved_segments(hdr, settings, {"vmaf"});
        ADD_FAILURE() << "taken";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
                  "--segments preset:hdr: none of the preset's conditions (cvvdp<9) is on a metric compared (-m)");
    }
    CompareArguments unknown = parse_arguments({"compare", "ref.mkv", "a.mkv", "--segments", "profile:CGI"}).compare;
    try {
        take_saved_segments(unknown, settings, {"vmaf"});
        ADD_FAILURE() << "taken";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
                  "--segments profile:CGI: no weak segment profile has that name; the app's: Default, Anime");
    }

    const test::Folder temporary;
    std::vector<TestVideo> videos(2);
    videos[0].path = "a.mkv";
    videos[0].result.emplace().metric_results.add(
        FrameMetricResult{"vmaf", {0, 1, 2, 3}, {0.0, 0.04, 0.08, 0.12}, {96.0F, 70.0F, 70.0F, 96.0F}, {}});
    videos[0].result->fps = 25.0;
    videos[1].path = "b.mkv";  // not scored: left out
    const std::vector<std::pair<const TestVideo*, std::filesystem::path>> report{{&videos[0], "ref.mkv"},
                                                                               {&videos[1], "ref.mkv"}};
    const SegmentRules rules = segment_rules(arguments);
    ASSERT_EQ(write_segments_report(report, rules, temporary / "r.json"), std::nullopt);
    const nlohmann::json json = nlohmann::json::parse(platform::read_file(temporary / "r.json").value_or(""));
    ASSERT_EQ(json["videos"].size(), 1U);
    EXPECT_EQ(json["videos"][0]["name"], "a.mkv");
    EXPECT_EQ(json["videos"][0]["segment_count"], 1);
    EXPECT_EQ(json["combine"], "all");
    ASSERT_EQ(write_segments_report(report, rules, temporary / "r.CSV"), std::nullopt);
    EXPECT_TRUE(platform::read_file(temporary / "r.CSV").value_or("").starts_with("video,reference,first_frame,"));

}

// A segment's worst score under the heading that names it: a metric whose
// lower scores are better (Butteraugli) has its highest, under "max".
TEST(CommandLine, SegmentsTableNamesEachMetricsWorstEnd) {
    std::vector<TestVideo> videos(1);
    videos[0].path = "a.mkv";
    ComparisonResult& result = videos[0].result.emplace();
    result.fps = 25.0;
    result.metric_results.add(
        FrameMetricResult{"butteraugli", {0, 1, 2, 3}, {0.0, 0.04, 0.08, 0.12}, {1.0F, 2.5F, 3.0F, 1.0F}, {}});
    result.metric_results.add(
        FrameMetricResult{"vmaf", {0, 1, 2, 3}, {0.0, 0.04, 0.08, 0.12}, {96.0F, 70.0F, 60.0F, 96.0F}, {}});
    SegmentRules rules;
    rules.conditions = {*parse_weak_condition("butteraugli>2"), *parse_weak_condition("vmaf<90")};
    rules.all = true;
    rules.min_frames = 1;
    rules.merge_frames = 0;
    const std::string text = segments_text(videos, rules, 10);
    EXPECT_NE(text.find("Butteraugli max"), std::string::npos) << text;
    EXPECT_EQ(text.find("Butteraugli min"), std::string::npos) << text;
    EXPECT_NE(text.find("VMAF v0.6.1 min"), std::string::npos) << text;
    EXPECT_NE(text.find("3.0000"), std::string::npos) << text;
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
    // A pattern kept because it found nothing says so; a missing file named as it is is left to ffprobe.
    EXPECT_EQ(found_nothing(folder / "nothing*.mkv"), "no file matches it");
    EXPECT_EQ(found_nothing(folder / "missing.mkv"), std::nullopt);
    std::error_code error;
    std::filesystem::remove_all(folder, error);
}

// A video named relative to the working folder is recorded from another
// folder too (the saved scores' context.json, read by the window's Saved
// Results), so the videos are made absolute: "..\clips\reference.mkv" was
// saved as typed, and the window called its reference not found.
TEST(CommandLine, VideosAreMadeAbsolute) {
    const std::filesystem::path here = std::filesystem::current_path();
    EXPECT_EQ(absolute_path(python_path("../clips/reference.mkv")), here / ".." / "clips" / "reference.mkv");
    EXPECT_EQ(absolute_path(here / "a.mkv"), here / "a.mkv");
    const std::vector<std::filesystem::path> found = expand_tests({"swap.mkv", "missing*.mkv"}, "reference.mkv");
    ASSERT_EQ(found.size(), 2U);
    EXPECT_EQ(found[0], here / "swap.mkv");
    EXPECT_EQ(found[1], here / "missing*.mkv");
    EXPECT_EQ(found_nothing(found[1]), "no file matches it");
}

// ? and [seq] stand for a character, however many bytes it takes in UTF-8.
TEST(CommandLine, PatternsMatchWholeCharacters) {
    const test::Folder temporary;
    const std::filesystem::path folder = temporary / "videos";
    std::filesystem::create_directory(folder);
    const auto name = [](const char* utf8) { return platform::path_from_text(utf8); };
    for (const char* file : {"\xC3\xA9t\xC3\xA9.mkv", "\xE6\x97\xA5.mkv", "ab.mkv", "ref.mkv"}) {
        std::ofstream(folder / name(file)).close();
    }
    const auto found = [&](const char* pattern) {
        std::vector<std::string> names;
        for (const std::filesystem::path& path :
             expand_tests({platform::path_text(folder / name(pattern))}, folder / "ref.mkv")) {
            names.push_back(platform::path_text(path.filename()));
        }
        return names;
    };
    using Names = std::vector<std::string>;
    EXPECT_EQ(found("?.mkv"), Names{"\xE6\x97\xA5.mkv"});
    EXPECT_EQ(found("?t?.mkv"), Names{"\xC3\xA9t\xC3\xA9.mkv"});
    EXPECT_EQ(found("[\xC3\xA0-\xC3\xAF]t*"), Names{"\xC3\xA9t\xC3\xA9.mkv"});
    EXPECT_EQ(found("[!a\xC3\xA9r]*.mkv"), Names{"\xE6\x97\xA5.mkv"});
#ifdef _WIN32
    EXPECT_EQ(found("\xC3\x89T*"), Names{"\xC3\xA9t\xC3\xA9.mkv"});  // names ignore case in every language
#endif
}

// --csv: a further display's scores beside the video's; a name taken is kept,
// and the new file's name said and given in the JSON.
TEST(CommandLine, CsvFilesKeepWhatIsThereAndSayWhereTheyWent) {
    const test::Folder temporary;
    const std::filesystem::path folder = temporary / "scores";
    std::filesystem::create_directory(folder);
    std::vector<TestVideo> videos(1);
    TestVideo& video = videos[0];
    video.path = "a b.mkv";  // written as a_b
    video.result.emplace().metric_results.add(FrameMetricResult{"vmaf", {0}, {0.0}, {90.0F}, {}});
    TestVideo& other = video.other_displays.emplace_back();
    other.path = video.path;
    other.display = "standard_hdr_pq";
    other.for_display_only = true;
    other.notes = {"calculated on the CPU"};
    const CvvdpSettings hdr = cvvdp_preset_named(std::string(hdr_cvvdp_preset), "")->settings;
    MetricProvenance made_for_hdr;
    const nlohmann::ordered_json hdr_parameters = hdr.parameters();
    for (const auto& [name, value] : hdr_parameters.items()) {
        made_for_hdr.parameters[name] = value.dump();
    }
    other.result.emplace().metric_results.add(SequenceMetricResult{"cvvdp", 8.5, made_for_hdr, {0}, {0.0}, {8.5F}});
    std::ofstream(folder / "a_b.csv") << "an earlier run's";
    std::vector<std::string> said;
    const auto ignore = [](const std::string&) {};
    const auto say = [&said](const std::string& text) { said.push_back(text); };

    ASSERT_TRUE(write_csvs(videos, folder, ignore, say));
    EXPECT_EQ(said, std::vector<std::string>{"note: a_b.csv is there already, so this is written as a_b_2.csv"});
    std::ifstream kept(folder / "a_b.csv");
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(kept), {}), "an earlier run's");
    EXPECT_TRUE(std::filesystem::exists(folder / "a_b_standard_hdr_pq.csv"));

    const VideoInfo source;
    const nlohmann::ordered_json document = results_document(source, videos, summarize(videos), false);
    const nlohmann::ordered_json& entry = document["videos"][0];
    EXPECT_EQ(entry["csv"], platform::path_text(folder / "a_b_2.csv"));
    // Each display's entry: its own CSV and notes (the first's are the
    // video's, on its entry), its parameters and CVVDP's whole summary.
    const nlohmann::ordered_json& first = entry["cvvdp_displays"][0];
    const nlohmann::ordered_json& further = entry["cvvdp_displays"][1];
    EXPECT_FALSE(first.contains("csv"));
    EXPECT_TRUE(first["display_parameters"].is_null());  // no CVVDP score
    EXPECT_EQ(further["csv"], platform::path_text(folder / "a_b_standard_hdr_pq.csv"));
    EXPECT_EQ(further["notes"][0], "calculated on the CPU");
    EXPECT_EQ(further["display_parameters"], nlohmann::ordered_json(hdr.parameters()));
    EXPECT_EQ(further["cvvdp"]["score"], 8.5);
    EXPECT_EQ(further["cvvdp"]["per_second"]["seconds"], 1);
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

// --batch's list: the groups by reference (the same file however written), in the order their references come;
// paths relative to the list's folder; the sequence and other cells kept; what is wrong said with the file's name.
TEST(CommandLine, ReadsABatchList) {
    const std::filesystem::path folder = std::filesystem::path("C:") / "clips";
    const std::vector<BatchGroup> groups = read_batch_file("\xEF\xBB\xBFReference,Test,Sequence,Class\r\n"
                                                           "a/ref.mkv,a/crf20.mkv,A,\"film, dark\"\r\n"
                                                           "b/ref.mkv,b/*.mkv,,game\r\n"
                                                           "./a/ref.mkv,a/crf30.mkv,A,\"film, dark\"\r\n",
                                                           folder, "list.csv");
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].reference, folder / "a" / "ref.mkv");
    ASSERT_EQ(groups[0].rows.size(), 2u);
    EXPECT_EQ(groups[0].rows[0].test, platform::path_text(folder / "a" / "crf20.mkv"));
    EXPECT_EQ(groups[0].rows[1].test, platform::path_text(folder / "a" / "crf30.mkv"));
    EXPECT_EQ(groups[0].rows[1].line, 4u);
    EXPECT_EQ(groups[0].rows[0].sequence, "A");
    EXPECT_EQ(groups[0].rows[0].text.at("class"), "film, dark");
    ASSERT_EQ(groups[1].rows.size(), 1u);
    EXPECT_EQ(groups[1].rows[0].test, platform::path_text(folder / "b" / "*.mkv"));
    EXPECT_EQ(groups[1].rows[0].sequence, "");
    try {
        read_batch_file("reference,video\nref.mkv,a.mkv\n", folder, "list.csv");
        FAIL();
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "list.csv: no test column");
    }
    EXPECT_THROW(read_batch_file("reference,test\nref.mkv,\n", folder, "list.csv"), std::runtime_error);
    EXPECT_THROW(read_batch_file("reference,test\n", folder, "list.csv"), std::runtime_error);
}

// --batch takes the place of REFERENCE and TEST, and cannot give --difference's two videos.
TEST(CommandLine, ReadsBatchAndScoresCsv) {
    const Arguments parsed = parse_arguments({"compare", "--batch", "list.csv", "--scores-csv", "scores.csv"});
    EXPECT_EQ(parsed.compare.batch, "list.csv");
    EXPECT_EQ(parsed.compare.scores_csv, "scores.csv");
    EXPECT_FALSE(parsed.compare.reference);
    for (const std::vector<std::string>& wrong : std::vector<std::vector<std::string>>{
             {"compare", "--batch", "list.csv", "ref.mkv", "a.mkv"},
             {"compare", "--batch", "list.csv", "--difference"},
             {"compare", "ref.mkv"},
         }) {
        SCOPED_TRACE(wrong.back());
        EXPECT_THROW(parse_arguments(wrong), UsageError);
    }
}

// --name (issue #33): a template's fields filled in for each test video, a
// field that is none refused, and every file of the video named after it,
// --run-dir's saved run too, which opens again under that name; the --json
// says where each went. A path's own braces are kept.
TEST(CommandLine, NamesEachVideosFilesAfterTheTemplate) {
    const CompareArguments read = parse_arguments({"compare", "ref.mkv", "a.mkv", "--name", "{test}-VML-{version}",
                                                   "--run-dir", "runs", "--log-file", "run.log"})
                                      .compare;
    EXPECT_EQ(read.name, "{test}-VML-{version}");
    EXPECT_EQ(read.run_dir, "runs");
    EXPECT_EQ(read.log_file, "run.log");
    const auto error_of = [](const std::vector<std::string>& arguments) {
        try {
            parse_arguments(arguments);
        } catch (const UsageError& error) {
            const std::string text = error.what();
            return text.substr(text.find("error: "));
        }
        return std::string("no error");
    };
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--name", "{tset}"}),
              "error: argument --name: {tset} is not a field: use {test}, {reference}, {version}, {date}, {time} or "
              "{sequence}");
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--name", "{test"}),
              "error: argument --name: a { with no } after it (write {{ for a brace)");
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--name", "{sequence}"}),
              "error: argument --name: {sequence} is a --batch row's sequence: only with --batch");
    EXPECT_EQ(error_of({"compare", "--batch", "list.csv", "--name", "{sequence}-{test}"}), "no error");
    EXPECT_TRUE(error_of({"compare", "ref.mkv", "a.mkv", "--name", "runs/{test}"}).starts_with(
        "error: argument --name: it names files, not folders"));

    std::tm started{};
    started.tm_year = 2026 - 1900;
    started.tm_mon = 10;
    started.tm_mday = 7;
    started.tm_hour = 9;
    started.tm_min = 5;
    started.tm_sec = 3;
    NameFields fields = run_fields(started);
    fields.test = "AB-2024-hdr";
    fields.reference = "AB-2024";
    EXPECT_EQ(expand_names("{test}-VML-{version}-{date}-{time}-{{x}}", fields),
              "AB-2024-hdr-VML-" VML_VERSION "-20261107-090503-{x}");
    // A path's braces that make no field stay; a field with no text stays too.
    EXPECT_EQ(expand_names("out{1}/{reference}-{sequence}.json", fields), "out{1}/AB-2024-{sequence}.json");
    EXPECT_EQ(video_field_in("{date}/{reference}.json"), "{reference}");
    EXPECT_EQ(video_field_in("{date}/{x}.json"), std::nullopt);

    const test::Folder temporary;
    std::vector<TestVideo> videos(1);
    TestVideo& video = videos[0];
    video.path = "AB-2024-hdr.mkv";
    video.file_name = expand_names("{test}-VML-{version}", fields);
    video.metrics = {"vmaf"};
    video.result.emplace().metric_results.add(FrameMetricResult{"vmaf", {0, 1}, {0.0, 0.04}, {96.0F, 80.0F}, {}});
    const std::string stem = "AB-2024-hdr-VML-" VML_VERSION;
    std::filesystem::create_directories(temporary / "runs");
    std::ofstream(temporary / "runs" / (stem + ".metrics.json")) << "an earlier run's";
    std::vector<std::string> said;
    const auto ignore = [](const std::string&) {};
    const auto say = [&said](const std::string& text) { said.push_back(text); };

    ASSERT_TRUE(write_csvs(videos, temporary / "scores", ignore, say));
    ASSERT_TRUE(write_run_files(videos, temporary / "runs", ignore, say));
    const SegmentRules rules =
        segment_rules(parse_arguments({"compare", "r.mkv", "a.mkv", "--segments", "vmaf<90"}).compare);
    ASSERT_TRUE(write_segment_files(videos, rules, temporary / "segments", ignore, say));

    EXPECT_EQ(said, std::vector<std::string>{"note: " + stem + ".metrics.json is there already, so this is written as " +
                                             stem + "_2.metrics.json"});
    EXPECT_TRUE(std::filesystem::exists(temporary / "scores" / (stem + ".csv")));
    const std::filesystem::path run_file = temporary / "runs" / (stem + "_2.metrics.json");
    const auto [loaded, label] = load_run(run_file);
    EXPECT_EQ(label, stem);
    EXPECT_EQ(loaded.metric_results.frame("vmaf")->values, (std::vector<float>{96.0F, 80.0F}));
    const std::filesystem::path segments = temporary / "segments" / (stem + ".vmaf.segments.json");
    EXPECT_TRUE(std::filesystem::exists(segments));

    const nlohmann::ordered_json entry = results_document(VideoInfo{}, videos, summarize(videos), false)["videos"][0];
    EXPECT_EQ(entry["run_file"], platform::path_text(run_file));
    EXPECT_EQ(entry["segment_files"][0], platform::path_text(segments));
    EXPECT_EQ(entry["segment_files"][1],
              platform::path_text(temporary / "segments" / (stem + ".vmaf.segments.csv")));
}

// A profile or preset changed on the command line: conditions after it
// replace its condition on their metric (in its place) or are added; one
// left out (its metric not compared) is not when a condition replaces it;
// --segments-any undoes the profile's "every condition"; and one source
// only. --segments-any and --segments-all are one or the other.
TEST(CommandLine, SegmentsChangeAProfileOrPresetFromTheCommandLine) {
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<80", "ssimulacra2<60"};
    settings.weak_segments_all = true;
    const auto taken = [&settings](const std::vector<std::string>& extra, const std::vector<std::string>& asked,
                                   std::vector<std::string>* left_out = nullptr) {
        std::vector<std::string> command{"compare", "ref.mkv", "a.mkv"};
        command.insert(command.end(), extra.begin(), extra.end());
        CompareArguments arguments = parse_arguments(command).compare;
        const std::vector<std::string> left = take_saved_segments(arguments, settings, asked);
        if (left_out != nullptr) {
            *left_out = left;
        }
        return arguments;
    };

    // The preset's VMAF condition replaced; PSNR added.
    CompareArguments changed = taken({"--segments", " preset:grainy-film, vmaf<80 ,psnr<30"}, {"vmaf", "psnr", "ssimulacra2"});
    EXPECT_EQ(changed.segments_preset, "grainy-film");
    EXPECT_EQ(condition_texts(changed.segments), (std::vector<std::string>{"vmaf<80", "ssimulacra2<55", "psnr<30"}));
    EXPECT_EQ(changed.segment_min_frames, 6);  // the preset's options stay
    // Issue #135: two on a metric the preset has, both kept in its place.
    changed = taken({"--segments", "preset:grainy-film,vmaf<85,vmaf>20"}, {"vmaf", "ssimulacra2"});
    EXPECT_EQ(condition_texts(changed.segments), (std::vector<std::string>{"vmaf<85", "vmaf>20", "ssimulacra2<55"}));

    // SSIMULACRA2 not compared: the saved profile's left out, unless
    // replaced (then the replacement is refused later, as given conditions
    // are).
    std::vector<std::string> left_out;
    changed = taken({"--segments", "saved,vmaf<85"}, {"vmaf"}, &left_out);
    EXPECT_EQ(condition_texts(changed.segments), std::vector<std::string>{"vmaf<85"});
    EXPECT_EQ(left_out, std::vector<std::string>{"ssimulacra2<60"});
    changed = taken({"--segments", "saved,ssimulacra2<50"}, {"vmaf"}, &left_out);
    EXPECT_EQ(condition_texts(changed.segments), (std::vector<std::string>{"vmaf<80", "ssimulacra2<50"}));
    EXPECT_TRUE(left_out.empty());
    EXPECT_EQ(segments_refusal(changed.segments, {"vmaf"}),
              "--segments: ssimulacra2 is not one of the metrics compared (-m)");

    // A profile with no conditions, given some: they are the rules.
    settings.weak_segment_profiles_json = R"([{"name": "Empty", "conditions": []}])";
    changed = taken({"--segments", "profile:empty,vmaf<90"}, {"vmaf"});
    EXPECT_EQ(condition_texts(changed.segments), std::vector<std::string>{"vmaf<90"});

    // Every condition (the profile's) or any (--segments-any).
    EXPECT_TRUE(taken({"--segments", "saved"}, {"vmaf", "ssimulacra2"}).segments_all);
    EXPECT_FALSE(taken({"--segments", "saved", "--segments-any"}, {"vmaf", "ssimulacra2"}).segments_all);
    EXPECT_TRUE(taken({"--segments", "preset:cgi", "--segments-all"}, {"vmaf", "ssimulacra2"}).segments_all);

    const auto error_of = [](const std::vector<std::string>& arguments) {
        try {
            parse_arguments(arguments);
        } catch (const UsageError& error) {
            const std::string text = error.what();
            return text.substr(text.find("error: "));
        }
        return std::string("no error");
    };
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--segments", "preset:hdr,profile:Film"}),
              "error: argument --segments: give one of saved, profile:NAME and preset:KEY, not preset:hdr and "
              "profile:Film");
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--segments", "vmaf<90", "--segments-all", "--segments-any"}),
              "error: argument --segments-any: not allowed with argument --segments-all");
    EXPECT_EQ(error_of({"compare", "ref.mkv", "a.mkv", "--segments-any"}),
              "error: argument --segments-any: only with --segments, whose segments it shapes");
    EXPECT_TRUE(error_of({"compare", "ref.mkv", "a.mkv", "--segments", "saved,vmaf=80"})
                    .starts_with("error: argument --segments: invalid condition 'vmaf=80'"));
}

// --gpu names a GPU DirectX lists, by its whole name or a part of one, or
// auto; another name is refused with the GPUs there are, and a part of more
// than one GPU's name with those GPUs. Where the system lists none (a Mac),
// the name is left to Vulkan's.
TEST(CommandLine, GpuNamesAGpuThePcHas) {
    EXPECT_EQ(parse_arguments({"compare", "ref.mkv", "a.mkv", "--gpu", "radeon"}).compare.gpu, "radeon");
    const std::vector<platform::GpuAdapter> adapters{{0, 0x10DE, "NVIDIA GeForce RTX 5090"},
                                                     {1, 0x1002, "AMD Radeon(TM) Graphics"}};
    EXPECT_EQ(gpu_refusal(std::string("radeon"), adapters), std::nullopt);
    EXPECT_EQ(gpu_refusal(std::string("auto"), adapters), std::nullopt);
    EXPECT_EQ(gpu_refusal(std::nullopt, adapters), std::nullopt);
    EXPECT_EQ(gpu_refusal(std::string("titan"), {}), std::nullopt);
    EXPECT_EQ(gpu_refusal(std::string("titan"), adapters),
              "--gpu titan: no GPU here has that name. The GPUs here: NVIDIA GeForce RTX 5090, AMD Radeon(TM) "
              "Graphics; or auto.");
    const std::vector<platform::GpuAdapter> radeons{{0, 0x1002, "AMD Radeon RX 7900 XTX"},
                                                    {1, 0x1002, "AMD Radeon(TM) Graphics"}};
    EXPECT_EQ(gpu_refusal(std::string("radeon"), radeons),
              "--gpu radeon: more than one GPU here has that in its name: AMD Radeon RX 7900 XTX, AMD Radeon(TM) "
              "Graphics. Give more of the name.");
    EXPECT_EQ(gpu_refusal(std::string("radeon(tm)"), radeons), std::nullopt);
    // Chosen by the settings or VML_GPU, one not here is said, not refused.
    EXPECT_EQ(chosen_gpu_problem(std::string("titan"), true, adapters),
              "The GPU the settings choose, titan: no GPU here has that name. The GPUs here: NVIDIA GeForce RTX 5090, "
              "AMD Radeon(TM) Graphics; or auto. No GPU is used for the GPU metrics: choose one with --gpu.");
    EXPECT_NE(chosen_gpu_problem(std::string("radeon"), false, radeons), std::nullopt);
    EXPECT_EQ(chosen_gpu_problem(std::string("AMD Radeon(TM) Graphics"), true, adapters), std::nullopt);
    EXPECT_EQ(chosen_gpu_problem(std::nullopt, true, adapters), std::nullopt);
}

// A run can be held apart from the app's profile: VML_HOME names the folder
// of the app's own files, --settings reads another settings file (one not
// there is an input not read, not the defaults), --no-settings takes the
// defaults, and --cache-dir, --no-save and --no-cache say where the saved
// scores are and whether they are read and saved. The JSON says which.
TEST(CommandLine, SettingsAndSavedScoresCanBeChosen) {
    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path() / ("vml_profile_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(temporary);
    const std::filesystem::path home = app_data_folder();
    platform::set_environment_variable("VML_HOME", platform::path_text(temporary / "home"));
    EXPECT_EQ(Settings::file(), temporary / "home" / "settings.json");
    EXPECT_EQ(default_cache_dir(), temporary / "home" / "results_cache");
    platform::set_environment_variable("VML_HOME", "");
    EXPECT_EQ(app_data_folder(), home);

    Settings mine = Settings::defaults();
    mine.parallel_jobs = 2;
    mine.default_compute_cvvdp = false;
    mine.cache_dir = platform::path_text(temporary / "theirs");
    ASSERT_FALSE(mine.save_to(temporary / "mine.json"));
    const Arguments parsed = parse_arguments(
        {"compare", "r.mkv", "a.mkv", "--settings", platform::path_text(temporary / "mine.json"), "--no-save"});
    const ChosenSettings chosen = choose_settings(parsed.compare.profile);
    EXPECT_EQ(chosen.from, "file");
    EXPECT_EQ(chosen.settings.parallel_jobs, 2);
    EXPECT_FALSE(chosen.settings.default_compute_cvvdp);
    const SavedScoresUse saved = saved_scores_use(parsed.compare.profile, chosen.settings);
    EXPECT_EQ(saved.folder, temporary / "theirs");
    EXPECT_TRUE(saved.read);
    EXPECT_FALSE(saved.save);
    const nlohmann::ordered_json json = profile_json(chosen, saved, options_from_settings(parsed.compare, false));
    EXPECT_EQ(json["settings"], "file");
    EXPECT_EQ(json["settings_file"], platform::path_text(temporary / "mine.json"));
    EXPECT_EQ(json["saved_scores"]["save"], false);
    const auto& taken = json["options_from_settings"];
    EXPECT_NE(std::find(taken.begin(), taken.end(), "--metrics"), taken.end());
    EXPECT_NE(std::find(taken.begin(), taken.end(), "--cache-dir"), taken.end());
    EXPECT_EQ(std::find(taken.begin(), taken.end(), "--gpu"), taken.end());  // not the settings' GPU

    const Arguments defaults = parse_arguments({"compare", "r.mkv", "a.mkv", "--no-settings", "--no-cache"});
    const ChosenSettings none = choose_settings(defaults.compare.profile);
    EXPECT_EQ(none.from, "defaults");
    EXPECT_TRUE(none.settings.default_compute_cvvdp);
    EXPECT_FALSE(none.file);
    const SavedScoresUse unused = saved_scores_use(defaults.compare.profile, none.settings);
    EXPECT_FALSE(unused.read || unused.save);
    EXPECT_TRUE(profile_json(none, unused, {})["saved_scores"]["folder"].is_null());
    EXPECT_THROW(parse_arguments({"compare", "r.mkv", "a.mkv", "--no-cache", "--cache-dir", "x"}), UsageError);
    EXPECT_THROW(parse_arguments({"compare", "r.mkv", "a.mkv", "--settings", "s.json", "--no-settings"}), UsageError);

    ProfileArguments missing;
    missing.settings_file = platform::path_text(temporary / "missing.json");
    EXPECT_THROW(choose_settings(missing), SettingsUnreadable);
    write_bytes_file(temporary / "broken.json", "{\"parallel_jobs\": ");
    missing.settings_file = platform::path_text(temporary / "broken.json");
    EXPECT_THROW(choose_settings(missing), SettingsUnreadable);
    EXPECT_FALSE(std::filesystem::exists(temporary / "broken.json.unreadable"));  // the file named is not touched
    std::filesystem::remove_all(temporary);
}

// A quality gate: --require's bars on a metric's score or another statistic
// (CVVDP's of its seconds), and --fail-on-segments. A bar missed, or a
// metric not scored, fails the gate; one that does not apply to the video
// (HDR VMAF of SDR) is left out of it.
TEST(CommandLine, QualityGatesCheckEachVideo) {
    const Arguments parsed = parse_arguments({"compare", "r.mkv", "a.mkv", "--require", "vmaf>=85, vmaf.min>65",
                                              "--require", "cvvdp.min>=8", "--segments", "vmaf<75",
                                              "--fail-on-segments"});
    const std::vector<GateRule>& rules = parsed.compare.require;
    ASSERT_EQ(rules.size(), 3u);
    EXPECT_EQ(rules[0].text(), "vmaf>=85");
    EXPECT_EQ(rules[1].text(), "vmaf.min>65");
    EXPECT_EQ(rules[2].statistic, "min");
    EXPECT_TRUE(parsed.compare.fail_on_segments);
    for (const char* wrong : {"vmaf=90", "vmaf.mean>=90", "foo>1", "vmaf>=x", ""}) {
        EXPECT_THROW(parse_arguments({"compare", "r.mkv", "a.mkv", "--require", wrong}), UsageError) << wrong;
    }
    EXPECT_THROW(parse_arguments({"compare", "r.mkv", "a.mkv", "--fail-on-segments"}), UsageError);

    std::vector<TestVideo> videos(2);
    TestVideo& scored = videos[0];
    scored.path = "a.mkv";
    scored.metrics = {"vmaf", "cvvdp"};
    scored.result.emplace().compared_frame_count = 3;
    scored.result->metric_results.add(FrameMetricResult{"vmaf", {0, 1, 2}, {0.0, 0.5, 1.0}, {90, 80, 70}, {}});
    scored.result->metric_results.add(SequenceMetricResult{"cvvdp", 8.5, {}, {0, 2}, {0.0, 1.0}, {8.0f, 9.0f}});
    TestVideo& failed = videos[1];
    failed.path = "b.mkv";
    failed.metrics = {"vmaf"};
    failed.failures = {{"vmaf", "Vulkan ran out of memory."}};
    failed.not_applicable = {{"vmaf_hdr", {metric_not_applicable, "HDR VMAF scores PQ (HDR10) video only", ""}}};
    const std::vector<Summaries> summaries = summarize(videos);
    const SegmentRules segments = segment_rules(parsed.compare);

    const std::vector<GateCheck> checks = gate_checks(scored, summaries[0], rules, &segments);
    ASSERT_EQ(checks.size(), 4u);
    EXPECT_EQ(checks[0].value, 80.0);
    EXPECT_EQ(checks[0].passed, false);  // a mean of 80
    EXPECT_EQ(checks[1].passed, true);   // its worst frame, 70
    EXPECT_EQ(checks[2].value, 8.0);     // CVVDP's worst second
    EXPECT_EQ(checks[2].passed, true);
    EXPECT_EQ(checks[3].rule, "no weak segments");
    EXPECT_EQ(checks[3].value, 1.0);
    EXPECT_EQ(checks[3].passed, false);
    EXPECT_FALSE(gate_passed(checks));
    const nlohmann::ordered_json gate = gate_json(checks);
    EXPECT_EQ(gate["passed"], false);
    EXPECT_EQ(gate["checks"][1]["statistic"], "min");
    EXPECT_FALSE(gate["checks"][3].contains("metric"));

    const std::vector<GateCheck> unscored =
        gate_checks(failed, summaries[1], read_gate_rules("vmaf_hdr>=50,vmaf>=85"), nullptr);
    EXPECT_FALSE(unscored[0].passed);  // not applicable: left out of the gate
    EXPECT_EQ(unscored[1].passed, false);
    EXPECT_EQ(unscored[1].why, "Vulkan ran out of memory.");
    EXPECT_FALSE(gate_passed(unscored));
    EXPECT_TRUE(gate_passed({unscored[0]}));
}

// --timeout and --video-timeout, on a clock of the test's own: the run is due
// once, from the command's start; each video from its own start, until it
// ends; and each video runs on a cancel token of its own, so stopping one
// leaves the others running (a regression: one token copied into each place
// stopped every video).
TEST(CommandLine, TimeoutsStopTheRunOrOneVideo) {
    const CompareArguments parsed = parse_arguments({"compare", "r.mkv", "a.mkv", "--timeout", "1:00",
                                                     "--video-timeout", "10", "--progress", "jsonl",
                                                     "--cancel-on-stdin-eof"})
                                        .compare;
    EXPECT_EQ(parsed.timeout, 60);
    EXPECT_EQ(parsed.video_timeout, 10);
    EXPECT_EQ(parsed.progress, "jsonl");
    EXPECT_TRUE(parsed.cancel_on_stdin_eof);
    EXPECT_THROW(parse_arguments({"compare", "r.mkv", "a.mkv", "--timeout", "0"}), UsageError);
    EXPECT_THROW(parse_arguments({"compare", "r.mkv", "a.mkv", "--progress", "json"}), UsageError);

    Deadlines deadlines(100.0, 60.0, 10.0);
    deadlines.started(0, 105.0);
    deadlines.started(1, 110.0);
    EXPECT_TRUE(deadlines.due(114.9).jobs.empty());
    EXPECT_EQ(deadlines.due(115.0).jobs, std::vector<std::size_t>{0});
    EXPECT_TRUE(deadlines.due(116.0).jobs.empty());  // said once
    deadlines.ended(1);
    EXPECT_TRUE(deadlines.due(130.0).jobs.empty());  // ended before its time
    EXPECT_FALSE(deadlines.due(159.9).run);
    EXPECT_TRUE(deadlines.due(160.0).run);
    EXPECT_FALSE(deadlines.due(161.0).run);
    Deadlines none(0.0, std::nullopt, std::nullopt);
    none.started(0, 0.0);
    const Deadlines::Due never = none.due(1e9);
    EXPECT_FALSE(never.run);
    EXPECT_TRUE(never.jobs.empty());

    std::vector<CancelToken> tokens = separate_tokens(3);
    tokens[1].cancel();
    EXPECT_FALSE(tokens[0].cancelled());
    EXPECT_TRUE(tokens[1].cancelled());
    EXPECT_FALSE(tokens[2].cancelled());
}

// --timeout ends the command at once outside the run, cancels the run, and
// does nothing once the results are given (which say it did not time out)
// or once Ctrl+C has cancelled the run (whose 130 it does not change).
TEST(CommandLine, TimeoutActsByTheRunsStage) {
    EXPECT_EQ(on_run_timeout(RunStage::outside, false), TimeoutAction::end_command);
    EXPECT_EQ(on_run_timeout(RunStage::running, false), TimeoutAction::cancel_run);
    EXPECT_EQ(on_run_timeout(RunStage::running, true), TimeoutAction::nothing);
    EXPECT_EQ(on_run_timeout(RunStage::results_given, false), TimeoutAction::nothing);
}

// The automation contract: every error, failure, note and why in the JSON has
// a code beside it, a run that compared nothing still writes a document, and
// an input not read is exit code 4, not a wrong command's 2.
TEST(CommandLine, ErrorsNotesAndWhysComeWithCodes) {
    std::vector<TestVideo> videos(2);
    videos[0].path = "Y:/v/missing.mkv";
    videos[0].error = "it could not be read: no such file";
    videos[0].error_code = "unreadable";
    TestVideo& scored = videos[1];
    scored.path = "Y:/v/a.mkv";
    scored.metrics = {"psnr"};
    EXPECT_TRUE(scored.note("colours_guessed", "the reference has no colour tags"));
    EXPECT_FALSE(scored.note("colours_guessed", "the reference has no colour tags"));  // said once
    EXPECT_TRUE(scored.note("gpu_fallback", "VMAF fell back to the CPU"));
    scored.failures = {{"cvvdp", "CVVDP needs every frame"}, {"vmaf", "Vulkan ran out of memory."}};
    scored.failure_codes["cvvdp"] = "needs_every_frame";
    RunFacts facts;
    facts.metrics = {{"psnr", {metric_calculated, ""}},
                     {"vmaf", {metric_failed, "Vulkan ran out of memory."}},
                     {"vmaf_hdr", {metric_not_applicable, "HDR VMAF scores PQ (HDR10) video only", "hdr_vmaf_colours"}},
                     {"cvvdp", {metric_not_run, "no task was planned to calculate it"}}};
    scored.facts = facts;

    const nlohmann::ordered_json document = results_document(VideoInfo{}, videos, summarize(videos), false);
    EXPECT_EQ(document["videos"][0]["error_code"], "unreadable");
    const nlohmann::ordered_json& entry = document["videos"][1];
    EXPECT_EQ(entry["notes"].size(), 2u);
    EXPECT_EQ(entry["note_codes"], (nlohmann::ordered_json{"colours_guessed", "gpu_fallback"}));
    EXPECT_EQ(entry["failed_metric_codes"]["cvvdp"], "needs_every_frame");
    EXPECT_EQ(entry["failed_metric_codes"]["vmaf"], "calculation_failed");  // the run's own failure
    const nlohmann::ordered_json& status = entry["metric_status"];
    EXPECT_FALSE(status["psnr"].contains("code"));
    EXPECT_EQ(status["vmaf"]["code"], "calculation_failed");
    EXPECT_EQ(status["vmaf_hdr"]["status"], "not applicable");
    EXPECT_EQ(status["vmaf_hdr"]["code"], "hdr_vmaf_colours");
    EXPECT_EQ(status["cvvdp"]["code"], "not_planned");

    const nlohmann::ordered_json unread =
        error_document(std::string("Y:/v/reference.mkv"), "The reference could not be read", "reference_unreadable");
    EXPECT_EQ(unread["format"], document["format"]);
    EXPECT_EQ(unread["reference"]["path"], "Y:/v/reference.mkv");
    EXPECT_EQ(unread["error_code"], "reference_unreadable");
    EXPECT_TRUE(unread["videos"].empty());
    EXPECT_TRUE(error_document(std::nullopt, "--batch: no such file", "batch_unreadable")["batch"].empty());

    const std::filesystem::path temporary =
        std::filesystem::temp_directory_path() / ("vml_codes_test_" + std::to_string(std::random_device{}()));
    BdrateArguments rates;
    rates.anchor = platform::path_text(temporary / "missing.csv");
    rates.test = rates.anchor;
    rates.json_path = platform::path_text(temporary / "bdrate.json");
    rates.quiet = true;
    EXPECT_EQ(bdrate(rates), exit_input);
    const Json written = Json::parse(read_whole_file(temporary / "bdrate.json"));
    EXPECT_EQ(written["format"], "bdrate-1");
    EXPECT_EQ(written["error_code"], "rates_unreadable");
    std::filesystem::remove_all(temporary);
}

}  // namespace
}  // namespace vml::cli
