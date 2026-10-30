// The command line reads its arguments, finds its test videos and says where
// each metric is calculated as the Python app's does (reference/cli.json,
// made by its argparse parser and its expand_tests, and devices.json, by its
// devices command).
#include "arguments.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "compare.h"
#include "devices.h"
#include "reference.h"
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
        {"black_bars", a.black_bars},
        {"scale_to", a.scale_to},
        {"scaler", a.scaler},
        {"duration", either(a.duration)},
        {"frames", either(a.frames)},
        {"colour", either(a.colour)},
        {"film_grain", either(a.film_grain)},
        {"fps", either(a.fps)},
        {"size", either(a.size)},
        {"pix_fmt", either(a.pix_fmt)},
        {"cvvdp_displays", a.cvvdp_displays.empty() ? Json(nullptr) : Json(a.cvvdp_displays)},
        {"subsample", a.subsample},
        {"threads", either(a.threads)},
        {"vmaf_on", either(a.vmaf_on)},
        {"vship", either(a.vship)},
        {"cpu", a.cpu},
        {"no_gpu_decode", a.no_gpu_decode},
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
                for (const char* number : {"duration", "fps"}) {
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

}  // namespace
}  // namespace vml::cli
