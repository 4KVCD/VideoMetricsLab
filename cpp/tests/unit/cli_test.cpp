// The command line reads its arguments and finds its test videos as the
// Python app's does (reference/cli.json, made by its argparse parser and its
// expand_tests).
#include "arguments.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "compare.h"
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
        {"cvvdp_displays", a.cvvdp_displays.empty() ? Json(nullptr) : Json(a.cvvdp_displays)},
        {"subsample", a.subsample},
        {"threads", either(a.threads)},
        {"vmaf_on", either(a.vmaf_on)},
        {"ssimulacra2_on", either(a.ssimulacra2_on)},
        {"butteraugli_on", either(a.butteraugli_on)},
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
                if (values.is_object() && values.contains("duration")) {
                    const std::optional<double> duration = test::reference_number(values["duration"]);
                    values["duration"] = duration ? Json(*duration) : Json(nullptr);
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

}  // namespace
}  // namespace vml::cli
