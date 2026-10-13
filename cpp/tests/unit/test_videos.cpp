#include "test_videos.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <random>

#include "vml/ffmpeg_tools.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/settings.h"

namespace vml::test {

namespace {

void set_variable(const std::string& name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());  // an empty value removes it
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

}  // namespace

std::optional<Tools> tools() {
    const ToolsStatus status = check_tools(Settings{});
    if (!status.ok()) {
        return std::nullopt;
    }
    return Tools{status.ffmpeg.path, status.ffprobe.path, format_version(status.ffmpeg.version)};
}

Folder::Folder()
    : path_(std::filesystem::temp_directory_path() / ("vml_test_" + std::to_string(std::random_device()()))) {
    std::filesystem::create_directories(path_);
}

Folder::~Folder() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
}

void run_ffmpeg(const Tools& found, std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), {found.ffmpeg, "-nostdin", "-v", "error", "-y"});
    const platform::ProcessOutput output = platform::run_program(arguments, std::chrono::minutes(2));
    EXPECT_EQ(output.exit_code, 0) << output.standard_error;
}

std::filesystem::path make_clip(const Tools& found, const std::filesystem::path& path, const std::string& encoder,
                                const std::string& pix_fmt, double seconds, std::vector<std::string> extra,
                                const std::string& size) {
    std::vector<std::string> arguments{"-f", "lavfi", "-i",
                                       "testsrc2=s=" + size + ":r=24000/1001:d=" + std::to_string(seconds),
                                       "-pix_fmt", pix_fmt, "-c:v", encoder};
    if (encoder == "libx264") {
        arguments.insert(arguments.end(), {"-preset", "veryfast", "-bf", "3"});
    } else {
        arguments.insert(arguments.end(), {"-preset", "ultrafast", "-x265-params", "bframes=4:log-level=error"});
    }
    arguments.insert(arguments.end(), extra.begin(), extra.end());
    arguments.push_back(platform::path_text(path));
    run_ffmpeg(found, arguments);
    return path;
}

ScopedVariable::ScopedVariable(std::string name, const std::string& value)
    : name_(std::move(name)), previous_(platform::environment_variable(name_).value_or("")) {
    set_variable(name_, value);
}

ScopedVariable::~ScopedVariable() { set_variable(name_, previous_); }

}  // namespace vml::test
