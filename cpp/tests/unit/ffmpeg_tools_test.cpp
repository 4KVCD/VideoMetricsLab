#include "vml/ffmpeg_tools.h"

#include <gtest/gtest.h>

#include <string>

#include "reference.h"

namespace vml {
namespace {

// FFmpeg's version is read from what its builds print as the Python app
// reads it, development builds giving none.
TEST(FfmpegTools, ReadsTheVersionAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("versions.json");
    ASSERT_FALSE(reference["cases"].empty());
    for (const nlohmann::json& item : reference["cases"]) {
        const std::string output = item["output"].get<std::string>();
        const std::optional<std::vector<int>> version = parse_version(output);
        EXPECT_EQ(version.has_value(), !item["version"].is_null()) << output;
        if (version) {
            EXPECT_EQ(*version, item["version"].get<std::vector<int>>()) << output;
        }
        EXPECT_EQ(format_version(version), item["text"].get<std::string>()) << output;
    }
}

// An FFmpeg older than 9 cannot be used; one with no version number can.
TEST(FfmpegTools, AnFfmpegOlderThan9IsAProblem) {
    ToolsStatus status{{"ffmpeg", "ffmpeg", true, std::vector<int>{8, 1}}, {"ffprobe", "ffprobe", true}};
    ASSERT_EQ(status.problems().size(), 1u);
    EXPECT_EQ(status.problems().front(), "ffmpeg 8.1 is too old -- version 9 or newer is required.");
    status.ffmpeg.version = std::vector<int>{9, 0, 1};
    EXPECT_TRUE(status.ok());
    status.ffmpeg.version.reset();
    EXPECT_TRUE(status.ok());
}

}  // namespace
}  // namespace vml
