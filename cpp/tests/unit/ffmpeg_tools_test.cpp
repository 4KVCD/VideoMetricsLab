#include "vml/ffmpeg_tools.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/platform/system.h"

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
    ToolsStatus status;
    status.ffmpeg.name = "ffmpeg";
    status.ffmpeg.runnable = true;
    status.ffmpeg.version = std::vector<int>{8, 1};
    status.ffprobe.name = "ffprobe";
    status.ffprobe.runnable = true;
    ASSERT_EQ(status.problems().size(), 1u);
    EXPECT_EQ(status.problems().front(), "ffmpeg 8.1 is too old -- version 9 or newer is required.");
    status.ffmpeg.version = std::vector<int>{9, 0, 1};
    EXPECT_TRUE(status.ok());
    status.ffmpeg.version.reset();
    EXPECT_TRUE(status.ok());
}

// FFmpeg's zscale filter, which HDR video converted to SDR needs: found in a
// -filters listing by its name, not in another filter's description; its
// absence named instead of FFmpeg's filter error.
TEST(FfmpegTools, AnFfmpegWithoutZscaleIsNamedForWhatItCannotDo) {
    const std::string listing =
        "Filters:\n"
        "  T.. = Timeline support\n"
        "  .S. = Slice threading\n"
        "  A = Audio input/output\n"
        "  | = Source or sink filter\n"
        "  ------\n"
        " .S tonemap           V->V       Conversion to/from different dynamic ranges.\n"
        " .S zscale            V->V       Apply resizing, colorspace and bit depth conversion.\n";
    EXPECT_TRUE(lists_filter(listing, "zscale"));
    EXPECT_TRUE(lists_filter(listing, "tonemap"));
    EXPECT_FALSE(lists_filter(listing, "libplacebo"));
    EXPECT_FALSE(lists_filter(" .. scale  V->V  Scale like zscale does.\n", "zscale"));
    EXPECT_FALSE(lists_filter("", "zscale"));
    // Windows' line ends, and blank lines at the end.
    EXPECT_TRUE(lists_filter(" .S zscale  V->V  Apply.\r\n\r\n\n", "zscale"));

    const std::string refused = "[AVFilterGraph @ 0x7909408000] No such filter: 'zscale'\nError : Filter not found";
    EXPECT_EQ(explain_ffmpeg_error(refused), missing_zscale_text());
    EXPECT_EQ(explain_ffmpeg_error("Invalid data found when processing input"),
              "Invalid data found when processing input");
#ifdef __APPLE__
    EXPECT_NE(missing_zscale_text().find("brew install ffmpeg-full"), std::string::npos);
#else
    EXPECT_EQ(missing_zscale_text(), "FFmpeg has no zscale filter: HDR video cannot be converted to SDR");
#endif
}

#ifdef __APPLE__
// Homebrew's ffmpeg-full (with zscale, not linked into its bin folder) comes
// before the PATH on a Mac, this system's Homebrew's only (an Intel one's
// left by a migration never comes before an Apple silicon PATH's), unless
// the folders are named for a test.
TEST(FfmpegTools, AMacLooksForHomebrewsFfmpegFullFirst) {
    const std::vector<std::filesystem::path> preferred = mac_preferred_ffmpeg_folders();
    if (platform::environment_variable(ffmpeg_folders_variable)) {
        GTEST_SKIP() << "VML_FFMPEG_FOLDERS is set where the test runs";
    }
#if defined(__aarch64__) || defined(__arm64__)
    EXPECT_EQ(preferred, (std::vector<std::filesystem::path>{"/opt/homebrew/opt/ffmpeg-full/bin"}));
#else
    EXPECT_EQ(preferred, (std::vector<std::filesystem::path>{"/usr/local/opt/ffmpeg-full/bin"}));
#endif
    const test::ScopedVariable named(std::string(ffmpeg_folders_variable), "/nowhere");
    EXPECT_TRUE(mac_preferred_ffmpeg_folders().empty());
}
#endif

}  // namespace
}  // namespace vml
