// The update check held to the Python app's (reference/update_check.json,
// from its update_check.py): versions compared as numbers, and GitHub's
// answers read into the same release, or refused where it refuses them.
#include "vml/update_check.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>

#include "reference.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

TEST(UpdateCheck, ReadsGitHubsAnswerAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("update_check.json");
    EXPECT_EQ(releases_url, reference.at("url").get<std::string>());
    EXPECT_EQ(update_check_timeout_seconds, reference.at("timeout").get<double>());

    const nlohmann::json& versions = reference.at("versions");
    ASSERT_FALSE(versions.empty());
    for (const nlohmann::json& item : versions) {
        const std::string candidate = item.at("candidate");
        const std::string current = item.at("current");
        SCOPED_TRACE("'" + candidate + "' against '" + current + "'");
        EXPECT_EQ(is_newer(candidate, current), item.at("newer").get<bool>());
        const auto tuple = version_tuple(candidate);
        EXPECT_EQ(tuple.has_value(), !item.at("tuple").is_null());
        if (tuple && !item.at("tuple").is_null()) {
            EXPECT_EQ(*tuple, item.at("tuple").get<std::vector<long long>>());
        }
    }

    const nlohmann::json& answers = reference.at("answers");
    ASSERT_FALSE(answers.empty());
    for (const nlohmann::json& item : answers) {
        const std::string body = item.at("body");
        SCOPED_TRACE(body);
        if (item.at("release").is_null()) {
            EXPECT_THROW(release_from_answer(body), UpdateCheckError);
            continue;
        }
        const Release release = release_from_answer(body);
        EXPECT_EQ(release.version, item.at("release").at("version").get<std::string>());
        EXPECT_EQ(release.page_url, item.at("release").at("page_url").get<std::string>());
        EXPECT_EQ(release.notes, item.at("release").at("notes").get<std::string>());
    }
}

// A release's package for this system is the file the release scripts name
// for it, uploaded whole; a release without it is not offered here (where a
// package is published at all).
TEST(UpdateCheck, OffersThisSystemsPackage) {
    const auto answer = [](const nlohmann::json& assets) {
        return release_from_answer(nlohmann::json{{"tag_name", "v9.1"}, {"assets", assets}}.dump());
    };
    const std::string base = "https://github.com/4KVCD/VideoMetricsLab/releases/download/v9.1/";
    const nlohmann::json windows{{"name", "VideoMetricsLab-windows.zip"},
                                 {"browser_download_url", base + "VideoMetricsLab-windows.zip"},
                                 {"state", "uploaded"}};
    const nlohmann::json mac{{"name", "VideoMetricsLab-macos-arm64.zip"},
                             {"browser_download_url", base + "VideoMetricsLab-macos-arm64.zip"},
                             {"state", "uploaded"}};
    const nlohmann::json mac_checksum{{"name", "VideoMetricsLab-macos-arm64.zip.sha256"},
                                      {"browser_download_url", base + "VideoMetricsLab-macos-arm64.zip.sha256"}};
    nlohmann::json mac_unfinished = mac;
    mac_unfinished["state"] = "open";
    nlohmann::json windows_unfinished = windows;
    windows_unfinished["state"] = "open";

    const Release both = answer({mac_checksum, windows, mac});
    const Release windows_only = answer({windows, mac_unfinished});
    const Release mac_only = answer({mac, windows_unfinished});
    const Release none = release_from_answer(R"({"tag_name": "v9.1"})");
    const Release odd = answer({"text", {{"name", 3}}, {{"name", "VideoMetricsLab-windows.zip"}}});
    EXPECT_FALSE(none.download_url);
    EXPECT_FALSE(odd.download_url);
    const std::optional<std::string_view> here = package_name_here();
    if (!here) {  // no package published here: every release, as before
        for (const Release* release : {&both, &windows_only, &mac_only, &none}) {
            EXPECT_FALSE(release->download_url);
            EXPECT_TRUE(offers_package_here(*release));
        }
        return;
    }
    EXPECT_EQ(both.download_url, base + std::string(*here));
    EXPECT_FALSE(offers_package_here(none));
#ifdef _WIN32
    EXPECT_EQ(*here, "VideoMetricsLab-windows.zip");
    EXPECT_TRUE(offers_package_here(windows_only));
    EXPECT_FALSE(offers_package_here(mac_only));
#else
    EXPECT_EQ(*here, "VideoMetricsLab-macos-arm64.zip");
    EXPECT_FALSE(offers_package_here(windows_only));
    EXPECT_TRUE(offers_package_here(mac_only));
#endif
}

// The packages the release scripts make are the files the update check looks
// for: renamed, no user of that system would be told of a release.
TEST(UpdateCheck, TheReleaseScriptsMakeThePackagesItLooksFor) {
    const std::filesystem::path scripts = std::filesystem::path(VML_REFERENCE_DIR) / ".." / ".." / ".." / "scripts";
    const std::optional<std::string> windows = platform::read_file(scripts / "build_release.ps1");
    const std::optional<std::string> mac = platform::read_file(scripts / "build_release_macos.sh");
    ASSERT_TRUE(windows && mac);
    EXPECT_NE(windows->find("'VideoMetricsLab-windows.zip'"), std::string::npos);
    EXPECT_NE(mac->find("/VideoMetricsLab-macos-arm64.zip\""), std::string::npos);
}

}  // namespace
}  // namespace vml
