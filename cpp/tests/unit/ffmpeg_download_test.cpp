#include "vml/ffmpeg_download.h"

#include <miniz.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "test_videos.h"

namespace vml {
namespace {

// A release's files as GitHub lists them (trimmed from FFmpeg-Builds'
// latest release of 2026-11-16).
std::string release(const std::string& digest_of_9) {
    const auto asset = [](const std::string& name, const std::string& digest) {
        return R"({"name": ")" + name + R"(", "size": 194908247, "digest": )" +
               (digest.empty() ? std::string("null") : "\"" + digest + "\"") +
               R"(, "browser_download_url": "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/latest/)" +
               name + "\"}";
    };
    const std::string sha = "sha256:53328634D32826537ADB68BB2021E81FD7BB113CB026545426F263FA06681909";
    return R"({"tag_name": "latest", "assets": [)" + asset("checksums.sha256", "") + "," +
           asset("ffmpeg-master-latest-win64-gpl.zip", sha) + "," + asset("ffmpeg-n8.1-latest-win64-gpl-8.1.zip", sha) +
           "," + asset("ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip", sha) + "," +
           asset("ffmpeg-n9.0-latest-win64-lgpl-9.0.zip", sha) + "," +
           asset("ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz", sha) + "," +
           asset("ffmpeg-n9.0-latest-win64-gpl-9.0.zip", digest_of_9) + "]}";
}

// The stable Windows x64 GPL build of version 9 or newer, with its SHA-256
// from the release, else the release's checksums file to get it from.
TEST(FfmpegDownload, ChoosesTheNewestStableWindowsBuild) {
    const std::optional<FfmpegBuild> build =
        choose_ffmpeg_build(release("sha256:53328634D32826537ADB68BB2021E81FD7BB113CB026545426F263FA06681909"));
    ASSERT_TRUE(build);
    EXPECT_EQ(build->name, "ffmpeg-n9.0-latest-win64-gpl-9.0.zip");
    EXPECT_EQ(build->version, "9.0");
    EXPECT_EQ(build->size, 194908247);
    EXPECT_EQ(build->url,
              "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/latest/"
              "ffmpeg-n9.0-latest-win64-gpl-9.0.zip");
    EXPECT_EQ(build->sha256, "53328634d32826537adb68bb2021e81fd7bb113cb026545426f263fa06681909");
    EXPECT_EQ(build->checksums_url,
              "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/latest/checksums.sha256");

    EXPECT_EQ(choose_ffmpeg_build(release(""))->sha256, "");  // no digest: the checksums file then
    EXPECT_EQ(choose_ffmpeg_build(release("md5:0123"))->sha256, "");

    std::string newer = release("");
    newer.insert(newer.rfind(']'), R"(, {"name": "ffmpeg-n10.1-latest-win64-gpl-10.1.zip", "size": 5,
        "browser_download_url": "https://example.invalid/10.1.zip"})");
    EXPECT_EQ(choose_ffmpeg_build(newer)->version, "10.1");
    std::string too_long = release("");  // a number no int holds: left out, not thrown
    too_long.insert(too_long.rfind(']'), R"(, {"name": "ffmpeg-n99999999999.0-latest-win64-gpl-1.zip", "size": 5,
        "browser_download_url": "https://example.invalid/1.zip"})");
    EXPECT_EQ(choose_ffmpeg_build(too_long)->version, "9.0");

    std::string only_8 = release("");
    only_8.replace(only_8.find("ffmpeg-n9.0-latest-win64-gpl-9.0"), 32, "ffmpeg-n8.0-latest-win64-gpl-8.0");
    EXPECT_FALSE(choose_ffmpeg_build(only_8));  // too old for the app
    for (const char* answer : {"", "not json", "{}", R"({"assets": 3})", R"({"message": "API rate limit exceeded"})"}) {
        EXPECT_FALSE(choose_ffmpeg_build(answer)) << answer;
    }
}

TEST(FfmpegDownload, ReadsAChecksumsFile) {
    const std::string sha = "53328634d32826537adb68bb2021e81fd7bb113cb026545426f263fa06681909";
    const std::string list = std::string(64, 'a') + "  ffmpeg-n8.1-latest-win64-gpl-8.1.zip\r\n" + sha +
                             " *ffmpeg-n9.0-latest-win64-gpl-9.0.zip\r\nnot a line\n";
    EXPECT_EQ(checksum_from_list(list, "ffmpeg-n9.0-latest-win64-gpl-9.0.zip"), sha);
    EXPECT_EQ(checksum_from_list(list, "ffmpeg-n9.0-latest-win64-gpl-9.0"), std::nullopt);
    EXPECT_EQ(checksum_from_list("xyz  a.zip\n", "a.zip"), std::nullopt);  // not a SHA-256
}

// A build's zip with these entries, written to `file`.
void write_zip(const std::filesystem::path& file, const std::vector<std::pair<std::string, std::string>>& entries) {
    mz_zip_archive zip{};
    ASSERT_TRUE(mz_zip_writer_init_heap(&zip, 0, 0));
    for (const auto& [name, data] : entries) {
        ASSERT_TRUE(mz_zip_writer_add_mem(&zip, name.c_str(), data.data(), data.size(), MZ_DEFAULT_LEVEL));
    }
    void* archive = nullptr;
    size_t size = 0;
    ASSERT_TRUE(mz_zip_writer_finalize_heap_archive(&zip, &archive, &size));
    std::ofstream(file, std::ios::binary).write(static_cast<const char*>(archive), static_cast<std::streamsize>(size));
    mz_free(archive);
    mz_zip_writer_end(&zip);
}

std::string contents(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

// Only ffmpeg.exe and ffprobe.exe come out of the zip's bin folder,
// replacing older ones; a zip without one, or a Cancel, leaves the folder
// as it was.
TEST(FfmpegDownload, UnpacksOnlyTheTwoPrograms) {
    const test::Folder folder;
    const std::filesystem::path out = folder / "ffmpeg folder";
    std::filesystem::create_directories(out);
    std::ofstream(out / "ffmpeg.exe") << "old";
    write_zip(folder / "build.zip", {{"ffmpeg-n9.0-latest-win64-gpl-9.0/bin/ffmpeg.exe", std::string(70'000, 'f')},
                                     {"ffmpeg-n9.0-latest-win64-gpl-9.0/bin/ffprobe.exe", "probe"},
                                     {"ffmpeg-n9.0-latest-win64-gpl-9.0/bin/ffplay.exe", "play"},
                                     {"ffmpeg-n9.0-latest-win64-gpl-9.0/doc/ffmpeg.html", "doc"}});
    unpack_ffmpeg(folder / "build.zip", out, CancelToken());
    EXPECT_EQ(contents(out / "ffmpeg.exe"), std::string(70'000, 'f'));
    EXPECT_EQ(contents(out / "ffprobe.exe"), "probe");
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(out)) {
        names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    EXPECT_EQ(names, (std::vector<std::string>{"ffmpeg.exe", "ffprobe.exe"}));

    // Not "<top>/bin/": not taken, so the zip lacks ffprobe.exe.
    std::ofstream(out / "ffmpeg.exe", std::ios::trunc) << "old";
    write_zip(folder / "bad.zip", {{"top/bin/ffmpeg.exe", "new"}, {"../bin/ffprobe.exe", "x"}, {"a/b/bin/ffprobe.exe", "x"}});
    EXPECT_THROW(unpack_ffmpeg(folder / "bad.zip", out, CancelToken()), FfmpegUnpackError);
    EXPECT_EQ(contents(out / "ffmpeg.exe"), "old");
    EXPECT_FALSE(std::filesystem::exists(out / "ffmpeg.exe.part"));

    CancelToken cancelled;
    cancelled.cancel();
    EXPECT_THROW(unpack_ffmpeg(folder / "build.zip", out, cancelled), Cancelled);
    EXPECT_EQ(contents(out / "ffmpeg.exe"), "old");
    EXPECT_FALSE(std::filesystem::exists(out / "ffmpeg.exe.part"));

    std::ofstream(folder / "not.zip") << "not a zip";
    EXPECT_THROW(unpack_ffmpeg(folder / "not.zip", out, CancelToken()), FfmpegUnpackError);
}

}  // namespace
}  // namespace vml
