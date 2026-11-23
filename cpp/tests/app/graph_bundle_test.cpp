// The names a graph export gives its videos' files (graph_bundle.h): each
// video's own, made unique by its folder, then by a number, and fit for a
// file name; and the files whose bitrate it reads.
#include "graph_bundle.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace vml::app {
namespace {

BundleVideo video(const char* path, const char* label = "") {
    return BundleVideo{nullptr, QString::fromUtf8(label), std::filesystem::path(path)};
}

TEST(GraphBundle, EachVideosFilesAreNamedAfterItAndNamesAreNotShared) {
    // Different names: each its label, or its file's name without one.
    EXPECT_EQ(bundle_names({video("a/Film CRF18.mkv", "Film CRF18"), video("a/Film CRF20.mkv")}),
              (std::vector<QString>{"Film CRF18", "Film CRF20"}));
    // The same name in two folders: each with its folder's.
    EXPECT_EQ(bundle_names({video("x264/Film.mkv"), video("x265/Film.mkv"), video("x265/Other.mkv")}),
              (std::vector<QString>{"Film (x264)", "Film (x265)", "Other"}));
    // The same name in the same folder (two rows of one file): numbered.
    EXPECT_EQ(bundle_names({video("enc/Film.mkv"), video("enc/Film.mkv")}),
              (std::vector<QString>{"Film (enc)", "Film (enc) 2"}));
    EXPECT_EQ(file_safe("VMAF: a/b <c>?"), "VMAF_ a_b _c__");
}

TEST(GraphBundle, BitrateIsReadFromTheFilesWhereTheyAreNow) {
    // Saved scores found for files moved since: their result names the old
    // paths. The bitrate is read where the export looks it up, the files'
    // paths now, with the rest of what was probed kept; a resolution test,
    // which has no file of its own, adds none.
    const auto saved = [](const char* source, const char* distorted, bool resample) {
        auto result = std::make_shared<ComparisonResult>();
        result->source_info.path = source;
        result->source_info.fps = 24.0;
        result->distorted_info.path = distorted;
        result->distorted_info.fps = 25.0;
        if (resample) {
            result->resample_target = ResampleTarget{};
        }
        return std::shared_ptr<const ComparisonResult>(std::move(result));
    };
    const std::vector<BundleVideo> videos{
        {saved("old/ref.mkv", "old/crf20.mkv", false), "crf20", "new/crf20.mkv"},
        {saved("old/ref.mkv", "old/ref.mkv", true), "720p", "new/ref.mkv"},
        {saved("old/ref.mkv", "old/crf24.mkv", false), "crf24", "new/crf24.mkv"},
    };
    const std::vector<VideoInfo> files = bitrate_files("new/ref.mkv", videos);
    ASSERT_EQ(files.size(), 3u);
    EXPECT_EQ(files[0].path, std::filesystem::path("new/ref.mkv"));
    EXPECT_EQ(files[0].fps, 24.0);
    EXPECT_EQ(files[1].path, std::filesystem::path("new/crf20.mkv"));
    EXPECT_EQ(files[1].fps, 25.0);
    EXPECT_EQ(files[2].path, std::filesystem::path("new/crf24.mkv"));
    EXPECT_TRUE(bitrate_files("new/ref.mkv", {}).empty());
}

// Issue #134: two saved comparisons of one file name, an older encode
// replaced since by the newer, exported together: each draws its own
// bitrate, whichever comes first. The newer's is read from the file; the
// older's, as saved, under a key of its own, with which no file is read.
// (Keyed by path, the older's saved bitrate was drawn for both.)
TEST(GraphBundle, ComparisonsOfOneFileNameEachDrawTheirOwnBitrate) {
    auto result = std::make_shared<ComparisonResult>();
    result->source_info.path = "ref.mkv";
    result->distorted_info.path = "enc.mkv";
    const std::shared_ptr<const ComparisonResult> scored = result;
    const auto own = [](const std::filesystem::path& folder, bool, const std::filesystem::path&) {
        return folder == "cache/newer";
    };
    const auto load = [](const std::filesystem::path& folder, bool reference, const std::filesystem::path& file) {
        SavedBitrate saved;
        saved.info.path = file;
        saved.info.codec_name = folder.filename().string() + (reference ? " reference" : "");
        return std::optional(saved);
    };
    for (const bool newer_first : {true, false}) {
        std::vector<BundleVideo> videos{{scored, "enc", "enc.mkv"}, {scored, "enc", "enc.mkv"}};
        std::vector<std::filesystem::path> folders{"cache/newer", "cache/older"};
        if (!newer_first) {
            std::swap(folders[0], folders[1]);
        }
        const SavedBitrates saved = saved_bitrates(videos, folders, "ref.mkv", own, load);
        const BundleVideo& newer = videos[newer_first ? 0 : 1];
        const BundleVideo& older = videos[newer_first ? 1 : 0];
        EXPECT_EQ(newer.bitrate_path(), std::filesystem::path("enc.mkv"));
        EXPECT_NE(older.bitrate_path(), newer.bitrate_path());
        EXPECT_EQ(older.path, std::filesystem::path("enc.mkv"));  // its name and folder in the export
        ASSERT_TRUE(saved.contains(older.bitrate_path()));
        EXPECT_EQ(saved.at(older.bitrate_path())->info.codec_name, "older");
        EXPECT_FALSE(saved.contains(newer.bitrate_path()));
        // The reference, the first's: read from the file where it is its own.
        EXPECT_EQ(saved.contains("ref.mkv"), !newer_first);
        const std::vector<VideoInfo> files = bitrate_files("ref.mkv", videos, saved);
        ASSERT_EQ(files.size(), newer_first ? 2u : 1u);
        EXPECT_EQ(files.front().path, std::filesystem::path(newer_first ? "ref.mkv" : "enc.mkv"));
        EXPECT_EQ(files.back().path, std::filesystem::path("enc.mkv"));  // the newer's: read from the file
    }
}

}  // namespace
}  // namespace vml::app
