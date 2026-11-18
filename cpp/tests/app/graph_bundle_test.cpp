// The names a graph export gives its videos' files (graph_bundle.h): each
// video's own, made unique by its folder, then by a number, and fit for a
// file name.
#include "graph_bundle.h"

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace vml::app
