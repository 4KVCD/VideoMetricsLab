// A card's save icon's one-click save (quick_save.h, issue #224): the formats
// in Settings' order, and a save that stops at the first file it cannot
// write, saying why.
#include "quick_save.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "test_videos.h"
#include "vml/platform/system.h"
#include "vml/settings.h"

namespace vml::app {
namespace {

TEST(QuickSave, WritesInSettingsOrderAndStopsAtAFailure) {
    Settings settings;
    settings.quick_save_formats = {"csv", "json"};
    EXPECT_EQ(quick_formats(settings), (std::vector<std::string>{"json", "csv"}));
    EXPECT_EQ(formats_text(quick_formats(settings)).toStdString(), "JSON and CSV");
    settings.quick_save_formats = {"csv"};
    EXPECT_EQ(formats_text(quick_formats(settings)).toStdString(), "CSV");

    std::vector<QString> asked;
    const auto file = [&asked](const QString& name, const QString& error) {
        return QuickFile{name, [&asked, error](const QString& path) {
                             asked.push_back(path);
                             return error;
                         }};
    };
    const test::Folder folder;
    const ExportOutcome outcome =
        save_quick(QString::fromStdString(platform::path_text(folder / "out")),
                   {file("a.json", {}), file("a.csv", "Could not write a.csv"), file("a.png", {})});
    ASSERT_EQ(asked.size(), 2U);  // not the PNG, after the CSV failed
    ASSERT_EQ(outcome.files.size(), 1);
    EXPECT_TRUE(outcome.files.front().endsWith("a.json"));
    EXPECT_EQ(outcome.error.toStdString(), "Could not write a.csv");
}

}  // namespace
}  // namespace vml::app
