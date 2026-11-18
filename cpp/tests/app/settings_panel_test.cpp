// The Settings tab while metrics are calculated: it stays open, with only
// what a running calculation uses locked, and a note saying so.
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

#include <gtest/gtest.h>

#include "dialogs.h"
#include "file_writes.h"
#include "settings_panel.h"
#include "vml/ffmpeg_tools.h"
#include "vml/settings.h"

namespace vml::app {
namespace {

// While a run is active the ffmpeg folder, the saved results' folder and
// Clear saved results are locked, each telling why; the export folder (and
// the rest) can still be changed; the note shows. When the run ends all of
// it is as before, the tooltips too.
TEST(SettingsPanel, ARunLocksOnlyWhatItUsesAndSaysSo) {
    Settings settings;
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    auto* ffmpeg = panel.findChild<QLineEdit*>("ffmpegFolder");
    auto* saved_results = panel.findChild<QLineEdit*>("savedResultsFolder");
    auto* clear = panel.findChild<QPushButton*>("clearSavedResults");
    auto* note = panel.findChild<QLabel*>("runLockNote");
    ASSERT_TRUE(ffmpeg && saved_results && clear && note);
    QLineEdit* export_folder = nullptr;
    for (QLineEdit* edit : panel.findChildren<QLineEdit*>()) {
        if (edit != ffmpeg && edit != saved_results) {
            export_folder = edit;
        }
    }
    ASSERT_NE(export_folder, nullptr);
    clear->setToolTip("its own tip");

    panel.set_run_active(true);
    EXPECT_FALSE(ffmpeg->isEnabled());
    EXPECT_FALSE(saved_results->isEnabled());
    EXPECT_FALSE(clear->isEnabled());
    EXPECT_TRUE(clear->toolTip().contains("Locked while metrics are calculated"));
    EXPECT_TRUE(export_folder->isEnabled());
    EXPECT_TRUE(note->isVisibleTo(&panel));

    panel.set_run_active(false);
    EXPECT_TRUE(ffmpeg->isEnabled());
    EXPECT_TRUE(saved_results->isEnabled());
    EXPECT_TRUE(clear->isEnabled());
    EXPECT_EQ(clear->toolTip(), "its own tip");
    EXPECT_FALSE(note->isVisibleTo(&panel));
}

}  // namespace
}  // namespace vml::app
