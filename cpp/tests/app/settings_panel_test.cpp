// The Settings tab while metrics are calculated: it stays open, with only
// what a running calculation uses locked, and a note saying so.
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

#include <QSignalSpy>

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "dialogs.h"
#include "file_writes.h"
#include "segment_rules_editor.h"
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

// Settings > Weak segments shows the saved rules; an edit there is saved and
// told to the window (for Metric Graphs' Weak segments); rules the window
// saved (Weak segments' Save as default) are shown without counting as an
// edit.
TEST(SettingsPanel, WeakSegmentsEditsTheSavedRules) {
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<90", "butteraugli>2.5"};
    settings.weak_segment_merge_frames = 4;
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    QSignalSpy changed(&panel, &SettingsPanel::weak_segment_rules_changed);
    const SegmentRules shown = panel.weak_segments->rules();
    EXPECT_EQ(condition_texts(shown.conditions), (std::vector<std::string>{"vmaf<90", "butteraugli>2.5"}));
    EXPECT_EQ(shown.merge_frames, 4);

    panel.weak_segments->add_condition({"psnr", '<', 30.0});
    EXPECT_EQ(settings.weak_segment_conditions, (std::vector<std::string>{"vmaf<90", "butteraugli>2.5", "psnr<30"}));
    EXPECT_EQ(changed.count(), 1);

    settings.weak_segment_conditions = {"ssimulacra2<70"};
    panel.show_weak_segment_rules();
    EXPECT_EQ(condition_texts(panel.weak_segments->rules().conditions), std::vector<std::string>{"ssimulacra2<70"});
    EXPECT_EQ(changed.count(), 1);
}

}  // namespace
}  // namespace vml::app
