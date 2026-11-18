// The Settings tab while metrics are calculated: it stays open, with only
// what a running calculation uses locked, and a note saying so.
#include <QLabel>
#include <QComboBox>
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
#include "vml/segment_profiles.h"
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

// Settings > Weak segments shows the profile in use and edits it; an edit is
// saved and told to the window (for Metric Graphs' Weak segments). New...
// starts a profile from the one shown and puts it in use; Rename... and
// Delete work on the one in use (Default has neither); a name taken is
// refused. Profiles the window saved are shown without counting as an edit.
TEST(SettingsPanel, WeakSegmentsEditsTheProfiles) {
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<90", "butteraugli>2.5"};
    settings.weak_segment_merge_frames = 4;
    FileWrites writes;
    Dialogs dialogs = standard_dialogs(nullptr);
    QStringList answers{"Film", "default", "Grain"};
    QString alerted;
    dialogs.ask_text = [&answers](const QString&, const QString&, const QString&) -> std::optional<QString> {
        return answers.takeFirst();
    };
    dialogs.alert = [&alerted](const QString& title, const QString&) { alerted = title; };
    dialogs.ask_no_default = [](const QString&, const QString&) { return true; };
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    QSignalSpy changed(&panel, &SettingsPanel::weak_segment_rules_changed);
    EXPECT_EQ(panel.weak_profile->currentText(), "Default");
    EXPECT_FALSE(panel.weak_profile_rename->isEnabled());
    EXPECT_FALSE(panel.weak_profile_delete->isEnabled());
    const SegmentRules shown = panel.weak_segments->rules();
    EXPECT_EQ(condition_texts(shown.conditions), (std::vector<std::string>{"vmaf<90", "butteraugli>2.5"}));
    EXPECT_EQ(shown.merge_frames, 4);

    panel.weak_segments->add_condition({"psnr", '<', 30.0});
    EXPECT_EQ(settings.weak_segment_conditions, (std::vector<std::string>{"vmaf<90", "butteraugli>2.5", "psnr<30"}));
    EXPECT_EQ(changed.count(), 1);

    panel.new_weak_profile();  // "Film": Default's rules, in use, told to the window
    EXPECT_EQ(settings.weak_segment_profile, "Film");
    EXPECT_EQ(changed.count(), 2);
    EXPECT_EQ(panel.weak_profile->currentText(), "Film");
    EXPECT_TRUE(panel.weak_profile_delete->isEnabled());
    panel.weak_segments->set_rules(SegmentRules{{{"ssimulacra2", '<', 60.0}}});
    panel.weak_segments->add_condition({"vmaf", '<', 85.0});  // an edit of Film's
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Film")->rules.conditions),
              (std::vector<std::string>{"ssimulacra2<60", "vmaf<85"}));
    EXPECT_EQ(settings.weak_segment_conditions.size(), 3U);  // Default's as they were

    panel.rename_weak_profile();  // "default": refused
    EXPECT_EQ(alerted, "Profile not renamed");
    panel.rename_weak_profile();  // "Grain"
    EXPECT_EQ(settings.weak_segment_profile, "Grain");
    EXPECT_FALSE(segment_profile_named(settings, "Film"));

    int told = changed.count();
    panel.weak_profile->setCurrentIndex(0);  // Default picked: in use, shown, told
    EXPECT_EQ(settings.weak_segment_profile, "");
    EXPECT_EQ(changed.count(), told + 1);
    EXPECT_EQ(panel.weak_segments->rules().conditions.size(), 3U);
    panel.weak_profile->setCurrentIndex(1);
    panel.delete_weak_profile();
    EXPECT_EQ(segment_profiles(settings).size(), 1U);
    EXPECT_EQ(panel.weak_profile->currentText(), "Default");

    const int edits = changed.count();
    settings.weak_segment_conditions = {"ssimulacra2<70"};
    panel.show_weak_segment_rules();
    EXPECT_EQ(condition_texts(panel.weak_segments->rules().conditions), std::vector<std::string>{"ssimulacra2<70"});
    EXPECT_EQ(changed.count(), edits);
}

}  // namespace
}  // namespace vml::app
