// The Settings tab while metrics are calculated: it stays open, with only
// what a running calculation uses locked, and a note saying so.
#include <QApplication>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QTest>
#include <QtTest/qtestwheel.h>

#include <QSignalSpy>

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "dialogs.h"
#include "file_writes.h"
#include "segment_rules_editor.h"
#include "settings_panel.h"
#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"
#include "vml/ffmpeg_tools.h"
#include "vml/settings.h"
#include "widgets.h"

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
    QStringList answers{"Film", "default", "Film, 35mm", "Grain"};
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
    alerted.clear();
    panel.rename_weak_profile();  // "Film, 35mm": a comma, refused
    EXPECT_EQ(alerted, "Profile not renamed");
    EXPECT_EQ(settings.weak_segment_profile, "Film");
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

    // From preset...: a profile named after the preset (numbered where the
    // name is taken), with its rules, in use.
    panel.new_weak_profile_from_preset("animation");
    panel.new_weak_profile_from_preset("animation");
    EXPECT_EQ(settings.weak_segment_profile, "Animation 2");
    EXPECT_EQ(panel.weak_profile->currentText(), "Animation 2");
    ASSERT_TRUE(segment_profile_named(settings, "Animation"));
    EXPECT_EQ(condition_texts(panel.weak_segments->rules().conditions),
              condition_texts(segment_preset("animation")->rules.conditions));

    // Saved by the window (Default, put back in use): shown, not an edit.
    const int edits = changed.count();
    settings.weak_segment_profile.clear();
    settings.weak_segment_conditions = {"ssimulacra2<70"};
    panel.show_weak_segment_rules();
    EXPECT_EQ(condition_texts(panel.weak_segments->rules().conditions), std::vector<std::string>{"ssimulacra2<70"});
    EXPECT_EQ(changed.count(), edits);
}

// The wheel over a spin box or a combo box of the Settings tab scrolls the
// tab: their values stay as they were, and the focus stays where it was. A
// box elsewhere keeps the wheel.
TEST(SettingsPanel, TheWheelScrollsPastValueBoxes) {
    auto* window = new QWidget;
    auto* window_layout = new QVBoxLayout(window);
    auto* elsewhere = new QSpinBox;  // not the page's
    elsewhere->setValue(5);
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* typed = new QLineEdit;  // where the focus is
    auto* spin = new QSpinBox;
    spin->setValue(5);
    auto* combo = new QComboBox;
    combo->addItems({"first", "second", "third"});
    layout->addWidget(typed);
    layout->addWidget(spin);
    layout->addWidget(combo);
    new WheelPastValueBoxes(page);  // as SettingsPanel makes it: once its boxes exist, before it is shown
    window_layout->addWidget(elsewhere);
    window_layout->addWidget(page);
    window_layout->addSpacing(2000);
    ScrollingPage scrolling(window);
    scrolling.resize(300, 200);
    scrolling.show();
    scrolling.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&scrolling));
    typed->setFocus();
    QScrollBar* bar = scrolling.verticalScrollBar();
    ASSERT_GT(bar->maximum(), 0);
    for (QWidget* box : {static_cast<QWidget*>(spin), static_cast<QWidget*>(combo)}) {
        bar->setValue(0);
        // As the system sends it: Qt passes a wheel event the app makes
        // itself to no parent.
        QTest::wheelEvent(scrolling.windowHandle(), box->mapTo(&scrolling, box->rect().center()), QPoint(0, -120));
        EXPECT_GT(bar->value(), 0);
        EXPECT_TRUE(typed->hasFocus());
    }
    EXPECT_EQ(spin->value(), 5);
    EXPECT_EQ(combo->currentIndex(), 0);
    bar->setValue(0);
    QTest::wheelEvent(scrolling.windowHandle(), elsewhere->mapTo(&scrolling, elsewhere->rect().center()),
                      QPoint(0, -120));
    EXPECT_EQ(elsewhere->value(), 4);
    EXPECT_EQ(bar->value(), 0);
    // Closed before it goes: Qt's offscreen window, destroyed right after a
    // wheel event, never returned.
    scrolling.hide();
    QCoreApplication::processEvents();
}

}  // namespace
}  // namespace vml::app
