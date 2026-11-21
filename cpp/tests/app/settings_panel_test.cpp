// The Settings tab while metrics are calculated: it stays open, with only
// what a running calculation uses locked, and a note saying so.
#include <QApplication>
#include <QCheckBox>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QPushButton>
#include <QScrollBar>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTabWidget>
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
#include "vml/notifications.h"
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

// Settings is in sub-tabs: General, Metrics, Weak segments, Storage and
// Notifications, each
// with its own settings. The note on what a run locks is above them, not on
// one of them, so it shows whichever is open.
TEST(SettingsPanel, SubTabsHoldTheSettings) {
    Settings settings;
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    ASSERT_NE(panel.sections, nullptr);
    std::vector<QString> names;
    for (int index = 0; index < panel.sections->count(); ++index) {
        names.push_back(panel.sections->tabText(index));
    }
    EXPECT_EQ(names, (std::vector<QString>{"General", "Metrics", "Weak segments", "Storage", "Notifications"}));
    const auto section_of = [&panel](const QWidget* control) {
        for (int index = 0; index < panel.sections->count(); ++index) {
            if (panel.sections->widget(index)->isAncestorOf(control)) {
                return panel.sections->tabText(index);
            }
        }
        return QString();
    };
    EXPECT_EQ(section_of(panel.ffmpeg_edit), "General");
    EXPECT_EQ(section_of(panel.theme), "General");
    EXPECT_EQ(section_of(panel.decoded_videos), "General");
    EXPECT_EQ(section_of(panel.default_gpu), "Metrics");
    EXPECT_EQ(section_of(panel.gpu_together), "Metrics");
    EXPECT_EQ(section_of(panel.weak_profile), "Weak segments");
    EXPECT_EQ(section_of(panel.cache_edit), "Storage");
    EXPECT_EQ(section_of(panel.log_label), "Storage");
    EXPECT_EQ(section_of(panel.run_lock_note), QString());
    EXPECT_TRUE(panel.isAncestorOf(panel.run_lock_note));
}

// Settings > Notifications: a key typed is saved as the storage chosen says,
// encrypted where the system can, and stored again when the choice changes;
// a preset sets the three templates, and the preview follows them.
TEST(SettingsPanel, NotificationsStoreKeysAsChosenAndPreviewTheMessage) {
    Settings settings;
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    panel.pushover_user->setText("user-key");
    emit panel.pushover_user->editingFinished();
    EXPECT_EQ(secret_of(settings.pushover_user), "user-key");
    EXPECT_EQ(secret_is_encrypted(settings.pushover_user), secrets_can_be_encrypted());
    panel.secret_storage->setCurrentIndex(1);  // as plain text
    EXPECT_FALSE(settings.notify_secrets_encrypted);
    EXPECT_EQ(settings.pushover_user, "user-key");
    EXPECT_TRUE(panel.notify_preview->text().contains("encode-crf18.mkv"));  // Standard names the videos

    QMenu* presets = panel.notify_presets->menu();
    ASSERT_NE(presets, nullptr);
    presets->actions().front()->trigger();  // Status only
    EXPECT_EQ(settings.notify_title, "{app}: {outcome}");
    EXPECT_EQ(settings.notify_message, "{duration}");
    EXPECT_EQ(settings.notify_video_line, "");
    EXPECT_FALSE(panel.notify_preview->text().contains("encode-crf18.mkv"));
    EXPECT_TRUE(panel.notify_preview->text().contains("2:14:09"));
}

// The services: a list of all of them, its tick and the page's "Send to"
// box one setting either way; a page's keys stored as the others; a page
// for each, shown as its row is chosen.
TEST(SettingsPanel, NotificationServicesAreListedAndTickedEitherWay) {
    Settings settings;
    settings.gotify_url = "http://gotify.lan";
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    ASSERT_EQ(panel.notify_services->count(), static_cast<int>(notification_services().size()));
    QListWidgetItem* gotify = panel.notify_services->item(2);
    EXPECT_EQ(gotify->text(), "Gotify");
    gotify->setCheckState(Qt::Checked);
    EXPECT_TRUE(panel.gotify_enabled->isChecked());
    EXPECT_TRUE(settings.gotify_enabled);
    panel.gotify_enabled->setChecked(false);
    EXPECT_EQ(gotify->checkState(), Qt::Unchecked);
    EXPECT_FALSE(settings.gotify_enabled);

    panel.notify_services->setCurrentRow(2);
    EXPECT_TRUE(panel.gotify_url->isVisibleTo(panel.service_pages));
    EXPECT_EQ(panel.gotify_url->text(), "http://gotify.lan");
    panel.gotify_token->setText("app-token");
    emit panel.gotify_token->editingFinished();
    EXPECT_EQ(secret_of(settings.gotify_token), "app-token");
    EXPECT_EQ(secret_is_encrypted(settings.gotify_token), secrets_can_be_encrypted());
    panel.discord_mention_when->setCurrentIndex(1);
    EXPECT_EQ(settings.discord_mention_when, "always");
}

// Settings > Storage > Exports include: a box for each kind of export, in
// export_item_keys' order, showing the saved choice; a box for each format
// (JSON, CSV); and a .zip or loose files; After each run: on or off, and
// where (only to choose while on).
// Each edit is saved.
TEST(SettingsPanel, ExportsAreChosenOneByOne) {
    Settings settings;
    settings.export_items = {"graphs"};
    settings.export_formats = {"json"};
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    ASSERT_EQ(panel.export_item_boxes.size(), export_item_keys().size());
    EXPECT_TRUE(panel.export_item_boxes.at("graphs")->isChecked());
    EXPECT_FALSE(panel.export_item_boxes.at("results")->isChecked());
    panel.export_item_boxes.at("results")->setChecked(true);
    panel.export_item_boxes.at("weak_segments")->setChecked(true);
    EXPECT_EQ(settings.export_items, (std::vector<std::string>{"results", "graphs", "weak_segments"}));
    ASSERT_EQ(panel.export_format_boxes.size(), export_format_keys().size());
    EXPECT_TRUE(panel.export_format_boxes.at("json")->isChecked());
    EXPECT_FALSE(panel.export_format_boxes.at("csv")->isChecked());
    panel.export_format_boxes.at("csv")->setChecked(true);
    panel.export_format_boxes.at("json")->setChecked(false);
    EXPECT_EQ(settings.export_formats, std::vector<std::string>{"csv"});
    panel.export_form->setCurrentIndex(panel.export_form->findData(false));
    EXPECT_FALSE(settings.export_zip);
    EXPECT_FALSE(panel.after_run_place->isEnabled());
    panel.after_run_box->setChecked(true);
    EXPECT_TRUE(settings.after_run_export);
    EXPECT_TRUE(panel.after_run_place->isEnabled());
    panel.after_run_place->setCurrentIndex(panel.after_run_place->findData(QStringLiteral("folder")));
    EXPECT_EQ(settings.after_run_place, "folder");
    panel.export_item_boxes.at("graphs")->setChecked(false);
    EXPECT_EQ(settings.export_items, (std::vector<std::string>{"results", "weak_segments"}));
}

// Settings > Weak segments shows the profile in use and edits it; an edit is
// saved and told to the window (for Metric Graphs' Weak segments). New...
// starts a profile from the one shown and puts it in use; From preset loads
// a preset into the one in use; Rename... and
// Delete work on the one in use (Default has neither); a name taken is
// refused. Profiles the window saved are shown without counting as an edit.
TEST(SettingsPanel, WeakSegmentsEditsTheProfiles) {
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<90", "butteraugli>2.5"};
    settings.weak_segment_merge_frames = 4;
    FileWrites writes;
    Dialogs dialogs = standard_dialogs(nullptr);
    QStringList answers{"Film", "default", "Film, 35mm", "Grain", "Film"};
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

    // From preset...: the preset's rules into the profile in use, Default
    // here, saved and told as an edit. No profile is added, however often.
    const int before_preset = changed.count();
    panel.load_weak_preset("animation");
    panel.load_weak_preset("animation");
    EXPECT_EQ(segment_profiles(settings).size(), 1U);
    EXPECT_EQ(panel.weak_profile->count(), 1);
    EXPECT_EQ(settings.weak_segment_profile, "");
    EXPECT_EQ(panel.weak_profile->currentText(), "Default");
    const SegmentRules& animation = segment_preset("animation")->rules;
    EXPECT_EQ(condition_texts(panel.weak_segments->rules().conditions), condition_texts(animation.conditions));
    EXPECT_EQ(condition_texts(segment_profile_in_use(settings).rules.conditions),
              condition_texts(animation.conditions));
    EXPECT_EQ(settings.weak_segment_min_frames, animation.min_frames);
    EXPECT_EQ(settings.weak_segment_merge_frames, animation.merge_frames);
    EXPECT_GT(changed.count(), before_preset);
    // Into a named profile: its rules only, Default's left as they are.
    panel.new_weak_profile();  // "Film", from the rules shown
    panel.load_weak_preset("grainy-film");
    EXPECT_EQ(settings.weak_segment_profile, "Film");
    EXPECT_EQ(segment_profiles(settings).size(), 2U);
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Film")->rules.conditions),
              condition_texts(segment_preset("grainy-film")->rules.conditions));
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "")->rules.conditions),
              condition_texts(animation.conditions));
    panel.delete_weak_profile();
    EXPECT_EQ(panel.weak_profile->currentText(), "Default");

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
