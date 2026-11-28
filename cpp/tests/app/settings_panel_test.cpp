// The Settings tab while metrics are calculated: it stays open, with only
// what a running calculation uses locked, and a note saying so.
#include <QAbstractSpinBox>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QLabel>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateEdit>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QTest>
#include <QtTest/qtestwheel.h>

#include <QSignalSpy>

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "dialogs.h"
#include "file_writes.h"
#include "segment_rules_editor.h"
#include "metric_list.h"
#include "in_place_name.h"
#include "settings_panel.h"
#include "app_dialog.h"
#include "test_videos.h"
#include "vml/analysis_request.h"
#include "vml/metric_cache.h"
#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"
#include "vml/ffmpeg_tools.h"
#include "vml/notifications.h"
#include "vml/settings.h"
#include "widgets.h"

namespace vml::app {
namespace {

// A name typed in the profile field and Enter pressed (in_place_name.h).
void type_name(InPlaceName* field, const QString& name) {
    ASSERT_TRUE(field->asking());
    field->line_edit()->setText(name);
    field->commit();
}

// Why the name typed was refused, said beside the field; empty: it was not.
QString refused(const InPlaceName* field) { return field->problem_line()->text(); }

// While a run is active the ffmpeg folder, the saved results' folder and
// deleting saved results are locked, each telling why; the export folder (and
// the rest) can still be changed; the note shows. When the run ends all of
// it is as before, the tooltips too.
TEST(SettingsPanel, ARunLocksOnlyWhatItUsesAndSaysSo) {
    Settings settings;
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    auto* ffmpeg = panel.findChild<QLineEdit*>("ffmpegFolder");
    auto* saved_results = panel.findChild<QLineEdit*>("savedResultsFolder");
    auto* clear = panel.findChild<QPushButton*>("deleteSavedResults");
    auto* delete_old = panel.findChild<QComboBox*>("deleteSavedScope");
    auto* note = panel.findChild<QLabel*>("runLockNote");
    ASSERT_TRUE(ffmpeg && saved_results && clear && delete_old && note);
    QLineEdit* export_folder = nullptr;
    for (QLineEdit* edit : panel.findChildren<QLineEdit*>()) {
        if (edit != ffmpeg && edit != saved_results && edit != panel.accent_hex &&
            qobject_cast<QAbstractSpinBox*>(edit->parentWidget()) == nullptr) {  // not a spin box's own
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
    EXPECT_FALSE(delete_old->isEnabled());
    EXPECT_TRUE(export_folder->isEnabled());
    EXPECT_TRUE(note->isVisibleTo(&panel));
    EXPECT_TRUE(note->text().startsWith(QStringLiteral("\U0001F512 Metrics are being calculated")));  // Classic's padlock
    EXPECT_EQ(note->property("vmlRole").toString(), "notice");

    panel.set_run_active(false);
    EXPECT_TRUE(ffmpeg->isEnabled());
    EXPECT_TRUE(saved_results->isEnabled());
    EXPECT_TRUE(clear->isEnabled());
    EXPECT_EQ(clear->toolTip(), "its own tip");
    EXPECT_TRUE(delete_old->isEnabled());
    EXPECT_FALSE(note->isVisibleTo(&panel));
}

// Settings > Saved Results' Delete...: the count says how many there are and
// when they were saved; the row says what its scope would delete before it
// is clicked; the question says it too, and how many can't be scored again;
// and it deletes them all, those whose reference is gone too.
TEST(SettingsPanel, DeleteSaysWhatItDeletesAndDeletesIt) {
    const test::Folder folder;
    Settings settings;
    settings.cache_dir = (folder / "cache").string();
    std::map<std::string, std::filesystem::path> saved;
    for (const char* name : {"old", "new"}) {
        std::ofstream(folder / (std::string(name) + ".mkv")) << name;
        std::ofstream(folder / "t.mkv") << "t";
        ComparisonResult result;
        result.source = folder / (std::string(name) + ".mkv");
        result.distorted = folder / "t.mkv";
        result.fps = 24.0;
        FrameMetricResult vmaf;
        vmaf.key = "vmaf";
        vmaf.frame = {0, 1};
        vmaf.time = {0.0, 1.0 / 24.0};
        vmaf.values = {90.0F, 80.0F};
        result.metric_results.add(std::move(vmaf));
        const AnalysisRequest request =
            analysis_request(VmafOptions{}, {"vmaf"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
        saved[name] = store_result(folder / "cache", result.source, result.distorted, request.recipe, result, name,
                                   request.metrics);
    }
    std::filesystem::remove(folder / "old.mkv");  // its reference gone: not to be scored again
    for (const auto& entry : std::filesystem::directory_iterator(saved["old"])) {
        std::filesystem::last_write_time(entry.path(),
                                         std::filesystem::last_write_time(entry.path()) - std::chrono::days(40));
    }
    FileWrites writes;
    Dialogs dialogs = standard_dialogs(nullptr);
    Message asked;
    dialogs.message = [&asked](const Message& message) {
        asked = message;
        return MessageAnswer{0, false};  // Delete saved results
    };
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    panel.refresh_status();
    panel.wait_for_count();
    auto* summary = panel.findChild<QLabel*>("savedResultsSummary");
    ASSERT_NE(summary, nullptr);
    EXPECT_TRUE(summary->text().startsWith("2 saved results, ")) << summary->text().toStdString();
    EXPECT_TRUE(summary->text().contains(" to ")) << summary->text().toStdString();  // two dates

    panel.delete_scope->setCurrentIndex(panel.delete_scope->findData("all"));
    EXPECT_TRUE(panel.delete_preview->text().startsWith("Would delete 2 comparisons'"))
        << panel.delete_preview->text().toStdString();
    panel.delete_scope->setCurrentIndex(panel.delete_scope->findData("older"));
    panel.delete_days->setValue(30);
    EXPECT_TRUE(panel.delete_preview->text().startsWith("Would delete 1 comparison's"))
        << panel.delete_preview->text().toStdString();
    EXPECT_FALSE(panel.delete_days->isHidden());  // the scope's own control, the dates' hidden
    EXPECT_TRUE(panel.delete_from->isHidden());
    EXPECT_EQ(panel.delete_button->property("vmlRole").toString(), "danger");

    QSignalSpy idle(&writes, &FileWrites::became_idle);
    panel.delete_button->click();
    panel.wait_for_deletion_count();
    EXPECT_EQ(asked.kind, MessageKind::danger);
    EXPECT_TRUE(asked.headline.contains("1 comparison are deleted")) << asked.headline.toStdString();
    EXPECT_TRUE(asked.text.contains("1 of them can't be scored again")) << asked.text.toStdString();
    EXPECT_EQ(asked.default_button, 1);  // Cancel
    EXPECT_TRUE(idle.count() > 0 || idle.wait(30000));
    QCoreApplication::processEvents();
    EXPECT_FALSE(std::filesystem::exists(saved["old"]));
    EXPECT_TRUE(std::filesystem::exists(saved["new"] / "context.json"));
    // Said under Delete..., where it was asked for, past the count after it.
    panel.wait_for_count();
    EXPECT_EQ(panel.delete_preview->text(), "Deleted the saved scores of 1 comparison.");
    EXPECT_TRUE(panel.delete_preview->styleSheet().contains("font-weight: 700"));  // at first; then it fades to muted
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
    EXPECT_EQ(names,
              (std::vector<QString>{"General", "Appearance", "Metrics", "Weak segments", "Saved Results", "Storage",
                                    "Notifications"}));
    const auto section_of = [&panel](const QWidget* control) {
        for (int index = 0; index < panel.sections->count(); ++index) {
            if (panel.sections->widget(index)->isAncestorOf(control)) {
                return panel.sections->tabText(index);
            }
        }
        return QString();
    };
    EXPECT_EQ(section_of(panel.ffmpeg_edit), "General");
    EXPECT_EQ(section_of(panel.theme_buttons->buttons().front()), "Appearance");
    EXPECT_EQ(section_of(panel.accent_hex), "Appearance");
    EXPECT_EQ(section_of(panel.decoded_videos), "General");
    EXPECT_EQ(section_of(panel.default_gpu), "Metrics");
    EXPECT_EQ(section_of(panel.gpu_together), "Metrics");
    EXPECT_EQ(section_of(panel.weak_profile), "Weak segments");
    EXPECT_EQ(section_of(panel.cache_edit), "Saved Results");
    EXPECT_EQ(section_of(panel.use_cache), "Saved Results");
    EXPECT_EQ(section_of(panel.saved_versions_used), "Saved Results");
    EXPECT_EQ(section_of(panel.export_edit), "Storage");
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
    // Every key field stores its setting of secret_settings, the one list
    // of them (#221's review: the panel kept a list of its own).
    const QList<QLineEdit*> keys = panel.findChildren<QLineEdit*>();
    int key_fields = 0;
    for (QLineEdit* edit : keys) {
        if (edit->echoMode() == QLineEdit::Password) {
            edit->setText(QStringLiteral("key-%1").arg(++key_fields));
            emit edit->editingFinished();
        }
    }
    EXPECT_EQ(static_cast<std::size_t>(key_fields), secret_settings().size());
    for (const auto& [name, member] : secret_settings()) {
        EXPECT_TRUE(secret_of(settings.*member).value_or("").starts_with("key-")) << name;
    }
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
// Settings > Appearance shows the interface, the accent and light or dark
// chosen, and saves each as it is picked: the interface by its card, the
// accent by a disc, a code typed, or the system's (its code then shown),
// light or dark by its segment. The accent's choices are usable only while
// VML Modern is picked. (VML Modern is not picked here: applied, it would be
// every later test's style.)
// What an edit did goes to the window's indicator (notify), not a line at
// the tab's foot; a setting that shows only after a restart asks for one.
TEST(SettingsPanel, SaysWhatAnEditDidAndWhenARestartIsNeeded) {
    Settings settings;
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    QStringList said;
    panel.notify = [&said](const QString& text, ExportIndicator::Note) { said << text; };
    QSignalSpy restart(&panel, &SettingsPanel::restart_needed);
    panel.use_cache->toggle();
    EXPECT_EQ(said, QStringList{"Settings saved."});
    EXPECT_EQ(restart.count(), 0);
    const int started = panel.language->currentIndex();
    panel.language->setCurrentIndex(started == 1 ? 2 : 1);
    ASSERT_EQ(restart.count(), 1);
    EXPECT_TRUE(restart.at(0).at(0).toString().contains("next started"));
    // Back to the language the app started in: no restart needed, said with
    // nothing, which takes the window's note away.
    panel.language->setCurrentIndex(started);
    ASSERT_EQ(restart.count(), 2);
    EXPECT_TRUE(restart.at(1).at(0).toString().isEmpty());
}

TEST(SettingsPanel, AppearanceSavesEachChoice) {
    Settings settings;
    settings.interface_style = "modern";
    settings.accent_colour = "#4f8fe8";
    FileWrites writes;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SettingsPanel panel(settings, writes, [] { return ToolsStatus{}; }, dialogs);
    EXPECT_EQ(panel.interface_buttons->checkedId(), 1);
    ASSERT_NE(panel.accent_buttons->checkedButton(), nullptr);
    EXPECT_EQ(panel.accent_buttons->checkedButton()->property("hex").toString(), "#4f8fe8");
    EXPECT_EQ(panel.accent_hex->text(), "#4F8FE8");
    EXPECT_TRUE(panel.accent_controls->isEnabled());
    for (QAbstractButton* choice : panel.accent_buttons->buttons()) {
        // Every colour in a disc's sheet filled in: one left as {name} drops its rule (the ring).
        EXPECT_FALSE(choice->styleSheet().contains(QRegularExpression(R"(\{[a-z_]+\})")))
            << choice->styleSheet().toStdString();
    }

    panel.interface_buttons->button(0)->click();
    EXPECT_EQ(settings.interface_style, "classic");
    EXPECT_FALSE(panel.accent_controls->isEnabled());
    EXPECT_FALSE(panel.look_controls->isEnabled());
    // Usable again by hand: picking VML Modern would apply it.
    panel.accent_controls->setEnabled(true);
    panel.look_controls->setEnabled(true);  // VML Modern's alone, as the accent is

    panel.accent_hex->setText("123ABC");
    emit panel.accent_hex->editingFinished();
    EXPECT_EQ(settings.accent_colour, "#123abc");
    EXPECT_EQ(panel.accent_buttons->checkedButton(), nullptr);  // one of one's own: no disc
    EXPECT_EQ(panel.accent_hex->text(), "#123ABC");
    panel.accent_hex->setText("#12");
    emit panel.accent_hex->editingFinished();
    EXPECT_EQ(settings.accent_colour, "#123abc");  // not a colour: kept
    EXPECT_EQ(panel.accent_hex->text(), "#123ABC");

    QAbstractButton* system = panel.findChild<QAbstractButton*>("systemAccent");
    ASSERT_NE(system, nullptr);
    system->click();
    EXPECT_EQ(settings.accent_colour, "");
    EXPECT_EQ(panel.accent_hex->text(), QApplication::palette().color(QPalette::Accent).name().toUpper());

    const QList<QAbstractButton*> modes = panel.theme_buttons->buttons();
    ASSERT_EQ(modes.size(), 3);  // the system's, Light, Dark
    EXPECT_TRUE(modes[0]->isChecked());
    const QList<QAbstractButton*> light_looks = panel.light_look_buttons->buttons();
    const QList<QAbstractButton*> dark_looks = panel.dark_look_buttons->buttons();
    ASSERT_EQ(light_looks.size(), 4);  // Light, Dim, Paper, Desert
    ASSERT_EQ(dark_looks.size(), 5);   // Dark, Mocha, Joker, Nitrate, Phosphor
    EXPECT_TRUE(light_looks[0]->isChecked());
    EXPECT_TRUE(dark_looks[0]->isChecked());
    modes[1]->click();
    EXPECT_EQ(settings.theme, "light");
    modes[2]->click();
    EXPECT_EQ(settings.theme, "dark");
#ifdef __APPLE__
    const QString system_name = "Same as macOS";
#else
    const QString system_name = "Same as Windows";
#endif
    EXPECT_EQ(system->text(), system_name);
    // The accent's first choice is the look's own while the theme shows a
    // look with one: Mocha in Dark, Desert in Light.
    dark_looks[1]->click();
    EXPECT_EQ(settings.dark_look, "mocha");
    EXPECT_EQ(system->text(), "Theme default");  // Mocha's amber, not the system's
    dark_looks[2]->click();
    EXPECT_EQ(settings.dark_look, "joker");
    EXPECT_EQ(system->text(), "Theme default");  // Joker's green
    dark_looks[3]->click();
    EXPECT_EQ(settings.dark_look, "nitrate");
    EXPECT_EQ(system->text(), "Theme default");  // Nitrate's gold
    dark_looks[4]->click();
    EXPECT_EQ(settings.dark_look, "phosphor");
    EXPECT_EQ(system->text(), "Theme default");  // Phosphor's gold
    EXPECT_FALSE(system->toolTip().isEmpty());
    modes[1]->click();
    EXPECT_EQ(settings.theme, "light");
    EXPECT_EQ(system->text(), system_name);  // Light's own look: the system's
    light_looks[3]->click();
    EXPECT_EQ(settings.light_look, "desert");
    EXPECT_EQ(system->text(), "Theme default");  // Desert's burnt orange
    light_looks[2]->click();
    EXPECT_EQ(settings.light_look, "paper");
    EXPECT_EQ(system->text(), system_name);
    modes[0]->click();
    EXPECT_EQ(settings.theme, "");
    // The title bar's menu: a theme chosen there as by its button here, the
    // accent's first choice named for the look it shows (Mocha's amber).
    panel.choose_look(true, "mocha");
    modes[1]->click();
    EXPECT_EQ(system->text(), system_name);
    panel.choose_theme("dark");
    EXPECT_EQ(settings.theme, "dark");
    EXPECT_EQ(system->text(), "Theme default");
    // A look chosen there as by its button here.
    panel.choose_look(true, "joker");
    EXPECT_EQ(settings.dark_look, "joker");
    EXPECT_TRUE(dark_looks[2]->isChecked());
    panel.choose_look(false, "dim");
    EXPECT_EQ(settings.light_look, "dim");
}

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
// saved and told to the window (for the Worst Frames tab). New...
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
    type_name(panel.weak_profile_name, answers.takeFirst());
    EXPECT_EQ(settings.weak_segment_profile, "Film");
    EXPECT_EQ(changed.count(), 2);
    EXPECT_EQ(panel.weak_profile->currentText(), "Film");
    EXPECT_TRUE(panel.weak_profile_delete->isEnabled());
    panel.weak_segments->set_rules(SegmentRules{{{"ssimulacra2", '<', 60.0}}});
    panel.weak_segments->add_condition({"vmaf", '<', 85.0});  // an edit of Film's
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Film")->rules.conditions),
              (std::vector<std::string>{"ssimulacra2<60", "vmaf<85"}));
    EXPECT_EQ(settings.weak_segment_conditions.size(), 3U);  // Default's as they were

    panel.rename_weak_profile();  // "default": refused, beside the field, the name still being typed
    type_name(panel.weak_profile_name, answers.takeFirst());
    EXPECT_FALSE(refused(panel.weak_profile_name).isEmpty());
    EXPECT_TRUE(panel.weak_profile_name->asking());
    panel.weak_profile_name->cancel();  // Esc: the list back, nothing kept
    EXPECT_FALSE(panel.weak_profile_name->asking());
    EXPECT_TRUE(refused(panel.weak_profile_name).isEmpty());
    panel.rename_weak_profile();  // "Film, 35mm": a comma, refused
    type_name(panel.weak_profile_name, answers.takeFirst());
    EXPECT_FALSE(refused(panel.weak_profile_name).isEmpty());
    EXPECT_EQ(settings.weak_segment_profile, "Film");
    panel.rename_weak_profile();  // "Grain", typed over the refused name
    type_name(panel.weak_profile_name, answers.takeFirst());
    EXPECT_FALSE(panel.weak_profile_name->asking());
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
    type_name(panel.weak_profile_name, answers.takeFirst());
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
// A condition is edited as it was written: vmaf>98 (what is better than it
// need be) stays above, not turned to the metric's usual "below".
TEST(SegmentRulesEditor, KeepsEachConditionsDirection) {
    SegmentRulesEditor editor({"vmaf", "butteraugli"}, false);
    editor.set_rules(SegmentRules{{{"vmaf", '>', 98.0}, {"butteraugli", '>', 2.5}, {"vmaf", '<', 80.0}}});
    EXPECT_EQ(condition_texts(editor.rules().conditions),
              (std::vector<std::string>{"vmaf>98", "butteraugli>2.5", "vmaf<80"}));
}

// A value written with more decimals than the metric shows keeps them: the
// box shows as many as it has (up to 6), and the editor gives it back
// whole, rather than rounded to VMAF's 2 and saved so at the next edit.
TEST(SegmentRulesEditor, KeepsAValuesOwnDecimals) {
    EXPECT_EQ(threshold_decimals(98.0, 2), 2);
    EXPECT_EQ(threshold_decimals(80.125, 2), 3);
    EXPECT_EQ(threshold_decimals(0.5, 4), 4);
    EXPECT_EQ(threshold_decimals(0.123456789, 4), 6);
    SegmentRulesEditor editor({"vmaf", "ssim"}, false);
    editor.set_rules(SegmentRules{{{"vmaf", '<', 80.125}, {"ssim", '<', 0.987654}, {"vmaf", '<', 90.0}}});
    EXPECT_EQ(condition_texts(editor.rules().conditions),
              (std::vector<std::string>{"vmaf<80.125", "ssim<0.987654", "vmaf<90"}));
}

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
