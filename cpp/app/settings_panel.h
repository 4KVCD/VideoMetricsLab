// The Settings tab (the Python app's vmaf_app/ui/settings_panel.py): the
// app-wide settings, saved as soon as they are edited.
//
// The panel applies what is global itself: the theme.
// It tells the window, with signals, about the changes that concern the
// window: the defaults for newly added videos, how many test videos Video
// Compare decodes, and the FFmpeg folder, which the window then checks
// again. The saved-results folder needs nothing applied: the window reads it
// from the settings whenever it needs it.
#pragma once

#include <QString>
#include <QWidget>

#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dialogs.h"
#include "export_indicator.h"
#include "vml/ffmpeg_tools.h"
#include "vml/settings.h"

class QButtonGroup;
class QCheckBox;
class QRadioButton;
class QComboBox;
class QDateEdit;
class QGroupBox;
class QHBoxLayout;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QStackedWidget;
class QTabWidget;
class QTimer;
class QVariantAnimation;
template <typename T>
class QFutureWatcher;

namespace vml {
struct CacheSummary;
struct ClearedResults;
struct SavedComparison;
}

namespace vml::app {

class MetricList;

class FileWrites;
class Notifier;
class SegmentRulesEditor;

class SettingsPanel : public QWidget {
    Q_OBJECT

public:
    // Where the panel says what an action did (the window's tab bar
    // indicator, ExportIndicator::show_note), and how an export of the log
    // went (its show_outcome). Unset: nothing said (tests may set them).
    std::function<void(const QString& text, ExportIndicator::Note kind)> notify;
    ExportReport report_export;
    // `settings` are the window's own, which this panel edits and saves;
    // `file_writes` the window's queue of writes, which clearing the saved
    // results goes through; `tools` gives the window's latest check of
    // FFmpeg (each check starts two programs, so the panel does not check
    // again itself); `dialogs` the window's, which outlive the panel (its
    // tests answer them).
    SettingsPanel(Settings& settings, FileWrites& file_writes, std::function<ToolsStatus()> tools,
                  const Dialogs& dialogs, QWidget* parent = nullptr);
    ~SettingsPanel() override;

    // Shows the tools and the saved results as they are now. Called when the
    // tab is shown, when its folders change and when the saved results are
    // cleared. The saved results are counted on a thread of their own: a
    // thousand take 0.3 s, which would hold the window.
    void refresh_status();
    // While metrics are calculated (`active`), locks what a running
    // calculation uses (the ffmpeg folder, the saved results' folder, Clear
    // saved results) and says so at the top of the panel; the rest stays
    // editable. The window calls it as a run starts and ends.
    void set_run_active(bool active);
    // Settings.theme set to `value` ("", "light" or "dark") as its choice in
    // Settings > Appearance would: saved and applied (the title bar's theme
    // menu).
    void choose_theme(const QString& value);
    // Shows the export folder and Exports include as the settings have them
    // now: changed by the tab bar's Export panel.
    void show_export_choices();
    // Settings.light_look (`dark` false) or dark_look set to `value`, as its
    // button in Settings > Appearance would (the title bar's theme menu).
    void choose_look(bool dark, const QString& value);
    // Lists every CVVDP preset in Settings, the default selected. The window
    // calls it again when a preset is saved or deleted.
    void fill_cvvdp_default_combo();
    // Shows a FFmpeg folder the window has already saved (after the user
    // located ffmpeg.exe), without treating it as an edit.
    void show_ffmpeg_folder(const QString& directory);
    // Ticks or unticks a "Default metrics" box to match a default the window
    // has already saved (from the table's column heading), without treating
    // it as an edit.
    void show_default_metric(const std::string& key, bool checked);
    // Settings > Saved Results shown (the details panel's link).
    void show_saved_results_section();
    // Shows the film grain default the window has already saved (answered
    // "Don't ask again"), without treating it as an edit.
    void show_film_grain_default();
    // Settings' "Ask before closing the window during a run", after the
    // window's question was told not to ask again.
    void show_confirm_close();
    // Delete...: the saved results delete_scope names (every one, those last
    // saved more than delete_days ago, or from delete_from to delete_to),
    // after the question every deletion of them asks (saved_deletion.h),
    // which says what goes; read first on a thread of its own, the question
    // asked once it is (or at once, by the tests' wait_for_deletion_count).
    void delete_saved_results();
    void wait_for_deletion_count();
    // Waits for the saved results' count (refresh_status), and shows it.
    void wait_for_count();
    // Shows the weak segment profiles the window has already saved (Weak
    // segments' Save to profile), without treating it as an edit.
    void show_weak_segment_rules();
    // Settings > Weak segments' New..., Rename... and Delete (the tests call
    // them, with the dialogs answered).
    void new_weak_profile();
    // From preset...: preset `key`'s rules (segment_presets.h) into the
    // profile in use.
    void load_weak_preset(std::string_view key);
    void rename_weak_profile();
    void delete_weak_profile();

    // The controls, for the window's tests.
    // The sub-tabs, each a page of related boxes: General, Metrics, Weak
    // segments, Storage and Notifications.
    QTabWidget* sections = nullptr;
    QLineEdit* ffmpeg_edit = nullptr;
    // What a running calculation uses, locked while it runs (set_run_active),
    // and the note that says so.
    std::vector<QWidget*> run_locked_;
    QLabel* run_lock_note = nullptr;
    QLabel* ffmpeg_status = nullptr;
    QLineEdit* cache_edit = nullptr;
    QLabel* cache_summary = nullptr;
    // Settings > Saved Results' Saved scores box: a card for the score used
    // or for each, and the details panel's columns by their keys.
    QComboBox* saved_pick = nullptr;    // Use: which score (Settings::saved_score_pick)
    QComboBox* saved_device = nullptr;  // ...on which device first
    QComboBox* saved_engine = nullptr;  // ...by which engine first
    QRadioButton* saved_versions_used = nullptr;
    QRadioButton* saved_versions_each = nullptr;
    std::map<std::string, QCheckBox*> saved_details_boxes;
    // Delete...'s row: its scope ("all", "older", "between"), the days or
    // the dates it takes, what it would delete, and the button.
    QComboBox* delete_scope = nullptr;
    QSpinBox* delete_days = nullptr;
    QDateEdit* delete_from = nullptr;
    QDateEdit* delete_to = nullptr;
    QLabel* delete_preview = nullptr;
    QPushButton* delete_button = nullptr;
    QLineEdit* export_edit = nullptr;
    // Exported graphs' colours: the palette, and a chosen background and
    // text colour, each button's in its "chosen" property ("#rrggbb", or
    // empty for the palette's own).
    QComboBox* export_graph_theme = nullptr;
    // Exports include: a box for each kind of export (by its key in
    // export_item_keys), and whether as a .zip file (data true) or loose
    // files. After each run: whether, and where (data "beside", "folder").
    std::map<std::string, QCheckBox*> export_item_boxes;
    std::map<std::string, QCheckBox*> export_format_boxes;  // by export_format_keys' key
    QComboBox* export_form = nullptr;
    // Quick saves: a box for each format a save icon writes data in.
    std::map<std::string, QCheckBox*> quick_format_boxes;  // by export_format_keys' key
    QCheckBox* after_run_box = nullptr;
    QComboBox* after_run_place = nullptr;
    QPushButton* export_background_button = nullptr;
    QPushButton* export_text_button = nullptr;
    QCheckBox* use_cache = nullptr;
    QCheckBox* recognise_remuxes = nullptr;
    QCheckBox* check_frame_counts = nullptr;
    QLabel* log_label = nullptr;
    QPushButton* copy_log_button = nullptr;
    QPushButton* export_log_button = nullptr;
    QCheckBox* default_gpu = nullptr;
    MetricList* default_metrics = nullptr;  // Default metrics: what new videos start with ticked
    QComboBox* cvvdp_default = nullptr;
    QComboBox* film_grain_default = nullptr;
    QComboBox* vship_build_default = nullptr;  // none where stock Vship 5.1.2 is not bundled
    QCheckBox* gpu_together = nullptr;
    QComboBox* gpu_choice = nullptr;  // none on a PC with one GPU and none chosen
    QComboBox* vvc_decoder = nullptr;  // none where vvdec-vulkan is not bundled (a Mac)
    // Weak segments: the profile in use (its name the data, "" for Default),
    // its New, Rename and Delete, and the editor of its rules.
    QComboBox* weak_profile = nullptr;
    QPushButton* weak_profile_rename = nullptr;
    QPushButton* weak_profile_delete = nullptr;
    SegmentRulesEditor* weak_segments = nullptr;
    QComboBox* decoded_videos = nullptr;
    QComboBox* language = nullptr;
    // Settings > Appearance (settings_appearance.cpp, theme::apply_appearance):
    // the interface (0 Classic, 1 Modern), its cards; Modern's accent, the
    // system's ("") or "#rrggbb", its choices (each button's "hex"), the
    // code shown and typed, and Modern's card's picture of it; light or
    // dark (each button's "value": "", "light" or "dark").
    QButtonGroup* interface_buttons = nullptr;
    QButtonGroup* accent_buttons = nullptr;
    QWidget* accent_controls = nullptr;
    QLineEdit* accent_hex = nullptr;
    QWidget* modern_preview = nullptr;
    QString accent_value;
    QButtonGroup* theme_buttons = nullptr;
    QButtonGroup* light_look_buttons = nullptr;  // Settings.light_look's
    QButtonGroup* dark_look_buttons = nullptr;   // Settings.dark_look's
    QWidget* look_controls = nullptr;            // both, VML Modern's alone
    // How test videos' names are shown (video_names.h), and for "ends" how
    // many characters of each end.
    QComboBox* video_names = nullptr;
    QWidget* video_name_ends = nullptr;
    QSpinBox* video_name_start = nullptr;
    QSpinBox* video_name_end = nullptr;
    QCheckBox* remember_size = nullptr;
    QCheckBox* remember_videos = nullptr;
    QCheckBox* check_updates = nullptr;
    QCheckBox* confirm_close = nullptr;
    QCheckBox* keyboard_shortcuts = nullptr;
    QPushButton* shortcuts_list_button = nullptr;
    // Notifications (settings_notifications.cpp): when, the services and
    // their keys, how the keys are stored, and the message's templates with
    // their preview.
    QCheckBox* notify_finished = nullptr;
    QCheckBox* notify_with_failures = nullptr;
    QCheckBox* notify_cancelled = nullptr;
    QSpinBox* notify_min_minutes = nullptr;
    // The services: a list (ticked: on) and the chosen one's page.
    QListWidget* notify_services = nullptr;
    QStackedWidget* service_pages = nullptr;
    QCheckBox* pushover_enabled = nullptr;
    QLineEdit* pushover_user = nullptr;
    QLineEdit* pushover_token = nullptr;
    QComboBox* pushover_priority = nullptr;
    QCheckBox* discord_enabled = nullptr;
    QLineEdit* discord_webhook = nullptr;
    QLineEdit* discord_mention = nullptr;
    QComboBox* discord_mention_when = nullptr;  // data: "problems", "always"
    QCheckBox* discord_silent = nullptr;
    QCheckBox* gotify_enabled = nullptr;
    QLineEdit* gotify_url = nullptr;
    QLineEdit* gotify_token = nullptr;
    QSpinBox* gotify_priority = nullptr;
    QCheckBox* ntfy_enabled = nullptr;
    QLineEdit* ntfy_url = nullptr;
    QLineEdit* ntfy_topic = nullptr;
    QLineEdit* ntfy_token = nullptr;
    QComboBox* ntfy_priority = nullptr;
    QCheckBox* pushbullet_enabled = nullptr;
    QLineEdit* pushbullet_token = nullptr;
    QLineEdit* pushbullet_device = nullptr;
    QCheckBox* slack_enabled = nullptr;
    QLineEdit* slack_webhook = nullptr;
    QCheckBox* telegram_enabled = nullptr;
    QLineEdit* telegram_token = nullptr;
    QLineEdit* telegram_chat = nullptr;
    QCheckBox* telegram_silent = nullptr;
    QCheckBox* webhook_enabled = nullptr;
    QLineEdit* webhook_url = nullptr;
    QLineEdit* webhook_header = nullptr;
    QCheckBox* show_keys = nullptr;
    QComboBox* secret_storage = nullptr;  // data: true encrypted
    QLabel* secret_note = nullptr;
    QLineEdit* notify_title = nullptr;
    QPlainTextEdit* notify_message = nullptr;
    QPlainTextEdit* notify_video_line = nullptr;
    QPushButton* notify_presets = nullptr;
    QPushButton* insert_field = nullptr;
    QLabel* notify_preview = nullptr;

signals:
    // The defaults for newly added videos may have changed (any edit).
    void defaults_changed();
    // How many test videos Video Compare keeps decoding (any edit).
    void decoded_videos_changed(int count);
    // The FFmpeg folder changed and was saved: the window checks the tools.
    void ffmpeg_folder_changed();
    // Clear saved results has deleted them: the window's rows are to be
    // calculated again.
    void saved_results_cleared();
    // The weak segment profiles, or the one in use, were edited and saved:
    // the Worst Frames tab starts from them.
    void weak_segment_rules_changed();
    // How Saved Results shows a comparison's scores changed (a card for the
    // score used, or for each).
    void saved_scores_changed();
    // A setting saved shows only once the app starts again (the language,
    // VML Classic): `what` says so; the window offers to restart. Empty: the
    // setting went back, and no restart is needed any more.
    void restart_needed(const QString& what);
    // How test videos' names are shown changed (Video names).
    void video_names_changed();
    // Keyboard shortcuts was clicked: the window shows its list.
    void shortcuts_requested();
    // Use the keyboard shortcuts was ticked or unticked (saved).
    void keyboard_shortcuts_changed();

private:
    QGroupBox* build_ffmpeg_box();
    QGroupBox* build_storage_box();
    QGroupBox* build_saved_results_box();
    QGroupBox* build_saved_scores_box();
    QGroupBox* build_defaults_box();
    QGroupBox* build_weak_segments_box();
    // Settings > Storage > Exports include, and After each run.
    QWidget* build_export_choices();
    QWidget* build_quick_save_choices();
    QWidget* build_after_run_export();
    QGroupBox* build_gpu_box();
    QHBoxLayout* build_vvc_decoder_row();
    QGroupBox* build_compare_box();
    QGroupBox* build_window_box();
    QWidget* build_appearance_page();
    // The accent chosen (accent_value): its choice ticked, its code shown.
    void set_accent(const QString& hex);
    QWidget* build_notifications_page();
    QGroupBox* build_notify_when_box();
    QGroupBox* build_services_box();
    // A service's page: "Send to {service}" (`enabled`), its rows, where its
    // keys come from (`hint`), and Send test.
    QWidget* service_page(const QString& service, QCheckBox*& enabled, bool on,
                          const std::vector<std::pair<QString, QWidget*>>& rows, const QString& hint);
    // A field for a key: hidden as typed until Show keys, its stored value
    // shown (show_secret).
    // A key's field, for the secret setting `stored` (secret_settings), kept
    // with it in secret_edits_: stored from it as it changes.
    QLineEdit* secret_edit(std::string& stored);
    QGroupBox* build_keys_box();
    QGroupBox* build_message_box();
    // A secret's field: its stored value as it reads back, or empty with a
    // note where it cannot be read back here.
    void show_secret(QLineEdit* edit, const std::string& stored);
    // The notification controls into the settings (read_controls).
    void read_notification_controls();
    // The preview of the message, from a sample run.
    void show_notification_preview();
    // The keys' note: what their storage means.
    void show_secret_note();
    // Puts `field` ("{name}") where the cursor is in the template last edited.
    void insert_notification_field(const QString& field);
    // Sends the sample message to `service` alone, and says how it went.
    void send_notification_test(const QString& service);

    // The folder the saved results are in now.
    std::filesystem::path cache_folder() const;
    void show_cache_summary();
    void on_file_writes_idle();
    // Saves the tab, applies what is global, and tells the window.
    void on_edited();
    // Copies every control's value into the settings.
    void read_controls();
    // Applies at once what changed since `before` and is not the window's
    // alone: the theme.
    void apply_global_changes(const Settings& before);
    // Lets the user pick a folder for `edit`, and saves it.
    void pick_directory(const QString& title, QLineEdit* edit);
    // Exported graphs' colour buttons: each shows the colour exports use
    // now, and its value when one is chosen.
    void show_export_colors();
    // Lets the user choose a colour for `button` (or go back to the
    // palette's, with Reset color), and saves it.
    void choose_export_color(QPushButton* button, const QString& title);
    // Accent's Custom...: a colour picked in a dialog; none if it is closed.
    void choose_custom_accent();
    // Fills the profile list from Settings, the one in use picked, and
    // shows its rules; no edit.
    void fill_weak_profiles();
    void on_copy_log();
    void on_export_log();

    Settings& settings_;
    FileWrites& file_writes_;
    std::function<ToolsStatus()> tools_;
    const Dialogs& dialogs_;
    // The saved results being counted (refresh_status), and the folder; a
    // newer count replaces an older one before it is shown.
    QFutureWatcher<CacheSummary>* counting_ = nullptr;
    std::filesystem::path counted_folder_;
    // How many saved results deleting them removed, while it is under way.
    std::shared_ptr<ClearedResults> cleared_;
    // The saved comparisons read for Delete...'s question, and the folder.
    QFutureWatcher<std::vector<SavedComparison>>* deletion_listing_ = nullptr;
    std::filesystem::path deletion_folder_;
    bool dates_chosen_ = false;  // the dates edited: the count no longer sets them
    QString deleted_note_;  // what the last deletion deleted, said once under Delete...
    QVariantAnimation* delete_pulse_ = nullptr;  // that line fading from bold to muted
    void pulse_delete_preview();
    void say(const QString& text, ExportIndicator::Note kind = ExportIndicator::Note::done) const;
    std::string started_language_;  // the language the app started in: another shows after a restart
    QString restart_said_;           // the restart last asked for (restart_needed), or none
    // The range delete_scope names: the saved times from (none: no bound) up
    // to, not including, the second; everything with "all".
    std::pair<std::optional<std::chrono::system_clock::time_point>,
              std::optional<std::chrono::system_clock::time_point>>
    deletion_range() const;
    bool deleting_everything() const;
    // What Delete... would delete, from the count (cache_summary), under its row.
    void show_delete_preview();
    // The scope's own controls shown (the days, or the dates).
    void show_delete_scope();
    // The question, once the comparisons are read, and the deletion queued.
    void ask_and_delete(const std::vector<SavedComparison>& listed);
    bool filling_profiles_ = false;  // fill_weak_profiles: not an edit
    // Send test's sender; the templates' edits saved a moment after typing
    // stops; the template a field is inserted into.
    Notifier* test_notifier_ = nullptr;
    // Each key's field and its setting in settings_: hidden until Show keys.
    std::vector<std::pair<std::string*, QLineEdit*>> secret_edits_;
    QTimer* template_typing_ = nullptr;
    QWidget* last_template_ = nullptr;
};

}  // namespace vml::app
