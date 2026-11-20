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

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "dialogs.h"
#include "vml/ffmpeg_tools.h"
#include "vml/settings.h"

class QCheckBox;
class QComboBox;
class QGroupBox;
class QHBoxLayout;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QTabWidget;
class QTimer;
template <typename T>
class QFutureWatcher;

namespace vml {
struct CacheSummary;
struct ClearedResults;
}

namespace vml::app {

class FileWrites;
class Notifier;
class SegmentRulesEditor;

class SettingsPanel : public QWidget {
    Q_OBJECT

public:
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
    // Shows the film grain default the window has already saved (answered
    // "Don't ask again"), without treating it as an edit.
    void show_film_grain_default();
    // Deletes the saved results, after asking.
    void clear_saved_results();
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
    QLineEdit* export_edit = nullptr;
    // Exported graphs' colours: the palette, and a chosen background and
    // text colour, each button's in its "chosen" property ("#rrggbb", or
    // empty for the palette's own).
    QComboBox* export_graph_theme = nullptr;
    // Exports include: a box for each kind of export (by its key in
    // export_item_keys), and whether as a .zip file (data true) or loose
    // files. After each run: whether, and where (data "beside", "folder").
    std::map<std::string, QCheckBox*> export_item_boxes;
    QComboBox* export_form = nullptr;
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
    std::map<std::string, QCheckBox*> default_metrics;  // by metric key
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
    QComboBox* theme = nullptr;
    QCheckBox* remember_size = nullptr;
    QCheckBox* remember_videos = nullptr;
    QCheckBox* check_updates = nullptr;
    // Notifications (settings_notifications.cpp): when, the services and
    // their keys, how the keys are stored, and the message's templates with
    // their preview.
    QCheckBox* notify_finished = nullptr;
    QCheckBox* notify_with_failures = nullptr;
    QCheckBox* notify_cancelled = nullptr;
    QSpinBox* notify_min_minutes = nullptr;
    QCheckBox* pushover_enabled = nullptr;
    QLineEdit* pushover_user = nullptr;
    QLineEdit* pushover_token = nullptr;
    QComboBox* pushover_priority = nullptr;
    QPushButton* pushover_test = nullptr;
    QCheckBox* discord_enabled = nullptr;
    QLineEdit* discord_webhook = nullptr;
    QPushButton* discord_test = nullptr;
    QCheckBox* show_keys = nullptr;
    QComboBox* secret_storage = nullptr;  // data: true encrypted
    QLabel* secret_note = nullptr;
    QLineEdit* notify_title = nullptr;
    QPlainTextEdit* notify_message = nullptr;
    QPlainTextEdit* notify_video_line = nullptr;
    QPushButton* notify_presets = nullptr;
    QPushButton* insert_field = nullptr;
    QLabel* notify_preview = nullptr;
    QLabel* status = nullptr;

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
    // Metric Graphs' Weak segments starts from them.
    void weak_segment_rules_changed();

private:
    QGroupBox* build_ffmpeg_box();
    QGroupBox* build_storage_box();
    QGroupBox* build_defaults_box();
    QGroupBox* build_weak_segments_box();
    // Settings > Storage > Exports include, and After each run.
    QWidget* build_export_choices();
    QWidget* build_after_run_export();
    QGroupBox* build_gpu_box();
    QHBoxLayout* build_vvc_decoder_row();
    QGroupBox* build_compare_box();
    QGroupBox* build_window_box();
    QWidget* build_notifications_page();
    QGroupBox* build_notify_when_box();
    QGroupBox* build_pushover_box();
    QGroupBox* build_discord_box();
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
    // How many saved results clearing them removed and kept, while it is
    // under way.
    std::shared_ptr<ClearedResults> cleared_;
    bool filling_profiles_ = false;  // fill_weak_profiles: not an edit
    // Send test's sender; the templates' edits saved a moment after typing
    // stops; the template a field is inserted into.
    Notifier* test_notifier_ = nullptr;
    QTimer* template_typing_ = nullptr;
    QWidget* last_template_ = nullptr;
};

}  // namespace vml::app
