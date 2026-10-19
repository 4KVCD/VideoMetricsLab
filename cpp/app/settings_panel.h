// The Settings tab (the Python app's vmaf_app/ui/settings_panel.py): the
// app-wide settings, saved as soon as they are edited.
//
// The panel applies what is global itself: the theme and the GPU backends.
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
#include <vector>

#include "dialogs.h"
#include "vml/ffmpeg_tools.h"
#include "vml/settings.h"
#include "vml/video.h"

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPushButton;
template <typename T>
class QFutureWatcher;

namespace vml {
struct CacheSummary;
}

namespace vml::app {

class FileWrites;

class SettingsPanel : public QWidget {
    Q_OBJECT

public:
    // `settings` are the window's own, which this panel edits and saves;
    // `file_writes` the window's queue of writes, which clearing the saved
    // results goes through; `tools` gives the window's latest check of
    // FFmpeg (each check starts two programs, so the panel does not check
    // again itself); `dialogs` the window's, which outlive the panel (its
    // tests answer them).
    SettingsPanel(Settings& settings, FileWrites& file_writes, const std::vector<GpuVendor>& gpu_vendors,
                  std::function<ToolsStatus()> tools, const Dialogs& dialogs, QWidget* parent = nullptr);
    ~SettingsPanel() override;

    // Shows the tools and the saved results as they are now. Called when the
    // tab is shown, when its folders change and when the saved results are
    // cleared. The saved results are counted on a thread of their own: a
    // thousand take 0.3 s, which would hold the window.
    void refresh_status();
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
    // Deletes the saved results, after asking.
    void clear_saved_results();

    // The controls, for the window's tests.
    QLineEdit* ffmpeg_edit = nullptr;
    QLabel* ffmpeg_status = nullptr;
    QLineEdit* cache_edit = nullptr;
    QLabel* cache_summary = nullptr;
    QLineEdit* export_edit = nullptr;
    QCheckBox* use_cache = nullptr;
    QLabel* log_label = nullptr;
    QPushButton* copy_log_button = nullptr;
    QPushButton* export_log_button = nullptr;
    QCheckBox* default_gpu = nullptr;
    std::map<std::string, QCheckBox*> default_metrics;  // by metric key
    QComboBox* cvvdp_default = nullptr;
    QComboBox* gpu_backend = nullptr;
    QCheckBox* gpu_together = nullptr;
    QComboBox* decoded_videos = nullptr;
    QComboBox* language = nullptr;
    QComboBox* theme = nullptr;
    QCheckBox* remember_size = nullptr;
    QCheckBox* remember_videos = nullptr;
    QCheckBox* check_updates = nullptr;
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

private:
    QGroupBox* build_ffmpeg_box();
    QGroupBox* build_storage_box();
    QGroupBox* build_defaults_box();
    QGroupBox* build_gpu_box(const std::vector<GpuVendor>& gpu_vendors);
    QGroupBox* build_compare_box();
    QGroupBox* build_window_box();

    // The folder the saved results are in now.
    std::filesystem::path cache_folder() const;
    void show_cache_summary();
    void on_file_writes_idle();
    // Saves the tab, applies what is global, and tells the window.
    void on_edited();
    // Copies every control's value into the settings.
    void read_controls();
    // Applies at once what changed since `before` and is not the window's
    // alone: the theme and the GPU backends.
    void apply_global_changes(const Settings& before);
    // Lets the user pick a folder for `edit`, and saves it.
    void pick_directory(const QString& title, QLineEdit* edit);
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
    // How many saved results clearing them removed, while it is under way.
    std::shared_ptr<int> cleared_;
};

}  // namespace vml::app
