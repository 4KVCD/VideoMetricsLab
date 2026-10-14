// The Bitrate Viewer tab (the Python app's vmaf_app/ui/bitrate_panel.py and
// bitrate_worker.py): a list of videos of its own, each read packet by
// packet with ffprobe, and graphs of their bitrate per frame, per second or
// per group of pictures. It works apart from the Videos tab: a finished run
// adds its videos here, and files can be added here alone.
#pragma once

#include <QColor>
#include <QMetaType>
#include <QString>
#include <QThread>
#include <QWidget>

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dialogs.h"
#include "vml/bitrate.h"
#include "vml/cancel.h"
#include "vml/video.h"

class QCheckBox;
class QDragEnterEvent;
class QDropEvent;
class QLabel;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QTableWidget;
class QTableWidgetItem;

namespace vml::app {

class ChartWidget;
class VideoColors;

// How far a file in the Bitrate tab has got.
enum class BitrateState { ready, queued, reading, packets_read, complete, failed, stopped };

// One file in the Bitrate tab: its path, colour, details, packets, whether
// its line is shown, how far it has got, and its graphs worked out so far.
struct BitrateEntry {
    std::filesystem::path path;
    QColor color;
    int series = 0;  // its line's key on the chart
    std::optional<VideoInfo> info;
    std::shared_ptr<const BitrateData> data;
    bool enabled = true;
    BitrateState state = BitrateState::ready;
    long long packets_read = 0;  // while it is read
    QString error;               // why it could not be, once failed
    // By graph ("frame", "second", "gop") and whether the start is moved to 0.
    std::map<std::pair<std::string, bool>, std::shared_ptr<const BitratePlot>> plots;
};

// One file the worker has read: its details and its packets.
struct BitrateAnalyzed {
    QString path;
    VideoInfo info;
    std::shared_ptr<const BitrateData> data;
};

// Reads each file's packets in turn, off the window's thread: its details
// first with ffprobe when not known.
class BitrateWorker : public QThread {
    Q_OBJECT

public:
    using Job = std::pair<std::filesystem::path, std::optional<VideoInfo>>;
    BitrateWorker(std::string ffprobe, std::vector<Job> jobs, QObject* parent = nullptr);

    // Stops, ending ffprobe.
    void cancel() { cancel_.cancel(); }

signals:
    void file_started(const QString& path);
    void progress(const QString& path, long long done, long long total);
    void analyzed(const vml::app::BitrateAnalyzed& analyzed);
    void failed(const QString& path, const QString& error);

protected:
    void run() override;

private:
    std::string ffprobe_;
    std::vector<Job> jobs_;
    CancelToken cancel_;
};

class BitratePanel : public QWidget {
    Q_OBJECT

public:
    // `ffprobe`: the program reading the files, asked for each scan.
    // `colors`: the session's video colours, shared with Metric Graphs.
    BitratePanel(std::function<std::string()> ffprobe, VideoColors* colors, QWidget* parent = nullptr);
    ~BitratePanel() override;

    // Adds files to the list, each once. With `analyze`, the new ones and
    // those without results are read; `infos` are details already read.
    void add_files(const std::vector<std::filesystem::path>& paths,
                   const std::map<std::filesystem::path, VideoInfo>& infos = {}, bool analyze = false);
    // Adds videos read already, and reads their packets (a finished run's).
    void add_and_analyze(const std::vector<VideoInfo>& infos);
    // The scan running now, if any.
    std::vector<QThread*> live_workers() const;
    // Stops the analysis, keeping finished results.
    void cancel();

    // Its questions and messages; the tests answer them.
    Dialogs dialogs;

    QPushButton* add_button = nullptr;
    QPushButton* remove_button = nullptr;
    QPushButton* analyze_button = nullptr;
    QTableWidget* table = nullptr;
    QRadioButton* frame_radio = nullptr;
    QRadioButton* second_radio = nullptr;
    QRadioButton* gop_radio = nullptr;
    QCheckBox* adjust_start_checkbox = nullptr;
    QPushButton* reset_zoom_button = nullptr;
    QPushButton* export_button = nullptr;
    ChartWidget* chart = nullptr;
    QLabel* hover_label = nullptr;
    QLabel* status_label = nullptr;
    QProgressBar* progress = nullptr;

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    friend class BitratePanelTest;  // the panel's tests (tests/app)

    BitrateEntry* entry_for(const QString& path_text);
    int row_for_key(const QString& key) const;
    void on_add_files();
    void on_remove_selected();
    void on_calculate_or_stop();
    void on_analyze_checked();
    void queue_keys(const std::vector<QString>& keys);
    void start_pending();
    void on_file_started(const QString& path);
    void on_progress(const QString& path, long long done, long long total);
    void on_analyzed(const BitrateAnalyzed& analyzed);
    void on_failed(const QString& path, const QString& error);
    void on_worker_finished(BitrateWorker* worker);
    void rebuild_table();
    void refresh_table_values();
    QTableWidgetItem* cell(int row, int column);
    void set_table_row(int row, const BitrateEntry& entry);
    void on_table_item_changed(QTableWidgetItem* item);
    void choose_video_color(int row);
    void on_video_color_changed(const QString& path_key, const QColor& color);
    void update_buttons();
    std::string view_name() const;
    std::shared_ptr<const BitratePlot> entry_plot(BitrateEntry& entry) const;
    void refresh_plot();
    void on_hovered(double time);
    void on_hover_left();
    void on_export_png();

    std::function<std::string()> ffprobe_;
    VideoColors* colors_;
    // The files, in the order added, each with its key (its full path, in
    // one case).
    std::vector<std::pair<QString, BitrateEntry>> entries_;
    std::vector<QString> pending_;  // queued, in order
    std::set<QString> active_keys_;  // being read by the worker
    BitrateWorker* worker_ = nullptr;
    bool stopping_ = false;
    bool populating_ = false;
    int next_series_ = 0;
    // The lines drawn, in the table's order: each file's name and graph.
    std::vector<std::pair<QString, std::shared_ptr<const BitratePlot>>> current_plots_;
};

}  // namespace vml::app

Q_DECLARE_METATYPE(vml::app::BitrateAnalyzed)
