// The Bitrate Viewer tab (the Python app's vmaf_app/ui/bitrate_panel.py and
// bitrate_worker.py): a list of videos of its own, each read packet by
// packet with ffprobe, and graphs of their bitrate per frame, per second or
// per group of pictures. It works apart from the Videos tab: a finished run
// adds its videos here, and files can be added here alone.
#pragma once

#include <QColor>
#include <QMetaType>
#include <QPixmap>
#include <QString>
#include <QThread>
#include <QWidget>

#include <cstdint>
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
#include "export_image.h"
#include "theme.h"
#include "vml/bitrate.h"
#include "vml/cancel.h"
#include "vml/video.h"
#include "widgets.h"

class QCheckBox;
class QDragEnterEvent;
class QDropEvent;
class QLabel;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QTableWidget;
class QTableWidgetItem;
class QVBoxLayout;

namespace vml::app {

class ChartWidget;
class FrameControls;
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
    std::shared_ptr<const std::vector<std::uint8_t>> frame_types;  // read_frame_types: a type per frame, or none
    std::string read_as;  // the file's identity (file_identity) when `data` was read: its name may be used again
    bool enabled = true;
    bool reference = false;  // a run's reference (add_and_analyze): its exported plot is named after its test video
    BitrateState state = BitrateState::ready;
    long long packets_read = 0;  // while it is read
    QString error;               // why it could not be, once failed
    // By graph ("frame", "second", "gop") and whether the start is moved to 0.
    std::map<std::pair<std::string, bool>, std::shared_ptr<const BitratePlot>> plots;
    // The frame range's figures (frames_summary), worked out when the range
    // or the packets change; nothing without a range or any of its frames.
    std::optional<BitrateSummary> range_summary;
};

// One file the worker has read: its details and its packets.
struct BitrateAnalyzed {
    QString path;
    VideoInfo info;
    std::shared_ptr<const BitrateData> data;
    std::shared_ptr<const std::vector<std::uint8_t>> frame_types;
    std::string read_as;  // BitrateEntry::read_as
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
    // A raw stream that stores no frame rate: its details, and nothing read.
    void frame_rate_wanted(const vml::app::BitrateAnalyzed& wanted);

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
    // Adds videos read already, and reads their packets (a finished run's:
    // the first is its reference).
    void add_and_analyze(const std::vector<VideoInfo>& infos);
    // The scan running now, if any.
    std::vector<QThread*> live_workers() const;
    // Stops the analysis, keeping finished results.
    void cancel();
    // What Export PNG saves: the graph with its title (the graph chosen, the
    // frame range, the frame types' colours while marked) and a table of
    // the lines drawn, keyed by their colours, with the table's figures.
    QPixmap render_export_image();
    // render_export_image with the chart drawn at `size`; its own size is put back.
    QPixmap render_export_image(QSize size);

    // How far `path` has got: complete once read; nothing when not listed.
    std::optional<BitrateState> state_of(const std::filesystem::path& path) const;
    // Lists `paths` as `from` has read them (their packets, details and
    // colours; the first as the reference), reading nothing: the panel an
    // export draws apart from the tab (graph_bundle.h). Those `from` has not
    // read are left out.
    void adopt_read(const BitratePanel& from, const std::vector<std::filesystem::path>& paths);

    // Its questions and messages; the tests answer them.
    Dialogs dialogs;
    // The colours Export PNG draws in (Settings' Exported graphs), asked at
    // each export; the light chart colours when not set.
    std::function<theme::ChartColours()> export_colours;

    QPushButton* add_button = nullptr;
    QPushButton* remove_button = nullptr;
    QPushButton* analyze_button = nullptr;
    QTableWidget* table = nullptr;
    QRadioButton* frame_radio = nullptr;
    QRadioButton* second_radio = nullptr;
    QRadioButton* gop_radio = nullptr;
    QCheckBox* adjust_start_checkbox = nullptr;
    QCheckBox* difference_box = nullptr;
    QCheckBox* frame_types_box = nullptr;
    QLabel* frame_types_legend = nullptr;
    QPushButton* reset_zoom_button = nullptr;
    QPushButton* export_button = nullptr;
    ChartWidget* chart = nullptr;
    ReadoutLabel* hover_label = nullptr;
    // Go to frame and the frame range, shared with Metric Graphs, and what
    // the table's figures are of while a range is chosen.
    FrameControls* frame_controls = nullptr;
    QLabel* range_label = nullptr;
    QLabel* status_label = nullptr;
    QProgressBar* progress = nullptr;

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    // The constructor's parts: the panel's rows, top to bottom, added to
    // `root`.
    void add_file_controls(QVBoxLayout* root);
    void add_table(QVBoxLayout* root);
    void add_plot_controls(QVBoxLayout* root);
    void add_chart(QVBoxLayout* root);
    void add_frame_controls(QVBoxLayout* root);
    void add_status_row(QVBoxLayout* root);

    friend class BitratePanelTest;  // the panel's tests (tests/app)

    BitrateEntry* entry_for(const QString& path_text);
    BitrateEntry* entry_of_key(const QString& key);
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
    void on_frame_rate_wanted(const BitrateAnalyzed& wanted);
    void ask_frame_rates();
    void on_worker_finished(BitrateWorker* worker);
    void rebuild_table();
    void refresh_table_values();
    QTableWidgetItem* cell(int row, int column);
    void set_table_row(int row, const BitrateEntry& entry);
    // The figures the table shows for `entry`: the frame range's while one
    // is chosen, else the whole video's; none before it is read.
    std::optional<BitrateSummary> shown_summary(const BitrateEntry& entry) const;
    // The exported image's title and table (render_export_image).
    QString export_title() const;
    ExportTable export_table() const;
    void on_table_item_changed(QTableWidgetItem* item);
    void choose_video_color(int row);
    void on_video_color_changed(const QString& path_key, const QColor& color);
    void update_buttons();
    std::string view_name() const;
    std::shared_ptr<const BitratePlot> entry_plot(BitrateEntry& entry) const;
    void refresh_plot();  // the lines drawn again, the zoom kept
    void refresh_difference();  // the Difference band and box, for the lines drawn
    void show_frame_types_box();  // usable in the frame graph; its legend while ticked
    // One drawn line's point in the readout: its index on the line, and the
    // frame it shows.
    struct ReadoutPoint {
        std::size_t index = 0;
        std::size_t frame = 0;
    };
    // The readout at `time`: a line for each drawn line's point (in
    // current_plots_' order), or for one without, that it lacks
    // `asked_frame` (Go to frame), and the difference.
    void write_readout(double time, const std::vector<std::optional<ReadoutPoint>>& points,
                       std::optional<int> asked_frame);
    void on_hovered(double time);
    void on_hover_left();
    // Go to frame: each drawn line at its own frame `frame`.
    void on_frame_requested(int frame);
    // The frame the first drawn line shows at `time`, as the readout counts
    // it; nothing with no line drawn.
    std::optional<int> frame_at(double time) const;
    void on_range_dragged(double start, double now, bool finished);
    // The range changed (FrameControls::range_changed): shaded at once; its
    // figures in the table once `finished`.
    void on_range_changed(bool finished);
    void update_range_summary(BitrateEntry& entry) const;
    // The range shaded on the graph, as the first drawn line with its first
    // frame places it: from that frame's start to its last frame's end.
    void refresh_highlight();
    void refresh_range_label();
    void on_export_png();

    std::function<std::string()> ffprobe_;
    VideoColors* colors_;
    // The files, in the order added, each with its key (its full path, in
    // one case).
    std::vector<std::pair<QString, BitrateEntry>> entries_;
    std::vector<QString> pending_;  // queued, in order
    std::set<QString> active_keys_;  // being read by the worker
    std::vector<QString> frame_rates_wanted_;  // to be asked for once the worker ends
    BitrateWorker* worker_ = nullptr;
    bool stopping_ = false;
    bool populating_ = false;
    int next_series_ = 0;
    // The lines drawn, in the table's order: each file's name and graph.
    // The lines drawn; frame_types only for a frame graph's.
    struct DrawnPlot {
        QString name;
        std::shared_ptr<const BitratePlot> plot;
        std::shared_ptr<const std::vector<std::uint8_t>> frame_types;
        std::shared_ptr<const BitrateData> data;  // its frames' times, in the file's time
        double start = 0;                         // the file's time at the graph's 0
    };
    std::vector<DrawnPlot> current_plots_;
    // The graph's values are divided by this and shown in this unit (see
    // refresh_plot).
    double value_scale_ = 1;
    QString value_unit_ = QStringLiteral("kb/s");
    // The Difference band drawn, in the lines' unit, and its label; none
    // while the box is off or the graph does not show exactly two lines.
    std::shared_ptr<const BitratePlot> difference_;
    QString difference_label_;
};

}  // namespace vml::app

Q_DECLARE_METATYPE(vml::app::BitrateAnalyzed)
