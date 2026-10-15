#include "bitrate_panel.h"

#include <QAbstractItemView>
#include <QButtonGroup>
#include <QCheckBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMimeData>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>
#include <span>

#include "chart.h"
#include "formatting.h"
#include "i18n.h"
#include "series_colors.h"
#include "vml/ffprobe.h"
#include "vml/platform/system.h"
#include "vml/time_format.h"

namespace vml::app {

namespace {

enum Column { col_use, col_file, col_media, col_duration, col_frames, col_average, col_minimum, col_maximum, col_status };

const char* const hover_hint =
    N_("Hover over the plot to inspect values. Scroll to zoom, drag to pan, and double-click to reset.");
const char* const calculate_tooltip = N_("Calculate bitrate for videos with the Use checkbox checked.");

QString text_of(const std::string& text) { return QString::fromStdString(text); }

// A file's key in the list: its full path, in one case.
QString path_key(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(path, error);
    return text_of(platform::path_text(error ? path : resolved)).toCaseFolded();
}

QString file_name(const std::filesystem::path& path) { return text_of(platform::path_text(path.filename())); }

// A bitrate as shown: in kb/s, or Mb/s from 1000 kb/s.
QString rate_text(double kbps) {
    if (kbps >= 10'000) {
        return app::tr("{rate:.2f} Mb/s", {{"rate", kbps / 1000}});
    }
    if (kbps >= 1000) {
        return app::tr("{rate:.3f} Mb/s", {{"rate", kbps / 1000}});
    }
    return app::tr("{rate:.0f} kb/s", {{"rate", kbps}});
}

QString status_text(const BitrateEntry& entry) {
    switch (entry.state) {
        case BitrateState::ready: return app::tr("Ready to analyze");
        case BitrateState::queued: return app::tr("Queued");
        case BitrateState::reading: return app::tr("Reading video packets\xE2\x80\xA6");
        case BitrateState::packets_read:
            return app::ntr("{count:,} packet", "{count:,} packets", entry.packets_read);
        case BitrateState::complete: return app::tr("Complete");
        case BitrateState::failed: return app::tr("Failed");
        case BitrateState::stopped: return app::tr("Stopped");
    }
    return {};
}

QPixmap swatch_of(const QColor& color) {
    QPixmap swatch(12, 12);
    swatch.fill(color);
    return swatch;
}

}  // namespace

// ------------------------------------------------------------------ the worker

BitrateWorker::BitrateWorker(std::string ffprobe, std::vector<Job> jobs, QObject* parent)
    : QThread(parent), ffprobe_(std::move(ffprobe)), jobs_(std::move(jobs)) {}

void BitrateWorker::run() {
    for (const auto& [path, known] : jobs_) {
        if (cancel_.cancelled()) {
            return;
        }
        const QString name = text_of(platform::path_text(path));
        emit file_started(name);
        try {
            const VideoInfo info = known ? *known : probe_video(path, ffprobe_, cancel_);
            if (cancel_.cancelled()) {
                return;
            }
            // Here, off the window's thread: the summary walks every packet.
            auto data = std::make_shared<const BitrateData>(analyze_video_bitrate(
                ffprobe_, info, cancel_, [this, &name](long long done, long long total) {
                    emit progress(name, done, total);
                }));
            if (!cancel_.cancelled()) {
                emit analyzed({name, info, std::move(data)});
            }
        } catch (const Cancelled&) {
            return;
        } catch (const std::exception& error) {
            if (!cancel_.cancelled()) {
                emit failed(name, QString::fromUtf8(error.what()));
            }
        }
    }
}

// ------------------------------------------------------------------- the panel

BitratePanel::BitratePanel(std::function<std::string()> ffprobe, VideoColors* colors, QWidget* parent)
    : QWidget(parent), ffprobe_(std::move(ffprobe)), colors_(colors) {
    qRegisterMetaType<BitrateAnalyzed>();
    dialogs = standard_dialogs(this);
    connect(colors_, &VideoColors::changed, this,
            [this](const QString& key, const QColor& color) { on_video_color_changed(key, color); });
    setAcceptDrops(true);

    auto* root = new QVBoxLayout(this);
    auto* file_controls = new QHBoxLayout;
    add_button = new QPushButton(app::tr("Add files\xE2\x80\xA6"));
    connect(add_button, &QPushButton::clicked, this, [this] { on_add_files(); });
    remove_button = new QPushButton(app::tr("Remove selected"));
    connect(remove_button, &QPushButton::clicked, this, [this] { on_remove_selected(); });
    analyze_button = new QPushButton(app::tr("Calculate bitrate"));
    analyze_button->setToolTip(app::tr(calculate_tooltip));
    connect(analyze_button, &QPushButton::clicked, this, [this] { on_calculate_or_stop(); });
    file_controls->addWidget(add_button);
    file_controls->addWidget(remove_button);
    file_controls->addWidget(analyze_button);
    file_controls->addStretch(1);
    root->addLayout(file_controls);

    table = new QTableWidget(0, 9);
    auto* swatches = new ColorSwatchDelegate(table);
    connect(swatches, &ColorSwatchDelegate::clicked, this, [this](int row) { choose_video_color(row); });
    table->setItemDelegateForColumn(col_use, swatches);
    table->setHorizontalHeaderLabels({app::tr("Use"), app::tr("Video file"), app::tr("Media info"),
                                      app::tr("Duration"), app::tr("Frames"), app::tr("Average"), app::tr("Minimum"),
                                      app::tr("Maximum"), app::tr("Status")});
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::ExtendedSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setAlternatingRowColors(true);
    table->verticalHeader()->setVisible(false);
    QHeaderView* header = table->horizontalHeader();
    header->setSectionResizeMode(col_file, QHeaderView::Stretch);
    for (int column = 0; column < table->columnCount(); ++column) {
        if (column != col_file) {
            header->setSectionResizeMode(column, QHeaderView::ResizeToContents);
        }
    }
    table->setMaximumHeight(230);
    connect(table, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* item) { on_table_item_changed(item); });
    root->addWidget(table);

    auto* plot_controls = new QHBoxLayout;
    plot_controls->addWidget(new QLabel(app::tr("Plot view:")));
    auto* views = new QButtonGroup(this);
    frame_radio = new QRadioButton(app::tr("Frame based"));
    second_radio = new QRadioButton(app::tr("Second based"));
    gop_radio = new QRadioButton(app::tr("GOP based"));
    second_radio->setChecked(true);
    int id = 0;
    for (QRadioButton* button : {frame_radio, second_radio, gop_radio}) {
        views->addButton(button, id++);
        plot_controls->addWidget(button);
    }
    connect(views, &QButtonGroup::idClicked, this, [this](int /*id*/) { refresh_plot(); });
    adjust_start_checkbox = new QCheckBox(app::tr("Adjust stream start time to zero"));
    adjust_start_checkbox->setChecked(true);
    connect(adjust_start_checkbox, &QCheckBox::toggled, this, [this](bool /*checked*/) { refresh_plot(); });
    plot_controls->addSpacing(18);
    plot_controls->addWidget(adjust_start_checkbox);
    plot_controls->addStretch(1);
    reset_zoom_button = new QPushButton(app::tr("Reset zoom"));
    export_button = new QPushButton(app::tr("Export PNG\xE2\x80\xA6"));
    connect(export_button, &QPushButton::clicked, this, [this] { on_export_png(); });
    plot_controls->addWidget(reset_zoom_button);
    plot_controls->addWidget(export_button);
    root->addLayout(plot_controls);

    chart = new ChartWidget(app::tr("Video bitrate (kb/s)"));
    connect(reset_zoom_button, &QPushButton::clicked, chart, &ChartWidget::reset_view);
    connect(chart, &ChartWidget::hovered, this, [this](double time, double /*value*/) { on_hovered(time); });
    connect(chart, &ChartWidget::left, this, [this] { on_hover_left(); });
    root->addWidget(chart, 1);

    hover_label = new QLabel(app::tr(hover_hint));
    hover_label->setTextFormat(Qt::PlainText);
    root->addWidget(hover_label);

    auto* status_row = new QHBoxLayout;
    status_label = new QLabel(
        app::tr("Frame view shows encoded video-frame size; second and GOP views show video-only bitrate."));
    progress = new QProgressBar;
    progress->setRange(0, 100);
    progress->setValue(0);
    progress->setMaximumWidth(220);
    // Shown only while files are read: left at 100% it looked as if
    // something were still going on. Its room is kept, so that the plot
    // does not move when it comes and goes.
    QSizePolicy keep_room = progress->sizePolicy();
    keep_room.setRetainSizeWhenHidden(true);
    progress->setSizePolicy(keep_room);
    progress->setVisible(false);
    status_row->addWidget(status_label, 1);
    status_row->addWidget(progress);
    root->addLayout(status_row);
    update_buttons();
}

BitratePanel::~BitratePanel() {
    if (worker_ != nullptr) {
        worker_->cancel();
        worker_->wait();
    }
}

// ------------------------------------------------------------- for the window

void BitratePanel::add_files(const std::vector<std::filesystem::path>& paths,
                             const std::map<std::filesystem::path, VideoInfo>& infos, bool analyze) {
    std::vector<QString> added;
    for (const std::filesystem::path& path : paths) {
        const QString key = path_key(path);
        const auto info = infos.find(path);
        const auto found = std::find_if(entries_.begin(), entries_.end(),
                                        [&key](const auto& item) { return item.first == key; });
        if (found != entries_.end()) {
            if (info != infos.end()) {
                found->second.info = info->second;
            }
            if (analyze && !found->second.data) {
                added.push_back(key);
            }
            continue;
        }
        BitrateEntry entry;
        entry.path = path;
        entry.color = colors_->for_path(path);
        entry.series = next_series_++;
        if (info != infos.end()) {
            entry.info = info->second;
        }
        entries_.emplace_back(key, std::move(entry));
        added.push_back(key);
    }
    rebuild_table();
    refresh_plot();
    if (analyze) {
        queue_keys(added);
    }
}

void BitratePanel::add_and_analyze(const std::vector<VideoInfo>& infos) {
    // Each path once, in the order first given, with its latest details.
    std::vector<std::filesystem::path> paths;
    std::map<std::filesystem::path, VideoInfo> by_path;
    for (const VideoInfo& info : infos) {
        if (!by_path.contains(info.path)) {
            paths.push_back(info.path);
        }
        by_path.insert_or_assign(info.path, info);
    }
    add_files(paths, by_path, true);
}

std::vector<QThread*> BitratePanel::live_workers() const {
    if (worker_ != nullptr && worker_->isRunning()) {
        return {worker_};
    }
    return {};
}

void BitratePanel::cancel() {
    for (const QString& key : pending_) {
        const auto found = std::find_if(entries_.begin(), entries_.end(),
                                        [&key](const auto& item) { return item.first == key; });
        if (found != entries_.end()) {
            found->second.state = BitrateState::stopped;
        }
    }
    pending_.clear();
    if (worker_ != nullptr) {
        stopping_ = true;
        status_label->setText(app::tr("Stopping bitrate analysis\xE2\x80\xA6"));
        worker_->cancel();
    }
    refresh_table_values();
    update_buttons();
}

// ------------------------------------------------------------------ the files

void BitratePanel::dragEnterEvent(QDragEnterEvent* event) {
    const QMimeData* mime = event->mimeData();
    if (mime->hasUrls() &&
        std::any_of(mime->urls().begin(), mime->urls().end(), [](const QUrl& url) { return url.isLocalFile(); })) {
        event->acceptProposedAction();
    }
}

void BitratePanel::dropEvent(QDropEvent* event) {
    std::vector<std::filesystem::path> paths;
    for (const QUrl& url : event->mimeData()->urls()) {
        if (url.isLocalFile()) {
            paths.push_back(platform::path_from_text(url.toLocalFile().toStdString()));
        }
    }
    add_files(paths);
    event->acceptProposedAction();
}

void BitratePanel::on_add_files() {
    const QStringList chosen = QFileDialog::getOpenFileNames(this, app::tr("Add videos for bitrate analysis"));
    std::vector<std::filesystem::path> paths;
    for (const QString& path : chosen) {
        paths.push_back(platform::path_from_text(path.toStdString()));
    }
    if (!paths.empty()) {
        add_files(paths);
    }
}

void BitratePanel::on_remove_selected() {
    std::set<int> rows;
    for (const QModelIndex& index : table->selectionModel()->selectedIndexes()) {
        rows.insert(index.row());
    }
    for (auto row = rows.rbegin(); row != rows.rend(); ++row) {
        if (*row >= 0 && *row < static_cast<int>(entries_.size())) {
            const QString key = entries_[static_cast<std::size_t>(*row)].first;
            entries_.erase(entries_.begin() + *row);
            std::erase(pending_, key);
        }
    }
    rebuild_table();
    refresh_plot();
}

void BitratePanel::on_calculate_or_stop() {
    if (worker_ != nullptr) {
        cancel();
    } else {
        on_analyze_checked();
    }
}

void BitratePanel::on_analyze_checked() {
    // The ticked files, asking first whether to read again those read.
    if (worker_ != nullptr) {
        return;
    }
    std::vector<QString> keys;
    long long completed = 0;
    for (const auto& [key, entry] : entries_) {
        if (entry.enabled) {
            keys.push_back(key);
            completed += entry.data ? 1 : 0;
        }
    }
    if (completed > 0 &&
        !dialogs.ask_no_default(app::tr("Recalculate bitrate?"),
                     app::ntr("{count} checked video already has bitrate results.\n\nRecalculate it? Choose No to "
                              "keep its results and calculate only files without results.",
                              "{count} checked videos already have bitrate results.\n\nRecalculate those files? "
                              "Choose No to keep their results and calculate only files without results.",
                              completed))) {
        std::erase_if(keys, [this](const QString& key) {
            return std::any_of(entries_.begin(), entries_.end(),
                               [&key](const auto& item) { return item.first == key && item.second.data; });
        });
    }
    if (!keys.empty()) {
        queue_keys(keys);
    }
}

void BitratePanel::queue_keys(const std::vector<QString>& keys) {
    // Queues files, and starts reading if nothing is being read.
    if (stopping_) {
        return;
    }
    for (const QString& key : keys) {
        const auto found = std::find_if(entries_.begin(), entries_.end(),
                                        [&key](const auto& item) { return item.first == key; });
        if (found != entries_.end() && !active_keys_.contains(key)) {
            found->second.state = BitrateState::queued;
            if (std::find(pending_.begin(), pending_.end(), key) == pending_.end()) {
                pending_.push_back(key);
            }
        }
    }
    refresh_table_values();
    start_pending();
}

void BitratePanel::start_pending() {
    // One scan at a time, of every file queued so far.
    if (worker_ != nullptr) {
        return;
    }
    std::vector<BitrateWorker::Job> jobs;
    for (const QString& key : pending_) {
        const auto found = std::find_if(entries_.begin(), entries_.end(),
                                        [&key](const auto& item) { return item.first == key; });
        if (found != entries_.end()) {
            jobs.emplace_back(found->second.path, found->second.info);
        }
    }
    pending_.clear();
    if (jobs.empty()) {
        update_buttons();
        return;
    }
    active_keys_.clear();
    for (const auto& [path, info] : jobs) {
        active_keys_.insert(path_key(path));
    }
    auto* worker = new BitrateWorker(ffprobe_(), std::move(jobs), this);
    worker_ = worker;
    connect(worker, &BitrateWorker::file_started, this, [this](const QString& path) { on_file_started(path); });
    connect(worker, &BitrateWorker::progress, this,
            [this](const QString& path, long long done, long long total) { on_progress(path, done, total); });
    connect(worker, &BitrateWorker::analyzed, this, [this](const BitrateAnalyzed& analyzed) { on_analyzed(analyzed); });
    connect(worker, &BitrateWorker::failed, this,
            [this](const QString& path, const QString& error) { on_failed(path, error); });
    connect(worker, &QThread::finished, this, [this, worker] { on_worker_finished(worker); });
    update_buttons();
    progress->setValue(0);
    progress->setVisible(true);
    worker->start();
}

BitrateEntry* BitratePanel::entry_for(const QString& path_text) {
    const QString key = path_key(platform::path_from_text(path_text.toStdString()));
    const auto found =
        std::find_if(entries_.begin(), entries_.end(), [&key](const auto& item) { return item.first == key; });
    return found == entries_.end() ? nullptr : &found->second;
}

int BitratePanel::row_for_key(const QString& key) const {
    for (std::size_t row = 0; row < entries_.size(); ++row) {
        if (entries_[row].first == key) {
            return static_cast<int>(row);
        }
    }
    return -1;
}

void BitratePanel::on_file_started(const QString& path) {
    if (stopping_) {
        return;
    }
    if (BitrateEntry* entry = entry_for(path)) {
        entry->state = BitrateState::reading;
        status_label->setText(
            app::tr("Analyzing {name}\xE2\x80\xA6", {{"name", file_name(platform::path_from_text(path.toStdString()))}}));
        progress->setValue(0);
        refresh_table_values();
    }
}

void BitratePanel::on_progress(const QString& path, long long done, long long total) {
    if (stopping_) {
        return;
    }
    progress->setValue(static_cast<int>(
        std::min(99.0, std::nearbyint(static_cast<double>(done) / static_cast<double>(std::max(1LL, total)) * 100))));
    if (BitrateEntry* entry = entry_for(path)) {
        entry->state = BitrateState::packets_read;
        entry->packets_read = done;
        // That cell alone: every row again for each 500 packets would cost
        // the window's thread for nothing.
        const int row = row_for_key(path_key(entry->path));
        if (row >= 0 && table->item(row, col_status) != nullptr) {
            table->item(row, col_status)->setText(status_text(*entry));
        }
    }
}

void BitratePanel::on_analyzed(const BitrateAnalyzed& analyzed) {
    BitrateEntry* entry = entry_for(analyzed.path);
    if (entry == nullptr) {
        return;  // removed while it was read
    }
    entry->info = analyzed.info;
    entry->data = analyzed.data;
    entry->plots.clear();
    entry->state = BitrateState::complete;
    progress->setValue(100);
    refresh_table_values();
    refresh_plot();
}

void BitratePanel::on_failed(const QString& path, const QString& error) {
    BitrateEntry* entry = entry_for(path);
    if (entry == nullptr) {
        return;
    }
    entry->state = BitrateState::failed;
    entry->error = error;
    refresh_table_values();
    status_label->setText(app::tr("Could not analyze {name}: {error}",
                                  {{"name", file_name(entry->path)}, {"error", error}}));
}

void BitratePanel::on_worker_finished(BitrateWorker* worker) {
    // Cleans up after a scan, and starts the next if files were queued.
    if (worker != worker_) {
        worker->deleteLater();
        return;
    }
    const bool stopped = stopping_;
    if (stopped) {
        for (auto& [key, entry] : entries_) {
            if (active_keys_.contains(key) && entry.state != BitrateState::complete &&
                entry.state != BitrateState::failed) {
                entry.state = BitrateState::stopped;
            }
        }
        refresh_table_values();
    }
    stopping_ = false;
    worker_ = nullptr;
    active_keys_.clear();
    worker->deleteLater();
    if (!pending_.empty()) {
        start_pending();
    } else {
        status_label->setText(stopped ? app::tr("Bitrate analysis stopped. Completed results are kept.")
                                      : app::tr("Bitrate analysis complete. Values contain the first video "
                                                "stream only."));
        progress->setVisible(false);
        update_buttons();
    }
}

// ------------------------------------------------------------------ the table

void BitratePanel::rebuild_table() {
    populating_ = true;
    table->setRowCount(static_cast<int>(entries_.size()));
    for (std::size_t row = 0; row < entries_.size(); ++row) {
        set_table_row(static_cast<int>(row), entries_[row].second);
    }
    populating_ = false;
    update_buttons();
}

void BitratePanel::refresh_table_values() {
    populating_ = true;
    for (std::size_t row = 0; row < entries_.size(); ++row) {
        set_table_row(static_cast<int>(row), entries_[row].second);
    }
    populating_ = false;
}

QTableWidgetItem* BitratePanel::cell(int row, int column) {
    // Put into the table the first time only: Qt refuses setItem for an item
    // the table already holds, with a warning each time.
    QTableWidgetItem* item = table->item(row, column);
    if (item == nullptr) {
        item = new QTableWidgetItem;
        table->setItem(row, column, item);
    }
    return item;
}

void BitratePanel::set_table_row(int row, const BitrateEntry& entry) {
    QTableWidgetItem* use = cell(row, col_use);
    use->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
    use->setCheckState(entry.enabled ? Qt::Checked : Qt::Unchecked);
    use->setData(Qt::DecorationRole, swatch_of(entry.color));
    use->setToolTip(app::tr("Click the color square to change this video's color in both graphs."));
    QStringList values{file_name(entry.path),
                       entry.info ? media_info_string(*entry.info) : QString(),
                       entry.data ? text_of(format_hms(entry.data->duration(), 2)) : QString(),
                       entry.data ? python_format("{count:,}", {{"count", entry.data->frame_count()}}) : QString(),
                       QString(),
                       QString(),
                       QString(),
                       status_text(entry)};
    if (entry.data) {
        values[4] = rate_text(entry.data->summary.average_kbps);
        values[5] = rate_text(entry.data->summary.minimum_kbps);
        values[6] = rate_text(entry.data->summary.maximum_kbps);
    }
    for (int column = col_file; column <= col_status; ++column) {
        cell(row, column)->setText(values[column - col_file]);
    }
    cell(row, col_file)->setToolTip(text_of(platform::path_text(entry.path)));
    // Why the file failed, on its own row: kept with the file, so that a row
    // that moves up as another is removed shows its own.
    cell(row, col_status)->setToolTip(entry.state == BitrateState::failed ? entry.error : QString());
}

void BitratePanel::on_table_item_changed(QTableWidgetItem* item) {
    // A Use box ticked: the file's line shown or hidden.
    if (populating_ || item->column() != col_use) {
        return;
    }
    if (item->row() >= 0 && item->row() < static_cast<int>(entries_.size())) {
        entries_[static_cast<std::size_t>(item->row())].second.enabled = item->checkState() == Qt::Checked;
        refresh_plot();
    }
}

void BitratePanel::choose_video_color(int row) {
    if (row >= 0 && row < static_cast<int>(entries_.size())) {
        colors_->choose(entries_[static_cast<std::size_t>(row)].second.path, this);
    }
}

void BitratePanel::on_video_color_changed(const QString& key, const QColor& color) {
    // The swatches and lines repainted; packets, graphs and zoom kept.
    populating_ = true;
    for (std::size_t row = 0; row < entries_.size(); ++row) {
        BitrateEntry& entry = entries_[row].second;
        if (VideoColors::path_key(entry.path) != key) {
            continue;
        }
        entry.color = color;
        cell(static_cast<int>(row), col_use)->setData(Qt::DecorationRole, swatch_of(color));
        chart->set_series_color(entry.series, color);
    }
    populating_ = false;
}

void BitratePanel::update_buttons() {
    const bool running = worker_ != nullptr;
    analyze_button->setText(stopping_ ? app::tr("Stopping\xE2\x80\xA6")
                            : running ? app::tr("Stop")
                                      : app::tr("Calculate bitrate"));
    analyze_button->setToolTip(running ? app::tr("Stop bitrate analysis and keep completed results.")
                                       : app::tr(calculate_tooltip));
    analyze_button->setEnabled((running || !entries_.empty()) && !stopping_);
    remove_button->setEnabled(!entries_.empty());
    export_button->setEnabled(chart->has_data());
}

// ------------------------------------------------------------------ the graph

std::string BitratePanel::view_name() const {
    if (frame_radio->isChecked()) {
        return "frame";
    }
    if (gop_radio->isChecked()) {
        return "gop";
    }
    return "second";
}

std::shared_ptr<const BitratePlot> BitratePanel::entry_plot(BitrateEntry& entry) const {
    // A file's line for the graph chosen, worked out once.
    if (!entry.data) {
        return nullptr;
    }
    const std::pair<std::string, bool> key{view_name(), adjust_start_checkbox->isChecked()};
    auto& plot = entry.plots[key];
    if (!plot) {
        plot = std::make_shared<const BitratePlot>(key.first == "frame"   ? frame_plot(*entry.data, key.second)
                                                   : key.first == "gop" ? gop_plot(*entry.data, key.second)
                                                                        : second_plot(*entry.data, key.second));
    }
    return plot;
}

void BitratePanel::refresh_plot() {
    // Every ticked file's line, for the graph chosen.
    chart->clear();
    current_plots_.clear();
    std::vector<std::pair<const BitrateEntry*, std::shared_ptr<const BitratePlot>>> plots;
    for (auto& [key, entry] : entries_) {
        if (!entry.enabled) {
            continue;
        }
        if (std::shared_ptr<const BitratePlot> plot = entry_plot(entry)) {
            plots.emplace_back(&entry, std::move(plot));
        }
    }
    // In megabits once a line reaches 1000 kb/s, where the table also
    // switches to Mb/s: a 4K film's busiest second put 250000 on the axis.
    const bool mega = std::ranges::any_of(plots, [](const auto& item) {
        const std::vector<float>& values = item.second->values;
        return !values.empty() && *std::ranges::max_element(values) >= 1000;
    });
    value_scale_ = mega ? 1000 : 1;
    const std::string view = view_name();
    if (view == "frame") {
        chart->set_y_axis_label(mega ? app::tr("Frame size (Mbit)") : app::tr("Frame size (kbit)"));
        value_unit_ = mega ? app::tr("Mbit") : app::tr("kbit");
    } else {
        if (view == "gop") {
            chart->set_y_axis_label(mega ? app::tr("GOP bitrate (Mb/s)") : app::tr("GOP bitrate (kb/s)"));
        } else {
            chart->set_y_axis_label(mega ? app::tr("Video bitrate (Mb/s)") : app::tr("Video bitrate (kb/s)"));
        }
        value_unit_ = mega ? app::tr("Mb/s") : app::tr("kb/s");
    }
    for (const auto& [entry, plot] : plots) {
        if (mega) {
            // The line in megabits, kept alive by the graph with the plot
            // whose times it shares.
            struct Scaled {
                std::shared_ptr<const BitratePlot> plot;
                std::vector<float> values;
            };
            auto scaled = std::make_shared<Scaled>(Scaled{plot, plot->values});
            for (float& value : scaled->values) {
                value /= 1000;
            }
            chart->set_series(entry->series,
                              {std::span(plot->times), std::span(scaled->values), scaled, entry->color, true});
        } else {
            chart->set_series(entry->series, {std::span(plot->times), std::span(plot->values), plot, entry->color, true});
        }
        current_plots_.emplace_back(file_name(entry->path), plot);
    }
    chart->reset_view();
    on_hover_left();
    update_buttons();
}

void BitratePanel::on_hovered(double time) {
    // Each line's value at the time under the pointer.
    chart->set_cursor_time(time);
    QStringList lines{app::tr("Time: {time}", {{"time", text_of(format_hms(time, 3))}})};
    for (const auto& [name, plot] : current_plots_) {
        const std::vector<double>& times = plot->times;
        if (times.empty()) {
            continue;
        }
        // The point nearest the time, the earlier one when they tie.
        auto index = static_cast<std::size_t>(std::lower_bound(times.begin(), times.end(), time) - times.begin());
        index = std::min(times.size() - 1, index);
        if (index > 0 && std::abs(times[index - 1] - time) <= std::abs(times[index] - time)) {
            --index;
        }
        lines.append(python_format("{name}: {value:,.2f} {unit}",
                                   {{"name", name},
                                    {"value", static_cast<double>(plot->values[index]) / value_scale_},
                                    {"unit", value_unit_}}));
    }
    hover_label->setText(lines.join('\n'));
}

void BitratePanel::on_hover_left() {
    chart->set_cursor_time(std::nullopt);
    hover_label->setText(app::tr(hover_hint));
}

void BitratePanel::on_export_png() {
    if (!chart->has_data()) {
        dialogs.inform(app::tr("No bitrate data"), app::tr("Analyze at least one video first."));
        return;
    }
    const QString path =
        dialogs.save_file(app::tr("Export bitrate plot"), QStringLiteral("bitrate_plot.png"), app::tr("PNG image (*.png)"));
    if (!path.isEmpty() && !chart->render_to_pixmap().save(path)) {
        dialogs.alert(app::tr("Export failed"), app::tr("Could not write {path}", {{"path", path}}));
    }
}

}  // namespace vml::app
