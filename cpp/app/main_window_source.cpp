// The window's reference video (the Python app's MainWindow, its source
// part): choosing it, reading it with ffprobe, and what a new one changes
// in the rows already listed.
#include "main_window.h"

#include <QDropEvent>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QLineEdit>
#include <QMessageBox>
#include <QMouseEvent>

#include "formatting.h"
#include "frame_compare_panel.h"
#include "graph_panel.h"
#include "i18n.h"
#ifdef VML_NATIVE_PLAYBACK
#include "native_pool.h"
#endif
#include "vml/app_log.h"
#include "vml/ffmpeg_tools.h"
#include "vml/film_grain_scoring.h"
#include "vml/python_text.h"
#include "vml/platform/system.h"
#include "vml/time_format.h"
#include "widgets.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.main_window");

}  // namespace

// ----------------------------------------------------------------- the source

void MainWindow::on_browse_source() {
    const QString path =
        QFileDialog::getOpenFileName(this, app::tr("Select reference video"), video_dialog_dir("reference"));
    if (path.isEmpty()) {
        return;
    }
    remember_video_dir("reference", path_of(path));
    choose_source(path_of(path));
}

void MainWindow::choose_source(const std::filesystem::path& path) {
    // Read as the new reference (picked, dropped or typed in); the box shows
    // it once it is read (apply_source_info).
    source_reading_ = path;
    source_info_label_->setText(app::tr("Reading {name}...", {{"name", name_of(path)}}));
    show_reading(app::tr("Reading reference video..."));
    start_source_probe(path);
}

QString MainWindow::shown_source_path() const { return source_info_ ? text_of(source_info_->path) : QString(); }

void MainWindow::on_source_path_entered() {
    // A path typed or pasted into the reference box, read when Enter is
    // pressed or the box is left. Quotes, as Explorer's "Copy as path" puts
    // them, are taken off. Not as the window closes: the focus leaves the box
    // as it hides, and a reading started then would outlive the window.
    if (run_active_ || closing_ || source_edit_->isReadOnly()) {
        return;
    }
    QString text = source_edit_->text().trimmed();
    while (text.startsWith('"')) {
        text.remove(0, 1);
    }
    while (text.endsWith('"')) {
        text.chop(1);
    }
    text = text.trimmed();
    const QString shown = shown_source_path();
    // Compared as paths (c:/v.mkv is the same file as C:/V.MKV).
    if (text.isEmpty() || (!shown.isEmpty() && same_path(path_of(text), path_of(shown)))) {
        source_edit_->setText(shown);
        return;
    }
    if (source_reading_ && same_path(path_of(text), *source_reading_)) {
        return;  // already being read: editingFinished comes for Enter and again for leaving the box
    }
    const std::filesystem::path path = path_of(text);
    if (!QFileInfo(text).isFile()) {
        // Applied on Enter only. Leaving the box, say for another program to
        // copy the rest of the path, keeps what was typed.
        if (source_edit_->hasFocus()) {
            source_edit_->setText(shown_source_path());
            QMessageBox::warning(this, app::tr("No such video"),
                                 app::tr("There is no file at {path}.", {{"path", text}}));
        }
        return;
    }
    choose_source(path);
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    // A file dropped on the reference box replaces the reference; the line
    // edit's own drop would only paste the file's URL in as text.
    const QEvent::Type type = event->type();
#ifdef VML_NATIVE_PLAYBACK
    if (type == QEvent::MouseMove && watched == tabs_->tabBar() && !NativePool::loaded()) {
        const auto* move = static_cast<QMouseEvent*>(event);
        if (tabs_->tabBar()->tabAt(move->position().toPoint()) == tab_frame_compare) {
            frame_compare_panel_->prepare_playback();
        }
    }
#endif
    if (watched == source_edit_ && (type == QEvent::DragEnter || type == QEvent::DragMove || type == QEvent::Drop)) {
        auto* drop = static_cast<QDropEvent*>(event);
        // Found when the drag enters: a drag move comes with every mouse
        // move, and each look at the files is a trip to the disk.
        if (type != QEvent::DragMove) {
            reference_drop_ = FillColumnTable::dropped_files(drop->mimeData());
        }
        if (run_active_ || reference_drop_.size() != 1) {
            event->ignore();
            return true;
        }
        drop->setDropAction(Qt::CopyAction);
        drop->accept();
        if (type == QEvent::Drop) {
            remember_video_dir("reference", path_of(reference_drop_.front()));
            choose_source(path_of(reference_drop_.front()));
        }
        return true;
    }
    return QMainWindow::eventFilter(watched, event);
}

QString MainWindow::video_dialog_dir(const QString& kind) const {
    // The folder such a video last came from, while it still exists; else
    // the other's; else the dialog's default.
    const std::string& own = kind == "reference" ? settings_.last_reference_dir : settings_.last_test_dir;
    const std::string& other = kind == "reference" ? settings_.last_test_dir : settings_.last_reference_dir;
    for (const std::string* folder : {&own, &other}) {
        if (!folder->empty() && QFileInfo(QString::fromStdString(*folder)).isDir()) {
            return QString::fromStdString(*folder);
        }
    }
    return {};
}

void MainWindow::remember_video_dir(const QString& kind, const std::filesystem::path& path) {
    std::string& setting = kind == "reference" ? settings_.last_reference_dir : settings_.last_test_dir;
    const std::string folder = platform::path_text(path.parent_path());
    if (folder != setting) {
        setting = folder;
        if (const std::optional<std::string> error = settings_.save()) {
            logger.warning(*error);
        }
    }
}

void MainWindow::start_source_probe(const std::filesystem::path& path) {
    // Read without freezing the window. A newer choice replaces one still
    // being read.
    if (source_probe_worker_ != nullptr && source_probe_worker_->isRunning()) {
        source_probe_worker_->cancel();
    }
    const int generation = ++source_probe_generation_;
    auto* worker =
        new ProbeWorker({text_of(path)}, find_tool("ffprobe", settings_), find_tool("ffmpeg", settings_), this);
    source_probe_worker_ = worker;
    probe_workers_.push_back(worker);
    connect(worker, &ProbeWorker::probed, this,
            [this, generation](const Probed& probed) { on_source_probed(generation, probed); });
    connect(worker, &ProbeWorker::finished_all, this,
            [this, generation, worker] { on_source_probe_finished(generation, worker); });
    worker->start();
}

void MainWindow::on_source_probed(int generation, const Probed& probed) {
    if (generation != source_probe_generation_) {
        return;
    }
    source_reading_.reset();
    if (!probed.info) {
        source_edit_->setText(shown_source_path());  // a path typed in that would not read
        show_source_label();
        QMessageBox::critical(this, app::tr("Could not read video"), tr_message(probed.error));
        return;
    }
    apply_source_info(path_of(probed.path), *probed.info);
}

void MainWindow::forget_recalculations_unless(const std::filesystem::path& reference) {
    // What a Recalculate forgot (RowData::forgotten_metrics) was the old
    // pair's: kept only when `reference` is the same file again.
    if (source_info_ && same_file(source_info_->path, reference)) {
        return;
    }
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        row_data->forgotten_metrics.clear();
    }
}

void MainWindow::apply_source_info(const std::filesystem::path& path, const VideoInfo& info) {
    forget_recalculations_unless(path);
    source_info_ = info;
    // A reference just read (chosen again too) has no film grain choice:
    // Settings' default, or it is asked about at Calculate metrics.
    const std::optional<bool> with = film_grain_default_choice(settings_.film_grain_default);
    if (with && film_grain_choosable(*source_info_) && !source_info_->with_film_grain) {
        source_info_->with_film_grain = *with;
    }
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (row_data->options.resample_test && row_data->video_info) {
            take_film_grain(*row_data->video_info, *source_info_);  // its copy of the reference
        }
    }
    source_edit_->setText(text_of(path));
    show_source_label();
    check_rows_against_new_source(path);
}

void MainWindow::show_source_label() {
    if (!source_info_) {
        source_info_label_->setText(app::tr("No reference selected."));
        source_info_label_->setToolTip(QString());
        return;
    }
    source_info_label_->setText(media_info_string(*source_info_) + ", " + bitrate_string(*source_info_) + "  (" +
                                QString::fromStdString(format_hms(source_info_->duration, 1)) + ")");
    source_info_label_->setToolTip(film_grain_note(*source_info_));
}

void MainWindow::check_rows_against_new_source(const std::filesystem::path& path) {
    // The rows, after `path` became the reference: chosen, or taken from a
    // saved run. A resolution test is made from the reference, while a test
    // video is only compared with it, so a resolution test cannot outlive a
    // change of reference.
    const std::vector<QString> dropped = remove_rows_owned_by_the_previous_source();
    // Before the saved scores are looked up: they are found by the display.
    follow_reference_cvvdp();
    // Scores belong to a pair of videos: a new reference takes away every
    // score measured against another, which would otherwise stand in the
    // table as the new pair's, and keep a run from calculating it, and the
    // state of every row without scores (how its run against the old one
    // ended, failed or cancelled). Those measured against the very reference
    // chosen stay, with its film grain as it is scored now: choosing it again
    // confirms them.
    std::vector<int> stale;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (!row_data.completed_run || !same_file(row_data.completed_run->result.source, path) ||
            film_grain_route_identity(row_data.completed_run->result.source_info) !=
                film_grain_route_identity(*source_info_)) {
            stale.push_back(row);
        }
    }
    invalidate_completed_results(stale);
    // Every row is checked against the new reference: its Scaling cell
    // depends on it, and its saved scores with the new reference are looked
    // up off the window's thread (a long video's are several MB to read).
    std::vector<std::filesystem::path> paths;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        set_row_metrics(row);
        paths.push_back(rows_[static_cast<std::size_t>(row)]->path);
    }
    start_cache_lookup(paths);
    if (!dropped.empty()) {
        QStringList names;
        for (const QString& name : dropped) {
            names.append("  " + name);
        }
        dialogs_.inform(app::tr("Resolution tests removed"),
                        ntr("{count} resolution test belonged to the previous source and has been removed:\n\n",
                            "{count} resolution tests belonged to the previous source and have been removed:\n\n",
                            static_cast<long long>(dropped.size())) +
                            names.join("\n") + app::tr("\n\nAdd them again to test the new source."));
    }
}

std::vector<QString> MainWindow::remove_rows_owned_by_the_previous_source() {
    // Removes the resolution tests and returns their names, bottom to top.
    // Such a row has no test video of its own: it scales the reference down
    // and back up, and its name, media details, widths and saved scores'
    // identity all come from the reference chosen when it was added. Left in
    // place after the reference changes, the row would describe one video
    // while its job ran on another. They are removed rather than moved to the
    // new reference: the widths offered depend on the reference's width, two
    // moved rows could become the same test, and quietly changing what a row
    // means is worse than saying it is gone.
    std::vector<QString> removed;
    std::vector<std::string> lines;
    for (int row = static_cast<int>(rows_.size()) - 1; row >= 0; --row) {
        if (!rows_[static_cast<std::size_t>(row)]->options.resample_test) {
            continue;
        }
        if (const std::shared_ptr<CompletedRun>& run = rows_[static_cast<std::size_t>(row)]->completed_run) {
            lines.push_back(run->graph_identity);
        }
        removed.push_back(name_of(rows_[static_cast<std::size_t>(row)]->path));
        distorted_table_->removeRow(row);
        rows_.erase(rows_.begin() + row);
    }
    graph_panel_->remove_by_identities(lines);
    if (!removed.empty()) {
        on_table_selection_changed();
        sync_frame_compare();
    }
    return removed;
}

void MainWindow::invalidate_completed_results(const std::vector<int>& rows) {
    // The rows' scores go, with their state and its detail: the tooltip
    // would otherwise say "Not calculated" over the old result's "240 scored
    // frames; metrics: ...", or over a failure's reason.
    // Their lines go from the graph too, all at once.
    std::vector<std::string> lines;
    std::vector<std::filesystem::path> paths;
    for (const int row : rows) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        row_data.analysis_status.reset();
        row_data.status_detail.clear();
        if (row_data.completed_run) {
            lines.push_back(row_data.completed_run->graph_identity);
            paths.push_back(row_data.path);
        }
        row_data.completed_run.reset();
        set_row_metrics(row);
    }
    if (!lines.empty()) {
        graph_panel_->remove_by_identities(lines, paths);
        sync_frame_compare();
    }
}

void MainWindow::show_reading(const QString& text) {
    reading_text_ = text;
    show_status(text);
}

void MainWindow::show_ready() {
    // Only in place of the "Reading..." message: anything else on the status
    // line is a message for the user, which a reader finishing a moment
    // later must not overwrite.
    const QString text = status_label_->text();
    if (text.isEmpty() || text == reading_text_) {
        show_status(app::tr("Ready."));
    }
}

void MainWindow::retire(ProbeWorker* worker) {
    // Its last signal says it is done, and its thread ends a moment later: a
    // thread must have ended before it is deleted.
    worker->wait();
    std::erase(probe_workers_, worker);
    worker->deleteLater();
}

void MainWindow::on_source_probe_finished(int generation, ProbeWorker* worker) {
    retire(worker);
    if (generation == source_probe_generation_) {
        source_probe_worker_ = nullptr;
        source_reading_.reset();  // also when it stopped without a result
        if (!run_active_) {
            show_ready();
        }
        on_table_selection_changed();
    }
}

}  // namespace vml::app
