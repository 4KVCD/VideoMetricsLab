// The window's runs (the Python app's MainWindow, from _on_run_clicked to
// _on_all_finished): the ticked videos' metrics calculated on a RunWorker,
// the status line saying how much is done and for how long, a line for each
// video in progress, and each video's scores in its row as they arrive.
#include <QCheckBox>
#include <QFutureWatcher>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTableWidgetItem>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cmath>
#include <utility>
#include <variant>

#include "bitrate_panel.h"
#include "file_writes.h"
#include "graph_panel.h"
#include "i18n.h"
#include "main_window.h"
#include "options_panel.h"
#include "run_worker.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/app_log.h"
#include "vml/crop_detect.h"
#include "vml/ffmpeg_run.h"
#include "vml/geometry.h"
#include "vml/gpu_vmaf.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/perceptual.h"
#include "vml/run_line.h"
#include "vml/time_format.h"
#include "vml/vmaf_v1_gpu.h"
#include "vml/vship.h"

// The window's text is app::tr's, as in main_window.cpp.
namespace vml::app {

namespace {

const app_log::Logger logger("vml.main_window");

// A task's rate is its frames over the last ten seconds of the run's own
// clock, once two have passed (the Python app's _RATE_WINDOW_SECONDS and
// _RATE_MIN_SECONDS).
constexpr double rate_window_seconds = 10.0;
constexpr double rate_min_seconds = 2.0;

// The CPU's seconds per megapixel of a compared frame pair for SSIMULACRA2
// and Butteraugli, measured with the bundled libjxl 0.12.0 tools on a
// 3840x1608 (6.17 MP) pair: 1.1 s and 2.0 s. A rough guide, as CPUs differ.
double cpu_seconds_per_megapixel(const std::string& key) { return (key == "ssimulacra2" ? 1.1 : 2.0) / 6.17; }

// "about 3 days", "about 5 hours", "about 40 minutes", rounded as Python's
// round() rounds (halves to even, as nearbyint does).
QString rough_duration(double seconds) {
    if (seconds >= 2 * 86400) {
        return ntr("about {count} day", "about {count} days", static_cast<long long>(std::nearbyint(seconds / 86400)));
    }
    if (seconds >= 2 * 3600) {
        return ntr("about {count} hour", "about {count} hours", static_cast<long long>(std::nearbyint(seconds / 3600)));
    }
    return ntr("about {count} minute", "about {count} minutes",
               std::max(1LL, static_cast<long long>(std::nearbyint(seconds / 60))));
}

std::function<bool(bool)>& keep_awake() {
    static std::function<bool(bool)> keep = [](bool awake) { return platform::keep_system_awake(awake); };
    return keep;
}

// The run line's words in the window's language (run_line.h's Wording).
const Wording& window_wording() {
    static const Wording wording{
        [](std::string_view english, const WordingValues& values) {
            std::vector<NamedValue> named;
            for (const auto& [name, value] : values) {
                named.push_back({name.c_str(), std::visit([](const auto& held) { return FormatValue(held); }, value)});
            }
            return app::tr(qt_text(english), named).toStdString();
        },
        [](std::string_view message) { return tr_message(qt_text(message)).toStdString(); },
    };
    return wording;
}

bool contains(const std::vector<std::string>& keys, const std::string& key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

}  // namespace

void set_keep_awake(std::function<bool(bool awake)> keep) { keep_awake() = std::move(keep); }

std::vector<int> MainWindow::checked_rows() const {
    std::vector<int> rows;
    for (int row = 0; row < distorted_table_->rowCount(); ++row) {
        if (distorted_table_->item(row, col_check)->checkState() == Qt::Checked) {
            rows.push_back(row);
        }
    }
    return rows;
}

void MainWindow::on_run_clicked() {
    if (worker_ != nullptr && worker_->isRunning()) {
        return;
    }
    const std::vector<int> checked = checked_rows();
    if (!ready_to_run(checked)) {
        return;
    }
    // A ticked row that already has a score (say it was ticked before more
    // files were added) is not calculated again: rows with every requested
    // metric are skipped.
    std::vector<int> to_run;
    long long already_scored = 0;
    for (const int row : checked) {
        if (has_requested_results(*rows_[static_cast<std::size_t>(row)])) {
            ++already_scored;
        } else {
            to_run.push_back(row);
        }
    }
    if (to_run.empty()) {
        show_already_scored(checked, already_scored);
        return;
    }
    std::vector<std::shared_ptr<RowData>> job_rows;
    std::optional<std::vector<VmafJob>> jobs = plan_jobs(to_run, job_rows);
    if (!jobs || jobs->empty() || !confirm_long_cpu_perceptual(job_rows)) {
        return;
    }
    begin_run(checked, std::move(job_rows), already_scored);
    start_run_worker(std::move(*jobs));
}

// Whether a run of the `checked` rows can start: the reference read, a row
// checked, a metric chosen for each, and the GPU checks the run needs done
// (else they are waited for, and the run starts by itself once they are).
// Says why not, where it cannot.
bool MainWindow::ready_to_run(const std::vector<int>& checked) {
    if (source_probe_worker_ != nullptr && source_probe_worker_->isRunning()) {
        dialogs_.inform(app::tr("Still reading reference"), app::tr("Wait for the reference video to finish loading."));
        return false;
    }
    if (!source_info_) {
        QMessageBox::warning(this, app::tr("No reference"), app::tr("Please select a reference video."));
        return false;
    }
    if (checked.empty()) {
        QMessageBox::warning(this, app::tr("No test videos"),
                             app::tr("Check at least one test video to calculate metrics."));
        return false;
    }
    if (std::any_of(checked.begin(), checked.end(), [this](int row) {
            return requested_metrics(*rows_[static_cast<std::size_t>(row)]).empty();
        })) {
        QMessageBox::warning(this, app::tr("No metrics selected"),
                             app::tr("Select at least one metric for every checked video, or uncheck videos you do "
                                     "not want to calculate."));
        return false;
    }
    return !wait_for_gpu_checks(checked);
}

// Nothing to run: each of the `checked` rows (`count` of them) has every
// metric asked for. Their scores are what was asked for: on the graph.
void MainWindow::show_already_scored(const std::vector<int>& checked, long long count) {
    show_status(ntr("The {count} checked video already has all requested metrics -- nothing to run.",
                    "All {count} checked videos already have all requested metrics -- nothing to run.", count));
    std::vector<std::shared_ptr<CompletedRun>> done;
    for (const int row : checked) {
        if (const std::shared_ptr<CompletedRun>& run = rows_[static_cast<std::size_t>(row)]->completed_run) {
            done.push_back(run);
        }
    }
    if (!done.empty()) {
        open_or_update_graph(done);
    }
}

// The jobs for `rows`, and each job's row in `job_rows`. Nothing when the
// run is not to start: a video is still being read, or the user does not
// want to go on without the rows that cannot be compared.
std::optional<std::vector<VmafJob>> MainWindow::plan_jobs(const std::vector<int>& rows,
                                                          std::vector<std::shared_ptr<RowData>>& job_rows) {
    std::vector<VmafJob> jobs;
    std::vector<std::pair<std::shared_ptr<RowData>, QString>> skipped;  // cannot be compared, and why
    for (const int row : rows) {
        const std::shared_ptr<RowData>& row_data = rows_[static_cast<std::size_t>(row)];
        if (!row_data->video_info) {
            if (std::any_of(probe_workers_.begin(), probe_workers_.end(),
                            [](ProbeWorker* worker) { return worker->isRunning(); })) {
                dialogs_.inform(app::tr("Still reading videos"),
                                app::tr("Wait for every checked test video to finish loading before running."));
                return std::nullopt;
            }
            skipped.emplace_back(row_data, N_("It could not be read. Remove it or add the file again to retry."));
            continue;
        }
        std::variant<VmafJob, QString> job = job_for(*row_data);
        if (const QString* why = std::get_if<QString>(&job)) {
            skipped.emplace_back(row_data, *why);
            continue;
        }
        jobs.push_back(std::move(std::get<VmafJob>(job)));
        job_rows.push_back(row_data);
    }
    if (!skipped.empty() && !confirm_skipping(skipped, !jobs.empty())) {
        return std::nullopt;
    }
    return jobs;
}

// The job that calculates what `row_data` asks for and has not got (its
// video read); or why the two videos cannot be compared as asked.
std::variant<VmafJob, QString> MainWindow::job_for(const RowData& row_data) const {
    const VideoInfo& info = *row_data.video_info;
    const std::vector<std::string> metric_keys = requested_metrics(row_data);
    const PerceptualBackends backends(row_data.metric_backends.begin(), row_data.metric_backends.end());
    VmafOptions options = row_data.options;
    MetricResultSet kept = reusable_results(row_data);
    std::vector<MetricTask> tasks;
    try {
        if (!options.resample_test) {
            validate_video_pair(*source_info_, info, options);
        }
        // Sized by what the run will compare, not by the test video's own
        // resolution, because one side is scaled to the other before
        // libvmaf sees it. The run works this out again after auto crop,
        // which cannot be known here.
        const auto [width, height] = options.resample_test
                                         ? resample_analysis_dimensions(*source_info_, std::nullopt)
                                         : analysis_dimensions(*source_info_, info, options);
        options.model = options.compute_vmaf ? resolve_model(options, width, height) : "";
        tasks = plan_tasks(*source_info_, info, options, metric_keys, backends, kept);
    } catch (const std::exception& error) {
        return QString::fromStdString(error.what());
    }
    return VmafJob{*source_info_, info, std::move(options), row_data.path, metric_keys, std::move(tasks),
                   backends, row_data.cvvdp,
                   row_data.completed_run ? std::optional(row_data.completed_run->result) : std::nullopt,
                   std::move(kept), platform::path_text(row_data.path.stem())};
}

// The run's state, made anew for the `checked` rows, of which `job_rows`
// are run and `already_scored` are skipped; the window set for a run.
void MainWindow::begin_run(const std::vector<int>& checked, std::vector<std::shared_ptr<RowData>> job_rows,
                           long long already_scored) {
    job_rows_ = std::move(job_rows);
    checked_rows_for_run_.clear();
    for (const int row : checked) {
        checked_rows_for_run_.push_back(rows_[static_cast<std::size_t>(row)]);
    }
    job_options_.clear();
    job_cvvdp_.clear();
    for (const std::shared_ptr<RowData>& row_data : job_rows_) {
        set_row_status(*row_index_of(row_data.get()), RowState::queued);
        job_options_.push_back(row_data->options);
        job_cvvdp_.push_back(row_data->cvvdp);
    }
    job_task_progress_.clear();
    rate_samples_.clear();
    running_jobs_.clear();
    finished_jobs_.clear();
    job_line_slot_.clear();
    job_line_text_.clear();
    job_line_shape_.clear();
    job_lines_due_.clear();
    for (ElidedLabel* line : job_progress_labels_) {
        line->setVisible(false);
        line->clear_text();
    }
    run_failed_count_ = 0;
    run_partial_count_ = 0;
    run_was_cancelled_ = false;
    run_skipped_ = static_cast<int>(already_scored);
    run_started_at_ = clock_();
    paused_total_ = 0;
    paused_since_.reset();
    run_hold_ = RunHold::none;
    run_tick_timer_->start();
    set_run_ui_active(true);
    update_run_status();  // the counts, the skipped included, before any video starts
    pause_button_->setChecked(false);
    pause_button_->setText(app::tr("Pause"));
}

// Runs `jobs` on a RunWorker of their own, its signals wired to the window.
void MainWindow::start_run_worker(std::vector<VmafJob> jobs) {
    use_crop_store();  // black bars found are kept for the next run, beside the saved scores
    auto* worker = new RunWorker(std::move(jobs), parallel_jobs(), settings_.gpu_metrics_together, settings_, this);
    worker_ = worker;
    connect(worker, &RunWorker::job_started, this, [this](int index) { on_job_started(index); });
    connect(worker, &RunWorker::task_progress, this,
            [this](int index, const std::vector<TaskSnapshot>& tasks) { on_task_progress(index, tasks); });
    connect(worker, &RunWorker::result_updated, this,
            [this](int index, const ComparisonResult& result) { on_result_updated(index, result); });
    connect(worker, &RunWorker::job_finished, this,
            [this](int index, const ComparisonResult& result) { on_job_finished(index, result); });
    connect(worker, &RunWorker::job_partially_failed, this,
            [this](int index, const ComparisonResult& result, const QString& message, const QString& stderr_tail,
                   const MetricFailures& failures) {
                on_job_partially_failed(index, result, message, stderr_tail, failures);
            });
    connect(worker, &RunWorker::job_failed, this,
            [this](int index, const QString& message, const QString& stderr_tail) {
                on_job_failed(index, message, stderr_tail);
            });
    connect(worker, &RunWorker::cancelled, this, [this] { run_was_cancelled_ = true; });
    connect(worker, &RunWorker::all_finished, this, [this] { on_all_finished(); });
    connect(worker, &QThread::finished, this, [this, worker] { on_run_thread_finished(worker); });
    worker->start();
}

bool MainWindow::confirm_skipping(const std::vector<std::pair<std::shared_ptr<RowData>, QString>>& skipped,
                                  bool others) {
    // Names, all at once, every ticked video that cannot be compared with the
    // reference and why, and asks whether to calculate the rest. Each such
    // video is marked failed with its reason. Don't stop the whole run at the
    // first such video: the videos that can be compared are then not
    // calculated, and the next problem shows only on the next try.
    QStringList listed;
    for (const auto& [row_data, why] : skipped) {
        if (const std::optional<int> row = row_index_of(row_data.get())) {
            set_row_status(*row, RowState::failed, why);
        }
        listed.append("• " + name_of(row_data->path) + ": " + tr_message(why));
    }
    const QString text = app::tr("Cannot be compared with the reference:") + "\n\n" + listed.join("\n");
    if (!others) {
        QMessageBox::warning(this, app::tr("Cannot compare"), text);
        return false;
    }
    return QMessageBox::question(this, app::tr("Cannot compare"), text + "\n\n" + app::tr("Calculate the rest?")) ==
           QMessageBox::Yes;
}

bool MainWindow::wait_for_gpu_checks(const std::vector<int>& checked) {
    // The GPU checks the run's plan reads (which metrics the GPU takes: Vship,
    // VMAF on the GPU, and VMAF v1's when a row has it on the GPU), waited
    // for off the window's thread. They start with the app, but take seconds,
    // and waited for on it, a run asked for before they were done would hold
    // the window still until they were.
    bool v1 = false;
    for (const int row : checked) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        v1 = v1 || (row_data.options.vmaf_on_gpu && contains(requested_metrics(row_data), "vmaf_v1"));
    }
    if (vship_probe_ready() && gpu_vmaf_probe_ready() && (!v1 || v1_probe_ready())) {
        return false;
    }
    if (gpu_check_watcher_ == nullptr) {
        // Meanwhile Run waits, and Cancel drops the run asked for.
        status_before_gpu_checks_ = status_label_->text();
        show_status(app::tr("Checking the GPU before the run\xE2\x80\xA6"));
        run_button_->setEnabled(false);
        cancel_button_->setEnabled(true);
        gpu_check_watcher_ = new QFutureWatcher<void>(this);
        connect(gpu_check_watcher_, &QFutureWatcher<void>::finished, this, [this] {
            stop_waiting_for_gpu_checks();
            on_run_clicked();
        });
        gpu_check_watcher_->setFuture(QtConcurrent::run([v1] {
            detect_vship_device();
            gpu_vmaf_backend();
            if (v1) {
                v1_available();
            }
        }));
    }
    return true;
}

void MainWindow::stop_waiting_for_gpu_checks() {
    // The run asked for waits no more: started now, cancelled or the window
    // closing. Run is usable again, and the status line says what it said
    // before, unless something has said more since (a run says its own).
    gpu_check_watcher_->disconnect(this);
    gpu_check_watcher_->deleteLater();
    gpu_check_watcher_ = nullptr;
    run_button_->setEnabled(true);
    cancel_button_->setEnabled(false);
    if (status_label_->text() == app::tr("Checking the GPU before the run\xE2\x80\xA6")) {
        show_status(status_before_gpu_checks_);
    }
}

bool MainWindow::confirm_long_cpu_perceptual(const std::vector<std::shared_ptr<RowData>>& job_rows) {
    // Asks before SSIMULACRA2 or Butteraugli on the CPU for a video over ten
    // minutes. The CPU tools score one pair of still images at a time, 1-2 s
    // per tool for a 4K pair, which is days for a film. Only metrics this run
    // will calculate count. A metric set to the GPU counts too when no
    // supported GPU was found, because it runs on the CPU just the same.
    QStringList lines;
    std::optional<bool> gpu_missing;  // asked once, and only if needed
    for (const std::shared_ptr<RowData>& row_data : job_rows) {
        if (!row_data->video_info) {
            continue;
        }
        const VideoInfo& info = *row_data->video_info;
        const auto backend = [&row_data](const std::string& key) {
            const auto found = row_data->metric_backends.find(key);
            return found == row_data->metric_backends.end() ? std::string() : found->second;
        };
        const std::vector<std::string> reusable = reusable_metric_keys(*row_data);
        const std::vector<std::string> requested = requested_metrics(*row_data);
        std::vector<std::string> pending;
        for (const std::string key : {"ssimulacra2", "butteraugli"}) {
            if (contains(requested, key) && !contains(reusable, key)) {
                pending.push_back(key);
            }
        }
        const bool any_on_gpu =
            std::any_of(pending.begin(), pending.end(), [&](const std::string& key) { return backend(key) != "cpu"; });
        if (!gpu_missing && any_on_gpu) {
            gpu_missing = !vship_available();
        }
        // Calculated on the CPU: set to it, or set to the GPU where Vship can
        // use none.
        std::vector<std::string> cpu_keys;
        QStringList cpu_metrics;
        for (const std::string& key : pending) {
            if (backend(key) != "cpu" && !gpu_missing.value_or(false)) {
                continue;
            }
            cpu_keys.push_back(key);
            QString label = qt_text(metric_definition(key).label);
            if (backend(key) != "cpu") {
                label += app::tr(" (set to GPU, but no supported GPU was found)");
            }
            cpu_metrics.append(label);
        }
        if (cpu_keys.empty()) {
            continue;
        }
        double seconds = std::min(info.duration, source_info_->duration);
        if (row_data->options.duration_limit > 0) {
            seconds = std::min(seconds, row_data->options.duration_limit);
        }
        if (seconds <= long_cpu_run_seconds) {
            continue;
        }
        const double frames = seconds * (info.fps > 0 ? info.fps : 24.0) / std::max(1, row_data->options.n_subsample);
        const double megapixels = static_cast<double>(info.width) * info.height / 1e6;
        double per_megapixel = 0;
        for (const std::string& key : cpu_keys) {
            per_megapixel += cpu_seconds_per_megapixel(key);
        }
        lines.append(app::tr("• {name}: {time}, {metrics} on CPU — {duration} of scoring",
                             {{"name", name_of(row_data->path)},
                              {"time", format_hms(seconds)},
                              {"metrics", cpu_metrics.join(app::tr(" and "))},
                              {"duration", rough_duration(frames * megapixels * per_megapixel)}}));
    }
    if (lines.isEmpty()) {
        return true;
    }
    const auto answer = QMessageBox::warning(
        this, app::tr("CPU perceptual metrics on long videos"),
        app::tr("Calculating SSIMULACRA2 or Butteraugli on the CPU is not recommended for videos longer than 10 "
                "minutes:\n\n") +
            lines.join("\n") +
            app::tr("\n\nThe CPU tools score one still image pair at a time: about 1 s for SSIMULACRA2 and 2 s for "
                    "Butteraugli per 4K frame, so a film takes days (the estimates above assume a CPU like the one "
                    "they were measured on). Choose GPU for these metrics, or set a duration limit.\n\nCalculate "
                    "anyway?"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    return answer == QMessageBox::Yes;
}

void MainWindow::set_run_ui_active(bool active) {
    // Freezes every input that can change the meaning of a live job.
    run_active_ = active;
    // A run can take hours: the PC must not go to sleep under it. Released as
    // soon as the run ends.
    if (keep_awake()(active)) {
        logger.info(std::string("Windows keep-awake request ") + (active ? "held for the run" : "released"));
    }
    // The table itself stays live, so that the queue can be scrolled and
    // read while it runs. Nothing reachable through it can change a running
    // job: the metric tick boxes refuse edits (on_table_item_changed), and
    // each row's tick is read once, as the run starts.
    for (QWidget* widget : file_action_widgets_) {
        widget->setEnabled(!active);
    }
    // The same for what the table and the reference box take by hand:
    // dropped files, dragged rows, a typed path, a click to sort.
    distorted_table_->drops_allowed = !active;
    distorted_table_->setDragEnabled(!active);
    source_edit_->setReadOnly(active);
    options_panel_->setEnabled(!active && !panel_target_rows_.empty());
    tabs_->setTabEnabled(tab_settings, !active);
    run_button_->setEnabled(!active);
    pause_button_->setEnabled(active);
    cancel_button_->setEnabled(active);
    // Left enabled on purpose: it changes how fast the queue drains, never
    // what any result means, and it is during a run that someone wants it.
    parallel_jobs_check_->setEnabled(true);
}

int MainWindow::parallel_jobs() const { return parallel_jobs_check_->isChecked() ? max_parallel_jobs : 1; }

void MainWindow::on_parallel_jobs_changed(bool /*on*/) {
    // Applies the choice now, and remembers it for next time. A run under
    // way takes it: ticking lets a waiting lane start the next video at once,
    // and unticking stops another from starting without interrupting any.
    const int count = parallel_jobs();
    settings_.parallel_jobs = count;
    save_settings();
    if (worker_ != nullptr && worker_->isRunning()) {
        worker_->set_parallel_jobs(count);
    }
}

void MainWindow::on_cancel_clicked() {
    if (gpu_check_watcher_ != nullptr) {
        stop_waiting_for_gpu_checks();  // nothing started yet: the run is not
        return;
    }
    if (worker_ == nullptr) {
        return;
    }
    worker_->cancel();
    if (!closing_) {
        run_hold_ = RunHold::message;
        run_hold_message_ = app::tr("Cancelling...");
    }
    show_status(app::tr("Cancelling..."));
}

void MainWindow::on_pause_clicked() {
    if (worker_ == nullptr) {
        return;
    }
    if (pause_button_->isChecked()) {
        worker_->pause();
        pause_button_->setText(app::tr("Resume"));
        paused_since_ = clock_();
        run_hold_ = RunHold::paused;
    } else {
        worker_->resume();
        pause_button_->setText(app::tr("Pause"));
        if (paused_since_) {
            paused_total_ += clock_() - *paused_since_;
            paused_since_.reset();
        }
        if (run_hold_ == RunHold::paused) {
            run_hold_ = RunHold::none;
        }
    }
    redraw_job_lines();
    update_run_status();
}

void MainWindow::on_job_started(int index) {
    // Marks the row, and gives the video a line of its own.
    if (const std::optional<int> row = row_index_of(job_rows_[static_cast<std::size_t>(index)].get())) {
        set_row_status(*row, RowState::calculating);
    }
    if (std::find(running_jobs_.begin(), running_jobs_.end(), index) == running_jobs_.end()) {
        running_jobs_.push_back(index);
    }
    if (!job_line_text_.contains(index)) {
        job_line_text_[index] = {app::tr("{label} — starting…", {{"label", job_label(index)}}), QString()};
        arrange_job_lines();
    }
    update_run_status();
}

void MainWindow::redraw_job_lines() {
    // Every video's line with figures, now: on Pause and Resume.
    for (const auto& [index, line] : job_line_text_) {
        if (job_task_progress_.contains(index)) {
            render_job_progress(index);
        }
    }
}

void MainWindow::set_job_line(int index, const QString& text, const QString& tooltip) {
    // What a video's line says, kept so that the lines can be laid out again.
    const auto line = job_line_text_.find(index);
    if (line == job_line_text_.end()) {
        return;
    }
    line->second = {text, tooltip};
    if (const auto slot = job_line_slot_.find(index); slot != job_line_slot_.end()) {
        job_progress_labels_[static_cast<std::size_t>(slot->second)]->set_text(text, tooltip);
    }
}

void MainWindow::arrange_job_lines() {
    // The videos in progress, one line each, in list order. Don't give a
    // video the first free line: a later video can then sit above an earlier
    // one, the lines change places as videos finish, and a reused line keeps
    // the previous video's tooltip.
    std::vector<int> order;
    for (const auto& [index, line] : job_line_text_) {  // in index order
        if (order.size() == job_progress_labels_.size()) {
            break;
        }
        order.push_back(index);
    }
    job_line_slot_.clear();
    for (std::size_t slot = 0; slot < job_progress_labels_.size(); ++slot) {
        ElidedLabel* label = job_progress_labels_[slot];
        if (slot < order.size()) {
            job_line_slot_[order[slot]] = static_cast<int>(slot);
            const auto& [text, tooltip] = job_line_text_.at(order[slot]);
            label->setVisible(true);
            label->set_text(text, tooltip);
        } else {
            label->setVisible(false);
            label->clear_text();
        }
    }
}

void MainWindow::mark_job_over(int index) {
    // Takes a video out of the live figures and frees its line.
    finished_jobs_.insert(index);
    job_task_progress_.erase(index);
    job_lines_due_.erase(index);
    job_line_shape_.erase(index);
    if (job_line_text_.erase(index) > 0) {
        arrange_job_lines();
    }
    update_run_status();
}

QString MainWindow::job_label(int index) const {
    // The name a video is shown with: its file's, without the extension.
    if (index >= 0 && index < static_cast<int>(job_rows_.size())) {
        return text_of(job_rows_[static_cast<std::size_t>(index)]->path.stem());
    }
    return app::tr("video {number}", {{"number", index + 1}});
}

void MainWindow::update_run_status() {
    // How much is running, and for how long. There is no time left for the
    // whole queue: the videos' metrics run at very different speeds, in a
    // CPU queue and a GPU queue that overlap, and a video not started has no
    // speed of its own, so it would swing by minutes. Each video's own time
    // left is on its line. No video is named here: each running video has
    // its own line below, with its name.
    if (run_hold_ == RunHold::message) {
        show_status(run_hold_message_);
        return;
    }
    const int running = static_cast<int>(std::count_if(running_jobs_.begin(), running_jobs_.end(),
                                                       [this](int index) { return !finished_jobs_.contains(index); }));
    const int total = static_cast<int>(job_rows_.size());
    const std::optional<double> elapsed = run_elapsed();
    const QString elapsed_text =
        elapsed ? "   ·   " + app::tr("Elapsed: {time}", {{"time", format_hms(*elapsed)}}) : QString();
    const QString summary = run_summary(running, total);
    if (run_hold_ == RunHold::paused) {
        show_status(app::tr("Paused   ·   ") + summary + elapsed_text);
        return;
    }
    if (running == 0 && !(run_active_ && total > 0)) {
        return;
    }
    show_status(summary + elapsed_text);
}

std::optional<double> MainWindow::run_elapsed() const {
    // Seconds the run has been calculating: paused time left out.
    if (!run_started_at_) {
        return std::nullopt;
    }
    const double now = clock_();
    const double paused = paused_total_ + (paused_since_ ? now - *paused_since_ : 0.0);
    return std::max(0.0, now - *run_started_at_ - paused);
}

QString MainWindow::run_end_message() const {
    // How the run ended and how long it took, such as "Finished in 2:29:51:
    // 1 video failed, 1 with some metrics failed (hover over their names for
    // why)." A video with some metrics failed and the rest scored is counted
    // apart from a failed video.
    const std::optional<double> elapsed = run_elapsed();
    const QString took = elapsed ? QString::fromStdString(format_hms(*elapsed)) : QString();
    if (run_was_cancelled_) {
        return elapsed ? app::tr("Cancelled after {time}.", {{"time", took}}) : app::tr("Cancelled.");
    }
    QStringList issues;
    if (run_failed_count_ > 0) {
        issues.append(ntr("{count} video failed", "{count} videos failed", run_failed_count_));
    }
    if (run_partial_count_ > 0) {
        issues.append(app::tr("{count} with some metrics failed", {{"count", run_partial_count_}}));
    }
    if (issues.isEmpty()) {
        return elapsed ? app::tr("Done in {time}.", {{"time", took}}) : app::tr("Done.");
    }
    const QString why = run_failed_count_ + run_partial_count_ == 1 ? app::tr("hover over its name for why")
                                                                    : app::tr("hover over their names for why");
    const QString listed = issues.join(app::tr(", "));
    if (!elapsed) {
        return app::tr("Finished: {issues} ({why}).", {{"issues", listed}, {"why", why}});
    }
    return app::tr("Finished in {time}: {issues} ({why}).", {{"time", took}, {"issues", listed}, {"why", why}});
}

QString MainWindow::run_summary(int running, int total) const {
    // The queue in counts, such as "3 videos: 1 done, 2 in progress". "In
    // progress", not "running": a video whose CPU metrics are done while it
    // waits for the GPU is in progress. Videos skipped because they were
    // already scored are counted too, because the message saying so at the
    // start is replaced within a second.
    const int done = static_cast<int>(finished_jobs_.size());
    const int queued = std::max(0, total - done - running);
    // Failures as they happen: otherwise a video that failed would leave the
    // lines below and count as done until the run's end message said so.
    QStringList failures;
    if (run_failed_count_ > 0) {
        failures.append(app::tr("{count} failed", {{"count", run_failed_count_}}));
    }
    if (run_partial_count_ > 0) {
        failures.append(app::tr("{count} with failed metrics", {{"count", run_partial_count_}}));
    }
    QStringList counts;
    if (done > 0) {
        counts.append(app::tr("{count} done", {{"count", done}}) +
                      (failures.isEmpty() ? QString() : " (" + failures.join(app::tr(", ")) + ")"));
    }
    if (running > 0) {
        counts.append(app::tr("{count} in progress", {{"count", running}}));
    }
    if (queued > 0) {
        counts.append(app::tr("{count} queued", {{"count", queued}}));
    }
    QString summary =
        ntr("{count} video: {counts}", "{count} videos: {counts}", total, {{"counts", counts.join(app::tr(", "))}});
    if (run_skipped_ > 0) {
        summary += app::tr(" ({count} already scored, not recalculated)", {{"count", run_skipped_}});
    }
    return summary;
}

void MainWindow::pause_free_rates(int index, std::vector<TaskSnapshot>& tasks) {
    // Each running task's recent rate: its frames over the run's own clock,
    // which leaves pauses out (run_elapsed), over the last ten seconds once
    // there are two. Don't use the rate the tasks report: it is an average
    // from their start (FFmpeg's own fps, and Vship's passes') that counts
    // pauses: after a 4-second pause it reads about half the speed, and takes
    // most of the task to recover. A pass of a GPU task, or a retry that
    // counts again from the start, is measured afresh.
    const std::optional<double> now = run_elapsed();
    if (!now) {
        return;
    }
    for (TaskSnapshot& task : tasks) {
        if (task.state != "running") {
            continue;
        }
        auto& samples = rate_samples_[{index, task.backend, task.phase ? (*task.phase)[0] : 0}];
        if (!samples.empty() && task.current < samples.back().second) {
            samples.clear();  // counting again from the start: a retry
        }
        samples.emplace_back(*now, task.current);
        while (samples.size() > 2 && *now - samples[1].first >= rate_window_seconds) {
            samples.pop_front();
        }
        const auto [since, then] = samples.front();
        if (*now - since >= rate_min_seconds && task.current > then) {
            task.fps = static_cast<double>(task.current - then) / (*now - since);
        }
    }
}

void MainWindow::on_task_progress(int index, std::vector<TaskSnapshot> tasks) {
    // Kept; the line is drawn again at once only when what it says apart
    // from its figures changes, and its figures on the next tick.
    if (finished_jobs_.contains(index)) {
        return;  // the video is over; its line is gone
    }
    pause_free_rates(index, tasks);
    LineShape shape = line_shape(tasks);
    job_task_progress_[index] = std::move(tasks);
    const auto known = job_line_shape_.find(index);
    if (known == job_line_shape_.end() || known->second != shape) {
        job_line_shape_[index] = std::move(shape);
        render_job_progress(index);
        job_lines_due_.erase(index);
    } else {
        job_lines_due_.insert(index);
    }
}

void MainWindow::on_run_tick() {
    // Once a second during a run: the lines with new figures, then the
    // status line (the time it has taken).
    for (const int index : std::exchange(job_lines_due_, {})) {
        render_job_progress(index);
    }
    update_run_status();
}

void MainWindow::render_job_progress(int index) {
    // Draws a video's line from its tasks' latest state.
    const auto tasks = job_task_progress_.find(index);
    if (!job_line_text_.contains(index) || tasks == job_task_progress_.end() || tasks->second.empty()) {
        return;
    }
    const MetricLine line = metric_line(tasks->second, run_hold_ == RunHold::paused, window_wording());
    QStringList parts;
    for (const std::string& part : line.parts) {
        parts.append(qt_text(part));
    }
    if (!line.plans.empty()) {
        // A resolution test decodes only its source.
        const bool source_only = job_rows_[static_cast<std::size_t>(index)]->options.resample_test.has_value();
        parts.append(qt_text(decoder_text(line.plans, source_only, window_wording())));
    }
    set_job_line(index, job_label(index) + " — " + parts.join("   ·   "), qt_text(line.tooltip));
}

std::optional<int> MainWindow::row_index_of(const RowData* row_data) const {
    // The table row a row's data is at now, or nothing if it was removed
    // during the run. Found by identity: two rows can hold equal data.
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        if (rows_[static_cast<std::size_t>(row)].get() == row_data) {
            return row;
        }
    }
    return std::nullopt;
}

void MainWindow::on_result_updated(int index, const ComparisonResult& result) {
    // A video's result so far, saved and shown now: one of its tasks
    // finished while the rest runs on. Were it kept only once the whole video
    // was done, a Butteraugli score would wait hours for the video's VMAF,
    // and closing the app or a crash in between would lose it.
    if (finished_jobs_.contains(index)) {
        return;  // the video's own result has arrived: it is the one to show
    }
    on_job_finished(index, result, false);
}

std::optional<int> MainWindow::on_job_finished(int index, const ComparisonResult& result, bool final) {
    // Saves and shows a finished video's result, or with `final` false its
    // result so far (on_result_updated). Returns the row it was shown on, or
    // nothing when it was not: the row was removed, or the reference or the
    // row's settings changed since the video started.
    const auto job = static_cast<std::size_t>(index);
    RowData& row_data = *job_rows_[job];
    if (final) {
        mark_job_over(index);
        row_data.metric_failures.clear();
    }
    const std::optional<int> row = row_index_of(&row_data);
    if (!row) {
        return std::nullopt;  // the row was removed during the run: nothing to show the result in
    }
    const QString label = text_of(row_data.path.stem());
    save_job_result(job, row_data, result, label);
    // The packets' bitrate goes well with the scores, and costs much less
    // than they did: both files go to the Bitrate tab, which reads a file
    // once however many runs finish with it. A resolution test has no
    // second file.
    if (final) {
        std::vector<VideoInfo> bitrate_infos{result.source_info};
        if (!result.resample_target) {
            bitrate_infos.push_back(result.distorted_info);
        }
        bitrate_panel_->add_and_analyze(bitrate_infos);
    }
    if (!source_info_ || !same_path(source_info_->path, result.source)) {
        if (final) {
            set_row_status(*row, RowState::for_previous_source);
        }
        return std::nullopt;
    }
    if (job_options_[job] != row_data.options || !job_cvvdp_[job].same_as(row_data.cvvdp)) {
        // The row's settings changed after the video started, so the result
        // does not match what the row says now: shown, it would be labelled
        // with settings it was not calculated with.
        if (final) {
            set_row_status(*row, RowState::for_previous_settings);
        }
        return std::nullopt;
    }
    const std::shared_ptr<CompletedRun> previous = row_data.completed_run;
    // The video's own result so far, before this piece: its line is updated.
    const bool continuing = previous && previous->partial;
    const std::shared_ptr<CompletedRun> run = make_completed_run(row_data, result, label, final);
    row_data.completed_run = run;
    if (final) {
        row_data.analysis_status.reset();  // a result so far leaves the video in progress
    }
    if (!row_data.options.resample_test) {
        set_row_info(*row, result.distorted_info);  // its Scaling cell, with the crops the run used
    }
    if (final) {
        row_data.status_detail = scores_detail(run->result);
    }
    set_row_metrics(*row);
    // Straight onto the graph: a finished video is a line, and waiting for a
    // button to see it serves nobody.
    graph_panel_->add_run({std::shared_ptr<const ComparisonResult>(run, &run->result), run->label,
                           run->graph_identity, !continuing});
    sync_frame_compare();
    return row;
}

// Saves `result`, job `job`'s on `row_data`, in the saved scores.
// Saved under the videos and settings the video started with, never the
// reference chosen by the time it ends; and under the row's real file,
// whose size and time notice a file replaced. Off the window's thread: a
// long video's scores take seconds to write.
void MainWindow::save_job_result(std::size_t job, const RowData& row_data, const ComparisonResult& result,
                                 const QString& label) {
    const AnalysisRequest request =
        analysis_request(job_options_[job], requested_metrics(row_data), row_data.metric_backends, job_cvvdp_[job]);
    file_writes_->submit(QStringLiteral("cache %1").arg(label),
                         [base = cache_folder(), identity = row_data.identity_path(), result,
                          label = label.toStdString(), request] {
                             store_result(base, result.source, identity, request.recipe, result, label,
                                          request.metrics);
                         });
}

// The row's new run: `result` (`final`, or so far), labelled `label`, in
// place of the row's run before, whose line leaves the graph unless this
// continues it.
std::shared_ptr<CompletedRun> MainWindow::make_completed_run(const RowData& row_data, const ComparisonResult& result,
                                                             const QString& label, bool final) {
    const std::shared_ptr<CompletedRun> previous = row_data.completed_run;
    // The video's own result so far, before this piece: its line is updated.
    const bool continuing = previous && previous->partial;
    if (previous && !continuing && !graph_panel_->remove_by_identity(previous->graph_identity)) {
        graph_panel_->remove_by_path(row_data.path);
    }
    auto run = std::make_shared<CompletedRun>();
    run->result = result;
    run->label = label;
    run->partial = !final;
    if (continuing) {
        // The same line as the pieces before: its colour, and whether it was
        // hidden, stay as they were.
        run->graph_identity = previous->graph_identity;
    }
    if (previous) {
        // Scores the row was showing for metrics this run did not calculate
        // (unticked ones) stay shown, as the run's result holds only what it
        // calculated.
        const MetricResultSet& before = previous->result.metric_results;
        const std::vector<std::string> requested = requested_metrics(row_data);
        for (const std::string& key : before.keys()) {
            if (result.has_metric(key) || contains(requested, key)) {
                continue;
            }
            run->result.metric_results.copy_from(before, key);
        }
    }
    return run;
}

QString MainWindow::scores_detail(const ComparisonResult& result) const {
    QStringList labels;
    for (const MetricDefinition& metric : metrics()) {
        if (result.has_metric(std::string(metric.key))) {
            labels.append(qt_text(metric.label));
        }
    }
    const auto frames = static_cast<long long>(frame_scores_from_results(result.metric_results).frame.size());
    return ntr("{count} scored frame; metrics: ", "{count} scored frames; metrics: ", frames) + labels.join(", ");
}

void MainWindow::on_job_partially_failed(int index, const ComparisonResult& result, const QString& message,
                                         const QString& stderr_tail, const MetricFailures& failures) {
    // Some of a video's metrics failed and the others finished: the finished
    // scores are shown like any result, and the metrics that failed say so,
    // and why, in their own cells.
    ++run_partial_count_;
    const std::optional<int> row = on_job_finished(index, result);
    if (!row) {
        return;
    }
    RowData& row_data = *rows_[static_cast<std::size_t>(*row)];
    for (const auto& [key, why] : failures) {
        row_data.metric_failures[key] = QString::fromStdString(why);
    }
    set_row_status(*row, RowState::partly_failed, stderr_tail.isEmpty() ? message : message + "\n\n" + stderr_tail);
    set_row_metrics(*row);
}

void MainWindow::on_job_failed(int index, const QString& message, const QString& stderr_tail) {
    ++run_failed_count_;  // first: mark_job_over shows the counts
    mark_job_over(index);
    const std::optional<int> row = row_index_of(job_rows_[static_cast<std::size_t>(index)].get());
    if (!row) {
        return;  // the row was removed during the run
    }
    rows_[static_cast<std::size_t>(*row)]->metric_failures.clear();  // the video's reason is every metric's
    set_row_status(*row, RowState::failed, stderr_tail.isEmpty() ? message : message + "\n\n" + stderr_tail);
    set_row_metrics(*row);
}

void MainWindow::on_all_finished() {
    // Ends the run: settles the rows it did not finish, says how it ended
    // (in the log too), and shows its results on Metric Graphs.
    for (const std::shared_ptr<RowData>& row_data : job_rows_) {
        const std::optional<int> row = row_index_of(row_data.get());
        if (!row || (row_data->analysis_status != RowState::calculating &&
                     row_data->analysis_status != RowState::queued)) {
            continue;
        }
        // A video the run never reached is as it was, not cancelled.
        if (run_was_cancelled_ && row_data->analysis_status == RowState::calculating) {
            row_data->analysis_status = RowState::cancelled;
        } else {
            row_data->analysis_status.reset();
        }
        set_row_metrics(*row);
    }
    run_tick_timer_->stop();
    run_hold_ = RunHold::none;
    set_run_ui_active(false);
    pause_button_->setChecked(false);
    pause_button_->setText(app::tr("Pause"));
    for (ElidedLabel* line : job_progress_labels_) {
        line->setVisible(false);
    }
    show_status(run_end_message());
    {
        const InEnglish english;  // the log is always in English
        logger.info("Run status: " + run_end_message().toStdString());
    }
    // Every row ticked, those skipped as already scored too, so that the
    // graph shows all that was asked for; rows removed during the run left
    // out.
    std::vector<std::shared_ptr<CompletedRun>> finished;
    for (const std::shared_ptr<RowData>& row_data : checked_rows_for_run_) {
        if (row_data->completed_run && row_index_of(row_data.get())) {
            finished.push_back(row_data->completed_run);
        }
    }
    if (!finished.empty()) {
        // A cancelled run stays on the Videos tab, where Cancel was pressed
        // and the unfinished rows are; what it finished is on the graph all
        // the same.
        open_or_update_graph(finished, !run_was_cancelled_);
    }
}

void MainWindow::on_run_thread_finished(RunWorker* worker) {
    // Lets a run's thread go once it has ended (after on_all_finished), and
    // with it the run's jobs and the results they carry.
    if (worker == worker_) {
        worker_ = nullptr;
    }
    worker->deleteLater();
}

}  // namespace vml::app
