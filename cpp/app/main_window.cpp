#include "main_window.h"

#include "quick_save.h"

#include <QMenu>
#include <QAbstractItemView>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QEvent>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QProcess>
#include <QPushButton>
#include <QScreen>
#include <QSplitter>
#include <QStyleHints>
#include <QTableWidgetItem>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <tuple>

#include "bd_rate_panel.h"
#include "bitrate_panel.h"
#include "export_queue.h"
#include "cvvdp_display_dialog.h"
#include "ffmpeg_download_dialog.h"
#include "file_writes.h"
#include "frame_compare_panel.h"
#include "graph_panel.h"
#include "worst_frames_tab.h"
#include "i18n.h"
#include "keyboard.h"
#include "metric_list.h"
#include "metric_order.h"
#include "notifier.h"
#include "options_panel.h"
#include "run_worker.h"
#include "series_colors.h"
#include "settings_panel.h"
#include "stream_records.h"
#include "theme.h"
#include "video_names.h"
#include "click_twice.h"
#include "title_bar.h"
#include "update_offer.h"
#include "vml/app_log.h"
#include "vml/append.h"
#include "vml/ffmpeg_tools.h"
#include "vml/file_fingerprint.h"
#include "vml/hardware_decoding.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/python_text.h"
#include "vml/platform/system.h"
#include "vml/score_identity.h"
#include "vml/segment_profiles.h"
#include "vml/vship.h"
#include "widgets.h"
#include "window_helpers.h"

// The window's text is app::tr's (i18n.h), always written so: Q_OBJECT gives
// the window a tr() of Qt's own, which inside its functions would hide the
// app's and show the English.
namespace vml::app {

namespace {

const app_log::Logger logger("vml.main_window");

// The window's size where none was saved: 1440x900, at most 90 % of the
// screen's room (a 1080p laptop at 150 % has about 1280x690 of it, where
// 1440x900 opened past its edges). The least size (1050x650) still holds.
QSize default_window_size(const QScreen* screen) {
    const QSize preferred(1440, 900);
    if (screen == nullptr) {
        return preferred;
    }
    const QSize room = screen->availableGeometry().size();
    return preferred.boundedTo(QSize(room.width() * 9 / 10, room.height() * 9 / 10));
}

// What is wrong with FFmpeg, one text each.
QStringList tool_problems(const ToolsStatus& status) {
    QStringList problems;
    for (const std::string& problem : status.problems()) {
        problems.append(QString::fromStdString(problem));
    }
    return problems;
}

// FFmpeg's version and the GPUs found, for this session's log, from the
// checks made at the start (the Python app's log_tools).
void log_tools(const ToolsStatus& status, const std::vector<GpuVendor>& gpus) {
    const app_log::Logger main_logger("vml.main");
    if (status.ok()) {
        main_logger.info("FFmpeg " + format_version(status.ffmpeg.version) + ": " + status.ffmpeg.path);
    } else {
        for (const std::string& problem : status.problems()) {
            main_logger.warning("FFmpeg: " + problem);
        }
    }
    // Only where the system lists its GPUs: on macOS and Linux the list is
    // empty whatever the GPU, which is not "none detected"; the GPU metrics'
    // own lines name the GPU there.
    if (!platform::lists_gpus()) {
        return;
    }
    std::vector<std::string_view> names;
    for (const GpuVendor vendor : gpus) {
        names.push_back(vendor_label(vendor));
    }
    const std::string vendors = join(names, ", ");
    main_logger.info("GPUs: " + (vendors.empty() ? std::string("none detected") : vendors));
}

// The videos in progress at once: two with the CPU's metrics, and one more
// on the GPU (the Python app's job_runner MAX_VIDEOS_IN_FLIGHT).
constexpr int max_videos_in_flight = max_parallel_jobs + 1;

}  // namespace

namespace {

// Each metric's column of the test-video table, by its stable index.
const std::vector<MetricColumn>& column_of_each_metric() {
    static const std::vector<MetricColumn> columns{
        {col_vmaf, "vmaf"},
        {col_vmaf_neg, "vmaf_neg"},
        {col_vmaf_v1, "vmaf_v1"},
        {col_vmaf_hdr, "vmaf_hdr"},
        {col_psnr, "psnr"},
        {col_ssim, "ssim"},
        {col_xpsnr, "xpsnr"},
        {col_psnr_yuv, "psnr_yuv"},
        {col_psnr_yuv_y, "psnr_yuv_y"},
        {col_psnr_yuv_u, "psnr_yuv_u"},
        {col_psnr_yuv_v, "psnr_yuv_v"},
        {col_psnr_yuv_mse, "psnr_yuv_mse"},
        {col_ssim_yuv, "ssim_yuv"},
        {col_ssim_yuv_y, "ssim_yuv_y"},
        {col_ssim_yuv_u, "ssim_yuv_u"},
        {col_ssim_yuv_v, "ssim_yuv_v"},
        {col_ms_ssim, "ms_ssim"},
        {col_ms_ssim_yuv, "ms_ssim_yuv"},
        {col_ms_ssim_yuv_y, "ms_ssim_yuv_y"},
        {col_ms_ssim_yuv_u, "ms_ssim_yuv_u"},
        {col_ms_ssim_yuv_v, "ms_ssim_yuv_v"},
        {col_psnr_hvs, "psnr_hvs"},
        {col_psnr_hvs_y, "psnr_hvs_y"},
        {col_psnr_hvs_u, "psnr_hvs_u"},
        {col_psnr_hvs_v, "psnr_hvs_v"},
        {col_cambi, "cambi"},
        {col_ciede2000, "ciede2000"},
        {col_vmaf_yuv, "vmaf_yuv"},
        {col_vmaf_yuv_y, "vmaf_yuv_y"},
        {col_vmaf_yuv_u, "vmaf_yuv_u"},
        {col_vmaf_yuv_v, "vmaf_yuv_v"},
        {col_ssimulacra2, "ssimulacra2"},
        {col_butteraugli, "butteraugli"},
        {col_cvvdp, "cvvdp"},
    };
    return columns;
}

}  // namespace

const std::vector<MetricColumn>& metric_columns() {
    // In the order shown (shown_metrics: a metric's planes after it), made
    // again when the user orders the metrics otherwise.
    static std::vector<const MetricDefinition*> made_for;
    static std::vector<MetricColumn> columns;
    if (made_for != shown_metrics()) {
        made_for = shown_metrics();
        columns.clear();
        for (const MetricDefinition* metric : made_for) {
            for (const MetricColumn& item : column_of_each_metric()) {
                if (metric->key == item.key) {
                    columns.push_back(item);
                }
            }
        }
    }
    return columns;
}

const MetricColumn* metric_column_of(int column) {
    for (const MetricColumn& item : metric_columns()) {
        if (item.column == column) {
            return &item;
        }
    }
    return nullptr;
}

// ======================================================================= window

MainWindow::MainWindow() {
    setWindowTitle(QStringLiteral("VideoMetricsLab %1").arg(VML_VERSION));
    // Settings first: the ffmpeg location and what new rows default to both
    // come from them, so they must be applied before the tool check or any
    // row is added.
    settings_ = Settings::load();
    set_metric_order(settings_.metric_order);  // before any list of metrics is made
    logger.info(std::string("Settings: parallel CPU metrics ") + (settings_.parallel_jobs > 1 ? "on" : "off") +
                ", GPU decode " + (settings_.default_gpu_decode ? "on" : "off") + ", VVC decoded on the " +
                (settings_.vvc_decode_on_gpu ? "GPU" : "CPU") + ", VMAF v0.6.1 and NEG on " +
                (settings_.default_vmaf_on_gpu ? "GPU" : "CPU") + ", CAMBI on " +
                (settings_.default_cambi_on_gpu ? "GPU" : "CPU") + ", CIEDE2000 on " +
                (settings_.default_ciede2000_on_gpu ? "GPU" : "CPU") + ", MS-SSIM on " +
                (settings_.default_ms_ssim_on_gpu ? "GPU" : "CPU") + ", GPU metrics " +
                (settings_.gpu_metrics_together ? "in one pass" : "one at a time") + " with " +
                std::string(vship_build_label(default_vship_build(settings_.default_vship_build))) +
                ", saved results " + (settings_.use_cache ? "reused" : "not reused") + " (" +
                (settings_.cache_dir.empty() ? std::string("default folder") : settings_.cache_dir) + ")");
    // Where it was closed, maximized if it was (shown so by main's show());
    // Qt moves it onto a screen still there.
    const bool restored = settings_.remember_window_size && !settings_.window_geometry.empty() &&
                          restoreGeometry(QByteArray::fromBase64(QByteArray::fromStdString(settings_.window_geometry)));
    if (!restored) {
        // Before a place was first saved, or if it cannot be read, at the
        // size it was closed at; where none was ever saved (the settings'
        // default size), or none is remembered, at the default for the
        // screen.
        const Settings defaults;
        const bool saved_size = settings_.remember_window_size &&
                                (settings_.window_width != defaults.window_width ||
                                 settings_.window_height != defaults.window_height);
        resize(saved_size ? QSize(settings_.window_width, settings_.window_height) : default_window_size(screen()));
        // Centred in the screen's room. Qt placed the window for its size
        // before this resize: on a 4K screen at 175 % a 1440x900 window then
        // ran 312 px past its right edge and 245 px past its bottom (#173).
        if (const QScreen* room = screen(); room != nullptr) {
            move(room->availableGeometry().center() - QPoint(width() / 2, height() / 2));
        }
    }
    // Each video's own settings; the Options panel shows and edits the
    // selected rows' settings, and there is no one global setting.
    default_options_ = options_from_settings();
    default_extra_metric_keys_ = extra_metrics_from_settings();
    for (const std::string& key : settings_.hidden_metrics) {
        if (std::any_of(metric_columns().begin(), metric_columns().end(), [&key](const MetricColumn& item) {
                return key == item.key && !metric_definition(item.key).is_plane();
            })) {
            hidden_metrics_.insert(key);
        }
    }
    if (static_cast<long long>(hidden_metrics_.size()) >= listed_metric_count()) {
        hidden_metrics_.clear();  // a settings file that hides everything shows everything
    }

    // A run's clock, and its tick: once a second, its lines' new figures and
    // the time it has taken.
    clock_ = seconds_now;
    run_tick_timer_ = new QTimer(this);
    run_tick_timer_->setInterval(1000);
    connect(run_tick_timer_, &QTimer::timeout, this, [this] { on_run_tick(); });
    // Saved scores are written off the window's thread. A failure is said on
    // the status line, not in a dialog: a dialog taking the focus minutes
    // later would be worse than the failure it announces.
    file_writes_ = new FileWrites(this);
    update_offer_ = new UpdateOffer({this, settings_, [this] { save_settings(); }, [this] { return title_buttons_; },
                                     [this] { return cache_folder(); }});
    stream_records_ = std::make_shared<StreamRecords>();
    // What the records read, on the status line: queued to the window's
    // thread (the records' own calls it from theirs).
    stream_records_->set_reporting(
        [this](const std::filesystem::path& test) {
            QMetaObject::invokeMethod(
                this,
                [this, name = name_of(test)] {
                    show_stream_status(app::tr("Recording {name}'s stream, so that a remux of it keeps its scores\xE2\x80\xA6",
                                               {{"name", name}}));
                },
                Qt::QueuedConnection);
        },
        [this] { QMetaObject::invokeMethod(this, [this] { end_stream_status(); }, Qt::QueuedConnection); });
    connect(file_writes_, &FileWrites::write_failed, this, [this](const QString& description, const QString& error) {
        notify_failed(app::tr("Could not {description}: {error}", {{"description", description}, {"error", error}}));
    });

    // Before the window is built: building it may already change the videos.
    open_videos_timer_ = new QTimer(this);
    open_videos_timer_->setSingleShot(true);
    open_videos_timer_->setInterval(0);
    connect(open_videos_timer_, &QTimer::timeout, this, [this] { save_open_videos(); });
    // Where the focus is, shown while it is moved with the keys (VML Modern's ring).
    keyboard::install();
    build_ui();
    // At least 1050x650, and never smaller than the window's contents need.
    setMinimumSize(QSize(1050, 650).expandedTo(minimumSizeHint()));
    theme_refresh_due_ = false;
    // Light or dark changed (Settings, or Windows when following it).
    connect(QGuiApplication::styleHints(), &QStyleHints::colorSchemeChanged, this, [this] { theme_changed(); });
    use_crop_store();  // black bars found are kept beside the saved scores
    check_ffmpeg(true);  // the start's check: both tools present, ffmpeg new enough
    // The rows say which metrics can be calculated here (CVVDP needs a GPU
    // Vship can use) without waiting for the GPU check that starts with the
    // app (window_helpers.h), and are brought up to date when it finishes:
    // asked for before the first row, so that none misses it.
    on_vship_probe_finished([this] {
        QMetaObject::invokeMethod(this, [this] { on_vship_checked(); }, Qt::QueuedConnection);
    });
    // The saved scores' lookups (restore_open_videos' first of them) know a
    // remuxed reference from the start: check_ffmpeg, above, gave them the
    // tools Settings name now.
    restore_open_videos();
    on_table_selection_changed();
}

MainWindow::~MainWindow() {
    on_vship_probe_finished(nullptr);  // the window is going: a check finishing now has nothing to tell
    if (open_videos_lookup_) {
        const std::lock_guard lock(open_videos_lookup_->mutex);
        open_videos_lookup_->window = nullptr;  // its answer has no window to go to now
    }
    // Hidden first, while the window is whole: hiding moves the focus, and
    // the reference box reads what it holds when the focus leaves it.
    hide();
    // Each stops, ending the ffprobe reading a video.
    for (ProbeWorker* worker : probe_workers_) {
        worker->cancel();
    }
    // A run ends every program it started.
    if (worker_ != nullptr) {
        worker_->cancel();
    }
    for (ProbeWorker* worker : probe_workers_) {
        worker->wait();
    }
    if (worker_ != nullptr) {
        worker_->wait();
    }
    if (run_active_) {
        set_run_ui_active(false);  // the PC may sleep again
    }
}

void MainWindow::changeEvent(QEvent* event) {
    // The palette change comes after the colour scheme's signal; the style
    // sheets resolve palette() from the palette, which the signal may have
    // come before.
    if (event->type() == QEvent::PaletteChange || event->type() == QEvent::ApplicationPaletteChange) {
        theme_changed();
    }
    QMainWindow::changeEvent(event);
}

void MainWindow::theme_changed() {
    // Once for each burst of these signals and events; not while the window
    // is being built, which clears the flag when it is done.
    if (!theme_refresh_due_) {
        theme_refresh_due_ = true;
        QTimer::singleShot(0, this, [this] { apply_theme_colours(); });
    }
}

void MainWindow::apply_theme_colours() {
    theme_refresh_due_ = false;
    // Once per theme: the colour scheme's signal and the palette's event both
    // ask, and the second finds the work done.
    const QPalette palette = QApplication::palette();
    const std::tuple<bool, unsigned, unsigned> applied{theme::is_dark(), palette.window().color().rgba(),
                                                       palette.highlight().color().rgba()};
    if (theme_applied_ == applied) {
        return;
    }
    theme_applied_ = applied;
    theme::refresh();
    make_title_buttons();  // VML Modern chosen in Settings: its buttons too
    if (title_buttons_ != nullptr) {
        show_theme_in_title();
    }
    // Only the cells with a colour of their own, each to its counterpart.
    for (int row = 0; row < distorted_table_->rowCount(); ++row) {
        for (int column = 0; column < distorted_table_->columnCount(); ++column) {
            if (QTableWidgetItem* item = distorted_table_->item(row, column)) {
                theme::recolour(item);
            }
        }
    }
    apply_best_worst();  // the backgrounds, in the new theme's greens and reds
    graph_panel_->refresh_theme();  // its table's colours are set cell by cell too
    bd_rate_panel_->refresh_theme();
}

void MainWindow::close_when_idle() {
    if (closing_) {
        close();
    }
}

std::vector<QThread*> MainWindow::live_workers() const {
    std::vector<QThread*> workers;
    for (ProbeWorker* worker : probe_workers_) {
        if (worker->isRunning()) {
            workers.push_back(worker);
        }
    }
    if (worker_ != nullptr && worker_->isRunning()) {
        workers.push_back(worker_);
    }
    for (QThread* worker : frame_compare_panel_->live_workers()) {
        workers.push_back(worker);
    }
    for (QThread* worker : bitrate_panel_->live_workers()) {
        workers.push_back(worker);
    }
    return workers;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    // A run in progress, paused or not, stops when the window closes: asked
    // first, unless told not to. Not while Windows ends the session, which
    // a question would hold up.
    if (!closing_ && run_active_ && settings_.confirm_close_during_run && !qApp->isSavingSession()) {
        const std::optional<CloseAnswer> answer = dialogs_.ask_close_during_run();
        if (!answer) {
            event->ignore();  // Keep running
            return;
        }
        if (answer->dont_ask_again) {
            settings_.confirm_close_during_run = false;
            save_settings();
            if (settings_panel_ != nullptr) {
                settings_panel_->show_confirm_close();
            }
        }
    }
    // Don't close while background work still writes into the widgets: it
    // would crash the app and leave the process running. The work is told
    // to stop (a run ends the programs it started), and the window closes
    // itself once it has.
    if (!closing_) {
        closing_ = true;
        close_clock_.start();
        if (gpu_check_watcher_ != nullptr) {
            stop_waiting_for_gpu_checks();  // a run asked for is not started now
        }
        if (run_waits_for_saved_scores_) {
            stop_waiting_for_saved_scores();
        }
        frame_compare_panel_->cancel();
        bitrate_panel_->cancel();
        // An export waiting for the bitrate the Bitrate tab was reading is
        // written now, without the bitrate not read, rather than lost: the
        // user was told it would be written (ExportQueue::queue), and the
        // writes are waited for below.
        export_queue_->write_ready(true);
        stream_records_->set_reporting({}, {});  // nothing more for the window
        stream_records_->stop();                 // a test stream being read is recorded at its next save
        cancel_stream_check();
        for (ProbeWorker* worker : probe_workers_) {
            worker->cancel();
        }
        if (worker_ != nullptr) {
            worker_->cancel();
        }
    }
    // Nor during a write of saved scores or a saved run, which closing would
    // lose, or of a graph's export, which it would cut short: the graph has
    // writes of its own.
    const bool written = file_writes_->wait_until_idle(std::chrono::milliseconds(500)) &&
                         graph_panel_->wait_until_file_writes_idle(std::chrono::milliseconds(500));
    // Nor, for a few seconds, while a run-end notification is being sent:
    // the run a close cancels ends with one, and quitting would cut it off.
    // Only with notifications set up is one ever sent. The run is busy until
    // its thread is let go (on_run_thread_finished), which comes after
    // on_all_finished has queued the notification: its thread ends as it
    // says all_finished, so it no longer running did not mean the
    // notification was queued. The seconds count from the end of the busy
    // time, when the notification is sent: a cancel that took longer than
    // them left none (#221's review).
    const bool busy = !live_workers().empty() || worker_ != nullptr || !written;
    if (busy) {
        close_clock_.restart();
    }
    const bool notifying = notifier_ != nullptr && notifier_->pending() > 0 &&
                           close_clock_.elapsed() < notification_close_wait_ms;
    if (busy || notifying) {
        // Held on the status line, where the run would put its counts.
        run_hold_ = RunHold::message;
        run_hold_message_ = busy ? app::tr("Finishing up; the window will close on its own.")
                                 : app::tr("Sending the run's notification; the window will close on its own.");
        show_status(run_hold_message_);
        QTimer::singleShot(200, this, [this] { close_when_idle(); });
        event->ignore();
        return;
    }
    // Remember the window's place and size and the videos, if asked to. The
    // size kept beside the place is the window's when not maximized, which an
    // older version reopens at.
    if (settings_.remember_window_size) {
        settings_.window_geometry = saveGeometry().toBase64().toStdString();
        const QRect normal = normalGeometry().isValid() ? normalGeometry() : geometry();
        settings_.window_width = normal.width();
        settings_.window_height = normal.height();
    }
    if (settings_.remember_window_size || settings_.remember_videos) {
        store_open_videos();
        settings_.save();
    }
    if (restart_on_close_) {
        // Restart now, asked for (offer_restart): the app again, as it was
        // started (the portable launcher's profile is in the environment it
        // inherits), now that this one has saved everything.
        QProcess::startDetached(QCoreApplication::applicationFilePath(), QCoreApplication::arguments().mid(1));
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::offer_restart(const QString& what) {
    if (what.isEmpty()) {  // changed back: no restart needed, and nothing to say so
        export_indicator_->dismiss_note(restart_note_);
        restart_note_.clear();
        return;
    }
    // A setting that shows only once the app starts again (the language, VML
    // Classic): offered at once in a question of the app's own, rather than
    // a line at Settings' foot that went unnoticed. During a run, restarting
    // would stop it: the title bar says so instead, until clicked.
    if (run_active_) {
        restart_note_ = what + " " + app::tr("Restart the app once the run ends.");
        export_indicator_->show_note(restart_note_, ExportIndicator::Note::kept);
        return;
    }
    Message message;
    message.kind = MessageKind::question;
    message.title = app::tr("Restart to apply");
    message.headline = what;
    message.text = settings_.remember_videos
                       ? app::tr("Restart the app now? The videos open now are opened again.")
                       : app::tr("Restart the app now? The videos open now are not opened again ({setting}).",
                                 {{"setting", app::tr("Settings") + " > " + app::tr("General") + " > " +
                                                  app::tr("Reopen the videos that were open when the app was "
                                                          "last closed")}});
    message.buttons = {{app::tr("Restart now"), MessageButton::Role::primary},
                       {app::tr("Later"), MessageButton::Role::plain}};
    message.default_button = 0;
    message.escape_button = 1;
    if (dialogs_.message(message).button == 0 && !run_active_) {
        restart_on_close_ = true;
        close();
        if (!closing_) {
            restart_on_close_ = false;  // the close was refused: no second copy later
        }
        return;
    }
    restart_note_ = what;
    export_indicator_->show_note(what, ExportIndicator::Note::kept);  // Later: kept until clicked
}

void MainWindow::store_open_videos() {
    // The files themselves, in the table's order. The rows made from other
    // rows are not kept: their paths are made up, no file's. Each decodes
    // another row's file (media_path: resolution tests, film grain twins,
    // other displays' rows), or is an opposite-direction row.
    if (!settings_.remember_videos) {
        return;
    }
    settings_.remembered_reference = shown_source_path().toStdString();
    settings_.remembered_tests.clear();
    for (const std::shared_ptr<RowData>& row : rows_) {
        const std::string path = platform::path_text(row->path);
        if (!row->media_path && !row->scale_direction_pinned &&
            std::find(settings_.remembered_tests.begin(), settings_.remembered_tests.end(), path) ==
                settings_.remembered_tests.end()) {
            settings_.remembered_tests.push_back(path);
        }
    }
}

void MainWindow::open_videos_changed() { open_videos_timer_->start(); }

void MainWindow::save_open_videos() {
    // The videos are saved as they change, not only as the window closes,
    // so that they are reopened after the app was ended another way (a
    // crash, the task manager, Windows shutting down). Not while the videos
    // open last time are being reopened, nor while a reference is read: the
    // reference is shown, and stored, only once it is read, and the end of
    // the reading asks again.
    if (!settings_.remember_videos || open_videos_restoring_ || source_reading_ || closing_) {
        return;
    }
    const std::string reference = settings_.remembered_reference;
    const std::vector<std::string> tests = settings_.remembered_tests;
    store_open_videos();
    if (settings_.remembered_reference != reference || settings_.remembered_tests != tests) {
        save_settings();
    }
}

void MainWindow::restore_open_videos() {
    // Looked for on a thread of its own: a path on a network drive that is
    // not there can take many seconds to say so, and the window would not
    // respond meanwhile. A thread let go of, not Qt's pool, which the app
    // waits for as it ends: closed meanwhile, it would stay running until
    // the drive had answered (the Python app's thread is a daemon too).
    if (!settings_.remember_videos) {
        return;
    }
    const QString reference = QString::fromStdString(settings_.remembered_reference);
    QStringList tests;
    for (const std::string& path : settings_.remembered_tests) {
        tests.append(QString::fromStdString(path));
    }
    if (reference.isEmpty() && tests.isEmpty()) {
        return;
    }
    open_videos_lookup_ = std::make_shared<OpenVideosLookup>();
    open_videos_lookup_->window = this;
    open_videos_restoring_ = true;
    std::thread([lookup = open_videos_lookup_, reference, tests] {
        const QString present_reference = !reference.isEmpty() && QFileInfo(reference).isFile() ? reference : QString();
        QStringList present;
        for (const QString& path : tests) {
            if (QFileInfo(path).isFile()) {
                present.append(path);
            }
        }
        // Held while the call is queued, so that the window cannot go in
        // between; one gone before the call comes takes it with it.
        const std::lock_guard lock(lookup->mutex);
        if (MainWindow* window = lookup->window) {
            QMetaObject::invokeMethod(
                window, [window, present_reference, present] { window->reopen_videos(present_reference, present); },
                Qt::QueuedConnection);
        }
    }).detach();
}

void MainWindow::reopen_videos(const QString& reference, const QStringList& tests) {
    // Saved from now on, the videos added meanwhile too (save_open_videos).
    open_videos_restoring_ = false;
    open_videos_changed();
    // Added unless the window has gained videos of its own since, or is
    // closing: their reading would hold the close up.
    if (!rows_.empty() || source_info_ || source_reading_ || run_active_ || closing_) {
        return;
    }
    const std::size_t remembered = settings_.remembered_tests.size();
    if (static_cast<std::size_t>(tests.size()) < remembered) {
        logger.info("Reopening " + std::to_string(tests.size()) + " of the " + std::to_string(remembered) +
                    " test videos open last time; the rest are no longer there");
    }
    if (!reference.isEmpty()) {
        choose_source(path_of(reference));
    } else if (!settings_.remembered_reference.empty()) {
        logger.info("The reference open last time is no longer there");
    }
    std::vector<std::filesystem::path> paths;
    for (const QString& path : tests) {
        paths.push_back(path_of(path));
    }
    add_test_videos(paths);
}

void MainWindow::show_status(const QString& text) { status_label_->setText(text); }

void MainWindow::notify_done(const QString& text) { export_indicator_->show_note(text); }

void MainWindow::notify_info(const QString& text) { export_indicator_->show_note(text, ExportIndicator::Note::info); }

void MainWindow::notify_failed(const QString& text) {
    export_indicator_->show_note(text, ExportIndicator::Note::failed);
}

void MainWindow::save_settings() {
    if (const std::optional<std::string> error = settings_.save()) {
        notify_failed(tr_message(QString::fromStdString(*error)));
    }
}

// ------------------------------------------------------------- construction

void MainWindow::build_ui() {
#ifdef _WIN32
    // VML Modern's own title bar, in the theme's colours, as the window's
    // menu widget: along its top, the full width, above the central widget's
    // margins. Asked for before the window is first shown (TitleBar::prepare).
    if (TitleBar::wanted()) {
        title_bar_ = new TitleBar(this);
        title_bar_->prepare();
        setMenuWidget(title_bar_);
    }
#endif
    auto* central = new QWidget;
    setCentralWidget(central);
    auto* root = new QVBoxLayout(central);

    // The FFmpeg warning and its button, in one row.
    auto* banner_row = new QHBoxLayout;
    ffmpeg_banner_ = new QLabel;
    theme::style(ffmpeg_banner_, "background: {banner_background}; color: {banner_text}; padding: 6px; "
                                 "border: 1px solid {banner_border};");
    ffmpeg_banner_->setWordWrap(true);
    ffmpeg_banner_->setVisible(false);
    locate_ffmpeg_button_ = new QPushButton(app::with_program_names(app::tr("Locate ffmpeg.exe")));
    connect(locate_ffmpeg_button_, &QPushButton::clicked, this, [this] { on_locate_ffmpeg(); });
    locate_ffmpeg_button_->setVisible(false);
    banner_row->addWidget(ffmpeg_banner_, 1);
#ifdef _WIN32
    download_ffmpeg_button_ = new QPushButton(app::tr("Download FFmpeg"));
    connect(download_ffmpeg_button_, &QPushButton::clicked, this, [this] { on_download_ffmpeg(); });
    download_ffmpeg_button_->setVisible(false);
    banner_row->addWidget(download_ffmpeg_button_);
#endif
    banner_row->addWidget(locate_ffmpeg_button_);
    root->addLayout(banner_row);

    // Tabs, not separate windows. Each tab but Video Compare scrolls when the
    // window is too small for it (ScrollingPage).
    tabs_ = new QuietTabWidget;
    root->addWidget(tabs_, 1);
#ifdef VML_NATIVE_PLAYBACK
    // The pointer on Video Compare's tab starts making what its playback
    // first needs (FrameComparePanel::prepare_playback, about 130 ms) in the
    // moment before the click: its first frames then show about 130 ms after
    // it opens rather than about 300 ms.
    tabs_->tabBar()->setMouseTracking(true);
    tabs_->tabBar()->installEventFilter(this);
#endif

    auto* videos_page = new QWidget;
    auto* videos_layout = new QVBoxLayout(videos_page);
    videos_layout->setContentsMargins(0, 0, 0, 0);
    auto* splitter = new QSplitter(Qt::Vertical);
    videos_layout->addWidget(splitter, 1);
    // The videos on top and the options below them, so the video table gets
    // the window's full width.
    splitter->addWidget(build_files_panel());
    // The GPUs here: the Options panel offers its GPU choices for them.
    detected_gpu_vendors_ = detected_gpu_vendors();
    options_panel_ = new OptionsPanel(detected_gpu_vendors_);
    show_metric_route_rows();
    connect(options_panel_, &OptionsPanel::field_edited, this,
            [this](const QString& field) { on_panel_field_edited(field); });
    connect(options_panel_, &OptionsPanel::model_chosen, this, [this](int index) { on_model_changed(index); });
    connect(options_panel_, &OptionsPanel::cvvdp_preset_chosen, this,
            [this](int index) { on_cvvdp_preset_chosen(index); });
    connect(options_panel_, &OptionsPanel::cvvdp_resize_toggled, this,
            [this](bool checked) { on_cvvdp_resize_toggled(checked); });
    connect(options_panel_, &OptionsPanel::cvvdp_add_requested, this, [this] { on_cvvdp_add_preset(); });
    connect(options_panel_, &OptionsPanel::cvvdp_edit_requested, this, [this] { on_cvvdp_edit_display(); });
    connect(options_panel_, &OptionsPanel::cvvdp_delete_requested, this, [this] { on_cvvdp_delete_preset(); });
    connect(options_panel_, &OptionsPanel::cvvdp_other_display_requested, this,
            [this] { on_cvvdp_other_display_requested(); });
    connect(options_panel_, &OptionsPanel::scale_direction_chosen, this,
            [this](int index) { on_scale_direction_combo_changed(index); });
    connect(options_panel_, &OptionsPanel::film_grain_chosen, this, [this](bool with) { on_film_grain_chosen(with); });
    run_cvvdp_dialog_ = [](CvvdpDisplayDialog& dialog) { return dialog.exec(); };
    // Settings' Export folder, read at each export: every save dialog starts there.
    const auto export_dir = [this] { return settings_.export_dir; };
    dialogs_ = standard_dialogs(this, export_dir);
    splitter->addWidget(options_panel_);
    splitter->addWidget(build_run_panel());
    splitter->setSizes({380, 300, 120});
    tabs_->addTab(new ScrollingPage(videos_page), app::tr("Videos"));

    video_colors_ = new VideoColors(this);
    graph_panel_ = new GraphPanel(video_colors_);
    graph_panel_->dialogs = standard_dialogs(graph_panel_, export_dir);
    graph_panel_->set_preferred_metric(settings_.graph_metric);
    graph_panel_->set_video_name_rule(video_name_rule(settings_));
    connect(graph_panel_, &GraphPanel::metric_changed, this, [this](const QString& key) {
        settings_.graph_metric = key.toStdString();  // the metric chosen, shown again next time
        save_settings();
    });
    tabs_->addTab(new ScrollingPage(graph_panel_), app::tr("Metric Graphs"));
    // Worst Frames: a video's worst frames and weak segments (issue #24).
    worst_tab_ = new WorstFramesTab(dialogs_);
    worst_tab_->set_worst_count(settings_.graph_worst_frames);
    worst_tab_->set_segment_profiles(segment_profiles(settings_), segment_profile_in_use(settings_).name);
    connect(worst_tab_, &WorstFramesTab::rules_saved, this, [this](const std::string& profile, const SegmentRules& rules) {
        store_segment_profile(settings_, profile, rules);
        save_settings();
        worst_tab_->set_segment_profiles(segment_profiles(settings_), segment_profile_in_use(settings_).name);
        if (settings_panel_ != nullptr) {
            settings_panel_->show_weak_segment_rules();
        }
    });
    connect(worst_tab_, &WorstFramesTab::count_changed, this, [this](int count) {
        settings_.graph_worst_frames = count;
        save_settings();
    });
    connect(worst_tab_, &WorstFramesTab::open_requested, this,
            [this](const std::string& identity, int frame, const WorstFrameList& list) {
                show_in_video_compare(identity, frame, list);
            });
    tabs_->addTab(new ScrollingPage(worst_tab_), app::tr("Worst Frames"));
    frame_compare_panel_ = new FrameComparePanel([this] { return tools_status_.ffmpeg.path; },
                                                 preview_color_mode_from(settings_.frame_preview_color_mode),
                                                 settings_.compare_decoded_videos);
    connect(frame_compare_panel_, &FrameComparePanel::worst_stop_shown, worst_tab_, &WorstFramesTab::keep_worst_frame);
    connect(frame_compare_panel_, &FrameComparePanel::color_mode_changed, this, [this](const QString& mode) {
        settings_.frame_preview_color_mode = mode.toStdString();  // the colour mode chosen, next time too
        save_settings();
    });
    frame_compare_panel_->set_video_name_rule(video_name_rule(settings_));
    frame_compare_panel_->set_loop_seconds(settings_.compare_loop_before_ms / 1000.0,
                                           settings_.compare_loop_after_ms / 1000.0);
    // Loop as it was left, as Highlight best/worst results is.
    frame_compare_panel_->set_loop_on(settings_.compare_loop);
    connect(frame_compare_panel_, &FrameComparePanel::loop_changed, this, [this](bool on) {
        settings_.compare_loop = on;
        save_settings();
    });
    connect(frame_compare_panel_, &FrameComparePanel::loop_seconds_changed, this, [this](double before, double after) {
        settings_.compare_loop_before_ms = static_cast<int>(std::lround(before * 1000));
        settings_.compare_loop_after_ms = static_cast<int>(std::lround(after * 1000));
        save_settings();
    });
    frame_compare_panel_->export_folder = dialogs_.export_folder;
    frame_compare_panel_->choose_folder = [this] {
        return dialogs_.choose_folder(app::tr("Export the frame pair to"), dialogs_.export_folder());
    };
    // Said on the indicator, as every export is: no window.
    connect(frame_compare_panel_, &FrameComparePanel::still_saved, this,
            [this](const QStringList& files) { report_export({files, {}, {}}); });
    connect(frame_compare_panel_, &FrameComparePanel::still_failed, this, [this](const QString& error) {
        report_export({{}, tr_message(error), tr_message(error)});
    });
    tabs_->addTab(frame_compare_panel_, app::tr("Video Compare"));
    // Its own list of videos: a finished run adds its own (on_job_finished).
    bitrate_panel_ = new BitratePanel([this] { return find_tool("ffprobe", settings_); }, video_colors_);
    bitrate_panel_->dialogs = standard_dialogs(bitrate_panel_, export_dir);
    // Its readings are saved with the saved scores, and found there again.
    bitrate_panel_->store_folder = [this] { return cache_folder(); };
    // The graphs' colours as Settings' Exported graphs say at each write.
    export_queue_ = new ExportQueue({
        .parent = this,
        .bitrate = *bitrate_panel_,
        .style =
            [this] {
                return BundleStyle{video_colors_,
                                   theme::export_chart_colours(settings_.export_graph_theme,
                                                               settings_.export_graph_background,
                                                               settings_.export_graph_text)};
            },
        .submit = [this](const QString& description, std::function<QStringList()> write,
                         bool quiet) { submit_export(description, std::move(write), quiet); },
        .show_saving = [this](const QString& why) { export_indicator_->show_saving(why); },
        .show_status = [this](const QString& text) { export_indicator_->show_note(text, ExportIndicator::Note::kept); },
    });
    // Settings' Exported graphs, read at each export as the folder is.
    const auto export_colours = [this] {
        return theme::export_chart_colours(settings_.export_graph_theme, settings_.export_graph_background,
                                           settings_.export_graph_text);
    };
    graph_panel_->export_colours = export_colours;
    worst_tab_->export_colours = export_colours;
    // Every tab's exports say how they went on the indicator (build_tab_corner),
    // and their save icons save in the export folder in Settings' Quick saves
    // formats, both read at each save.
    const auto report = [this](const ExportOutcome& outcome) { report_export(outcome); };
    const QuickSaves quick_saves = window_quick_saves();
    graph_panel_->export_report = report;
    graph_panel_->quick_saves = quick_saves;
    worst_tab_->export_report = report;
    worst_tab_->quick_saves = quick_saves;
    bitrate_panel_->export_report = report;
    bitrate_panel_->quick_saves = quick_saves;
    // Metric Graphs' statistics menu offers the Videos table's exports.
    graph_panel_->export_items = [this] { return exported_items(settings_); };
    graph_panel_->video_exportable = [this](const std::string& identity) {
        return row_of_graph(identity).has_value();
    };
    connect(graph_panel_, &GraphPanel::video_export_requested, this, [this](const std::string& identity) {
        if (const std::optional<int> row = row_of_graph(identity)) {
            export_this_video(*row);
        }
    });
    connect(graph_panel_, &GraphPanel::video_compare_requested, this,
            [this](const std::string& identity, int frame) { show_in_video_compare(identity, frame, {}); });
    graph_panel_->row_identities = [this] {
        std::vector<std::string> identities;
        for (const std::shared_ptr<RowData>& row_data : rows_) {
            if (row_data->completed_run) {
                identities.push_back(row_data->completed_run->graph_identity);
            }
        }
        return identities;
    };
    // Every row's line back, those removed with the cross too, as Open
    // metric graphs brought them back before the Videos tab lost it.
    connect(graph_panel_, &GraphPanel::lines_changed, this, [this] { repaint_row_marks(); });
    connect(graph_panel_, &GraphPanel::restore_removed_requested, this, [this] {
        std::vector<std::shared_ptr<CompletedRun>> runs;
        for (const std::shared_ptr<RowData>& row_data : rows_) {
            if (row_data->completed_run) {
                runs.push_back(row_data->completed_run);
            }
        }
        open_or_update_graph(runs, false);
    });
    bitrate_panel_->export_colours = export_colours;
    tabs_->addTab(new ScrollingPage(bitrate_panel_), app::tr("Bitrate Viewer"));
    // Its encodes from the runs Metric Graphs lists, brought up to date first.
    bd_rate_panel_ = new BdRatePanel([this] {
        sync_graph();
        return graph_panel_->runs();
    });
    bd_rate_panel_->dialogs = standard_dialogs(bd_rate_panel_, export_dir);
    bd_rate_panel_->export_report = report;
    bd_rate_panel_->quick_saves = quick_saves;
    tabs_->addTab(new ScrollingPage(bd_rate_panel_), app::tr("BD-rate"));
    saved_results_panel_ = new SavedResultsPanel(settings_, dialogs_, *file_writes_);
    saved_results_panel_->quick_formats = [this] { return quick_formats(settings_); };
    // The Videos tab's own check of a saved score, with the settings as a new
    // row has them (metric_cache.h's run_takes): the file its request reads,
    // VMAF's model included, answering it, made where its metric is
    // calculated now. Each comparison's videos read once, until the list is
    // read again (its comparisons deleted, or saved anew).
    const auto infos = std::make_shared<std::map<std::filesystem::path, std::pair<VideoInfo, VideoInfo>>>();
    connect(saved_results_panel_, &SavedResultsPanel::listed, this, [infos] { infos->clear(); });
    saved_results_panel_->videos_would_use =
        [this, infos](
            const SavedComparison& saved, const std::string& key, const SavedScoreChoice& saved_score) {
            auto found = infos->find(saved.folder);
            if (found == infos->end()) {
                const std::optional<std::pair<ComparisonResult, std::string>> context = saved_comparison(saved.folder);
                if (!context) {
                    return false;
                }
                found = infos->emplace(saved.folder, std::pair(context->first.source_info, context->first.distorted_info))
                            .first;
            }
            const auto& [source, distorted] = found->second;
            const VmafOptions options = options_from_settings();
            std::optional<CvvdpPreset> display = cvvdp_preset_named(settings_.cvvdp_default_preset,
                                                                    settings_.cvvdp_presets_json);
            const CvvdpSettings cvvdp = display ? display->settings : automatic_cvvdp_preset(&source).settings;
            try {
                const AnalysisRequest request =
                    analysis_request(options, {key}, automatic_perceptual_backends, cvvdp, source, distorted);
                for (const MetricRequestSpec& spec : request.metrics) {
                    if (spec.key != key) {
                        continue;
                    }
                    const auto choice = automatic_perceptual_backends.find(key);
                    const VshipBuild build = options.vship_build;
                    return saved_score.provenance &&
                           run_takes(saved.folder, spec, saved_score.file, *saved_score.provenance,
                                     choice == automatic_perceptual_backends.end() ? std::nullopt
                                                                                   : std::optional(choice->second),
                                     [build](const std::string&) { return vship_available(build); },
                                     colour_tags(source, distorted));
                }
            } catch (const std::invalid_argument&) {
                // No way to calculate it with these settings: the Videos tab takes none.
            }
            return false;
        };
    saved_results_panel_->save_score_summary = [this](const std::vector<LoadedSavedResult>& results) {
        if (!results.empty()) {
            quick_save_score_summary(saved_score_summary(results), results.front().label);
        }
    };
    saved_results_panel_->save_results = [this](const std::vector<LoadedSavedResult>& results) {
        if (results.empty()) {
            return;
        }
        std::vector<ExportedRun> runs;
        for (const LoadedSavedResult& saved : results) {
            runs.push_back({saved.result, saved.label});
        }
        const std::filesystem::path& reference = results.front().result->source;
        const bool one_reference = std::all_of(results.begin(), results.end(), [&reference](const auto& saved) {
            return same_path(saved.result->source, reference);
        });
        quick_save_results(
            export_document(one_reference ? std::optional(results.front().result->source_info) : std::nullopt, runs,
                            current_machine()),
            results.size() == 1 ? results.front().label : text_of(reference.stem()));
    };
    connect(saved_results_panel_, &SavedResultsPanel::open_requested, this,
            [this](const std::vector<LoadedSavedResult>& results) {
                // On Metric Graphs as saved runs: lines of their own, beside the
                // Videos tab's, and named apart from a row of the same video.
                for (const LoadedSavedResult& saved : results) {
                    graph_panel_->add_run({saved.result, app::tr("{name} (saved)", {{"name", saved.label}}),
                                           "saved:" + platform::path_text(saved.folder)});
                }
                if (!results.empty()) {
                    tabs_->setCurrentIndex(tab_graph);
                }
            });
    connect(saved_results_panel_, &SavedResultsPanel::export_requested, this,
            [this](SavedResultsPanel::Export kind, const std::vector<LoadedSavedResult>& results) {
                export_saved_results(kind, results);
            });
    connect(saved_results_panel_, &SavedResultsPanel::deleted, this, [this] {
        if (settings_panel_ != nullptr) {
            settings_panel_->refresh_status();
        }
    });
    connect(saved_results_panel_, &SavedResultsPanel::columns_changed, this, [this] {
        settings_.saved_results_column_widths = saved_results_panel_->column_widths();
        settings_.saved_results_hidden_columns = saved_results_panel_->hidden_columns();
        save_settings();
    });
    tabs_->addTab(new ScrollingPage(saved_results_panel_), app::tr("Saved Results"));
    settings_page_ = new ScrollingPage(new QWidget);  // the panel put in when first wanted (settings_panel())
    tabs_->addTab(settings_page_, app::tr("Settings"));
    connect(saved_results_panel_, &SavedResultsPanel::scores_settings_requested, this, [this] {
        settings_panel().show_saved_results_section();
        tabs_->setCurrentWidget(settings_page_);
    });
    connect(tabs_, &QTabWidget::currentChanged, this, [this](int index) { on_tab_changed(index); });
    build_tab_corner();
    make_title_buttons();
    // The focus starts on the tabs, so Ctrl+Tab and the arrows switch them at
    // once. Left to Qt, the window's first control took it, VML Modern's
    // title bar progress button, outside the tabs: Ctrl+Tab did nothing
    // until a Tab or a click moved it. Taken when the window is first shown.
    tabs_->tabBar()->setFocus(Qt::OtherFocusReason);
    add_shortcuts();
}

void MainWindow::add_shortcuts() {
    // Window-wide, while the window is the active one (not over a dialog),
    // whatever has the focus: a shortcut presses its button, so it does what
    // a click would, and nothing while the button is grayed.
    for (const Shortcut& shortcut : shortcuts()) {
        auto* action = new QAction(this);
        action->setShortcut(shortcut.keys);
        action->setShortcutContext(Qt::WindowShortcut);
        const Command command = shortcut.command;
        connect(action, &QAction::triggered, this, [this, command] { run_command(command); });
        addAction(action);
    }
    for (int index = 0; index < tabs_->count() && index < 8; ++index) {
        auto* action = new QAction(this);
        action->setShortcut(tab_keys(index));
        action->setShortcutContext(Qt::WindowShortcut);
        connect(action, &QAction::triggered, this, [this, index] { tabs_->setCurrentIndex(index); });
        addAction(action);
    }
    // Delete in the test videos' table removes the selected rows.
    auto* remove = new QAction(distorted_table_);
    remove->setShortcut(remove_keys());
    remove->setShortcutContext(Qt::WidgetShortcut);
    connect(remove, &QAction::triggered, remove_selected_button_, &QPushButton::click);
    distorted_table_->addAction(remove);
    // Each button's key in its tooltip.
    const auto keys_of = [](Command command) {
        for (const Shortcut& shortcut : shortcuts()) {
            if (shortcut.command == command) {
                return shortcut.keys;
            }
        }
        return QKeySequence();
    };
    show_shortcut_in_tip(add_files_button_, keys_of(Command::add_videos));
    show_shortcut_in_tip(source_browse_button_, keys_of(Command::choose_reference));
    show_shortcut_in_tip(load_button_, keys_of(Command::import_run));
    show_shortcut_in_tip(run_button_, keys_of(Command::calculate));
    show_shortcut_in_tip(pause_button_, keys_of(Command::pause));
    show_shortcut_in_tip(cancel_button_, keys_of(Command::cancel));
    show_shortcut_in_tip(tab_export_button_, keys_of(Command::export_all));
    show_shortcut_in_tip(remove_selected_button_, remove_keys());
}

void MainWindow::run_command(Command command) {
    switch (command) {
    case Command::add_videos:
    case Command::choose_reference:
    case Command::import_run:
        // On the Videos tab, where the videos go.
        tabs_->setCurrentIndex(tab_videos);
        (command == Command::add_videos         ? add_files_button_
         : command == Command::choose_reference ? source_browse_button_
                                                : load_button_)
            ->click();
        break;
    case Command::calculate:
        run_button_->click();
        break;
    case Command::pause:
        pause_button_->click();
        break;
    case Command::cancel:
        cancel_button_->click();
        break;
    case Command::export_all:
        tab_export_button_->click();  // its panel opens on the press (PopupPanel::open_on_press)
        break;
    case Command::settings:
        tabs_->setCurrentIndex(tab_settings);
        break;
    case Command::shortcuts:
        show_shortcuts();
        break;
    }
}

void MainWindow::show_shortcuts() {
    if (shortcuts_dialog_ == nullptr) {
        shortcuts_dialog_ = new ShortcutsDialog(this);
    }
    shortcuts_dialog_->show();
    shortcuts_dialog_->raise();
    shortcuts_dialog_->activateWindow();
}

SettingsPanel& MainWindow::settings_panel() {
    if (settings_panel_ != nullptr) {
        return *settings_panel_;
    }
    settings_panel_ = new SettingsPanel(settings_, *file_writes_, [this] { return tools_status_; }, dialogs_);
    settings_panel_->notify = [this](const QString& text, ExportIndicator::Note kind) {
        export_indicator_->show_note(text, kind);
    };
    settings_panel_->report_export = [this](const ExportOutcome& outcome) { export_indicator_->show_outcome(outcome); };
    connect(settings_panel_, &SettingsPanel::restart_needed, this, [this](const QString& what) { offer_restart(what); });
    connect(settings_panel_, &SettingsPanel::defaults_changed, this, [this] { on_default_settings_changed(); });
    connect(settings_panel_, &SettingsPanel::decoded_videos_changed, frame_compare_panel_,
            &FrameComparePanel::set_decoded_videos);
    connect(settings_panel_, &SettingsPanel::ffmpeg_folder_changed, this, [this] { check_ffmpeg(false); });
    connect(settings_panel_, &SettingsPanel::shortcuts_requested, this, [this] { show_shortcuts(); });
    connect(settings_panel_, &SettingsPanel::saved_results_cleared, this, [this] { on_saved_results_cleared(); });
    connect(settings_panel_, &SettingsPanel::video_names_changed, this, [this] {
        graph_panel_->set_video_name_rule(video_name_rule(settings_));
        frame_compare_panel_->set_video_name_rule(video_name_rule(settings_));
        if (tabs_->currentWidget() == worst_tab_) {
            sync_worst_frames();  // else when it is next opened
        }
    });
    connect(settings_panel_, &SettingsPanel::weak_segment_rules_changed, this, [this] {
        worst_tab_->set_segment_profiles(segment_profiles(settings_), segment_profile_in_use(settings_).name);
    });
    connect(settings_panel_, &SettingsPanel::saved_scores_changed, saved_results_panel_,
            &SavedResultsPanel::scores_settings_changed);
    // Built from the settings as they are now; a run under way locks what it
    // uses, as set_run_ui_active does for a panel built already.
    settings_panel_->set_run_active(run_active_);
    settings_page_->setWidget(settings_panel_);  // the placeholder deleted
    // Built after the rest of the window, its controls came last in the Tab
    // order, after the tab bar's corner (Export, the title bar's buttons),
    // where every other tab has its own before them: the corner's go back to
    // the end.
    // (QWidget::setTabOrder moves only controls that take focus.)
    std::vector<QWidget*> corner;
    QWidget* last = nullptr;  // the last control that takes focus, not the corner's
    for (QWidget* widget = nextInFocusChain(); widget != this; widget = widget->nextInFocusChain()) {
        if (widget->focusPolicy() == Qt::NoFocus) {
            continue;
        }
        if (tab_corner_->isAncestorOf(widget)) {
            corner.push_back(widget);
        } else {
            last = widget;
        }
    }
    for (QWidget* widget : corner) {
        if (last != nullptr) {
            QWidget::setTabOrder(last, widget);
        }
        last = widget;
    }
    settings_panel_->setAutoFillBackground(false);  // the tab's pane behind it, as ScrollingPage makes its page
    return *settings_panel_;
}

void MainWindow::show_theme_in_title() {
    title_buttons_->set_theme(QString::fromStdString(settings_.theme), QString::fromStdString(settings_.light_look),
                              QString::fromStdString(settings_.dark_look));
}

void MainWindow::make_title_buttons() {
    if (tabs_ == nullptr) {
        return;  // the window being built: made with its tabs
    }
    const bool modern = theme::current_interface() == theme::Interface::modern;
    if (title_buttons_ != nullptr) {
        title_buttons_->set_modern(modern);  // Modern chosen in Settings: its buttons too
        return;
    }
#ifdef _WIN32
    if (title_bar_ != nullptr) {
        title_buttons_ = title_bar_->buttons();
    }
#endif
    if (title_buttons_ == nullptr) {
        // No bar of the app's own (VML Classic, another system, high
        // contrast, or Modern chosen since the start): at the tab bar's
        // right end, before the export indicator and Export, so that Export
        // is last as it is under Modern's bar (build_tab_corner).
        title_buttons_ = new TitleButtons;
        static_cast<QBoxLayout*>(tab_corner_->layout())->insertWidget(0, title_buttons_);
        tab_corner_->layout()->setAlignment(title_buttons_, Qt::AlignBottom);  // as the corner alone: on the tab pane
        title_buttons_->show();
    }
    title_buttons_->set_modern(modern);
    show_theme_in_title();
    show_progress_badge();
    connect(title_buttons_, &TitleButtons::progress_clicked, this, [this] { tabs_->setCurrentIndex(0); });
    connect(title_buttons_, &TitleButtons::progress_tooltip_wanted, this, [this] { show_progress_badge(); });
    connect(title_buttons_, &TitleButtons::theme_chosen, this, [this](const QString& value) {
        settings_panel().choose_theme(value);
        show_theme_in_title();
    });
    connect(title_buttons_, &TitleButtons::look_chosen, this, [this](bool dark, const QString& value) {
        settings_panel().choose_look(dark, value);
        show_theme_in_title();
    });
    connect(title_buttons_, &TitleButtons::theme_menu_opening, this, [this] { show_theme_in_title(); });
    connect(title_buttons_, &TitleButtons::update_clicked, this, [this] { update_offer_->show_pending(); });
}

bool MainWindow::nativeEvent(const QByteArray& type, void* message, qintptr* result) {
#ifdef _WIN32
    if (title_bar_ != nullptr && title_bar_->native_event(message, result)) {
        return true;
    }
#endif
    return QMainWindow::nativeEvent(type, message, result);
}

void MainWindow::showEvent(QShowEvent* event) {
#ifdef _WIN32
    if (title_bar_ != nullptr) {
        title_bar_->attach();  // the native window is made now
    }
#endif
    QMainWindow::showEvent(event);
}

QWidget* MainWindow::build_files_panel() {
    // Classic: one box, framed and titled. VML Modern: no box, a card for
    // the reference and one for the test videos, the table's buttons a
    // footer along its bottom (theme::classic_box, theme::container: Classic
    // lays out as it did).
    files_box_ = new QGroupBox;
    theme::classic_box(files_box_, app::tr("Videos"));
    auto* files_layout = new QVBoxLayout(files_box_);

    QBoxLayout* reference_layout = nullptr;
    files_layout->addWidget(theme::container("card", false, &reference_layout));
    add_reference_row(reference_layout);
    auto* tests_card = new QWidget;
    tests_card->setObjectName("testsCard");
    theme::set_role(tests_card, "card");
    auto* tests_layout = new QVBoxLayout(tests_card);
    tests_layout->setContentsMargins(0, 0, 0, 0);
    QBoxLayout* table_area = nullptr;
    tests_layout->addWidget(theme::container("padding", false, &table_area), 1);
    // One block, heading row and table, with no spacing between them in
    // Classic, where the metric picker sits on the table's corner. VML
    // Modern's taller button would touch the table's header there: a gap.
    auto* table_block = new QVBoxLayout;
    table_block->setSpacing(0);
    table_block->addLayout(build_table_heading());
    auto* heading_gap = new QWidget;
    heading_gap->setFixedHeight(8);
    theme::modern_only(heading_gap);
    table_block->addWidget(heading_gap);
    build_test_table();
    table_block->addWidget(distorted_table_, 1);
    table_area->addLayout(table_block, 1);
    QBoxLayout* footer = nullptr;
    QWidget* footer_strip = theme::container("footer", true, &footer);
    footer_strip->setObjectName("testsFooter");
    tests_layout->addWidget(footer_strip);
    files_layout->addWidget(tests_card, 1);
    const std::vector<QWidget*> file_buttons = add_file_buttons(footer);

    // Frozen during a run one control at a time. Don't disable the whole box:
    // a disabled QGroupBox disables its children, the table's scroll bar
    // among them, so a long queue could not be scrolled while it runs.
    file_action_widgets_ = {source_browse_button_};
    append(file_action_widgets_, file_buttons);
    file_action_widgets_.push_back(metrics_button_);
    return files_box_;
}

// The reference's row: its path, Browse..., and what it is.
void MainWindow::add_reference_row(QBoxLayout* layout) {
    auto* classic_label = new QLabel(app::tr("Reference video:"));
    theme::classic_only(classic_label);
    layout->addWidget(classic_label);
    auto* source_row = new QHBoxLayout;
    auto* title = new QLabel(app::tr("Reference"));
    theme::set_role(title, "title");
    title->setContentsMargins(0, 0, 10, 0);
    theme::modern_only(title);
    source_row->addWidget(title);
    // A path can be pasted or typed in (Enter, or leaving the box, reads it),
    // and a file dropped on it from a file manager.
    source_edit_ = new QLineEdit;
    source_edit_->setPlaceholderText(app::tr("Drop a video here, paste its path, or click Browse"));
    connect(source_edit_, &QLineEdit::editingFinished, this, [this] { on_source_path_entered(); });
    source_edit_->installEventFilter(this);
    source_browse_button_ = new QPushButton(app::tr("Browse"));
    connect(source_browse_button_, &QPushButton::clicked, this, [this] { on_browse_source(); });
    source_row->addWidget(source_edit_, 1);
    source_row->addWidget(source_browse_button_);
    layout->addLayout(source_row);
    source_info_label_ = new QLabel(app::tr("No reference selected."));
    theme::style(source_info_label_, "color: {muted};");
    layout->addWidget(source_info_label_);
}

// The row above the test videos' table: what to do with it, and the
// metric picker.
QHBoxLayout* MainWindow::build_table_heading() {
    // The metric picker belongs to the table: it sits on the table's
    // top-right corner, in the row that introduces the table.
    metrics_button_ = new QPushButton(app::tr("Add/remove metrics"));
    metrics_button_->setToolTip(
        app::tr("Choose which metrics appear in the test-video table. A hidden metric is not calculated; scores "
                "already saved for it come back when it is shown again. Also available by right-clicking the column "
                "headers."));
    PopupPanel::open_on_press(metrics_button_, [this] {
        show_metrics_panel(metrics_button_->mapToGlobal(metrics_button_->rect().bottomLeft()),
                           metrics_button_->mapToGlobal(metrics_button_->rect().bottomRight()), metrics_button_);
    });
    auto* table_heading = new QHBoxLayout;
    auto* title = new QLabel(app::tr("Test videos"));  // VML Modern's: the card's title
    theme::set_role(title, "title");
    title->setContentsMargins(0, 0, 10, 0);  // its gap to the hint, gone with it in Classic
    theme::modern_only(title);
    table_heading->addWidget(title, 0, Qt::AlignBottom);
    auto* table_hint = new QLabel(
        app::tr("Test videos to compare against the reference. Check rows to calculate; select rows to edit their "
                "settings below. Metric header shortcuts apply to all rows.") +
        " " +
        app::tr("Drop video files here to add them, drag rows to reorder them, and click a column heading to sort."));
    // Wrapped where it does not fit: on one line, in a longer language, it
    // set the window's minimum width.
    table_hint->setWordWrap(true);
    theme::set_role(table_hint, "hint");  // VML Modern: muted beside the card's title
    table_heading->addWidget(table_hint, 1, Qt::AlignBottom);
    // How many of the metrics the table shows (apply_metric_visibility).
    metrics_count_label_ = new QLabel;
    theme::style(metrics_count_label_, "color: {muted};");
    metrics_count_label_->setContentsMargins(10, 0, 4, 0);
    table_heading->addWidget(metrics_count_label_, 0, Qt::AlignBottom | Qt::AlignRight);
    table_heading->addWidget(metrics_button_, 0, Qt::AlignBottom);
    // The card's save icon, at its top right (issue #224): the selected
    // videos' results, else every scored video's.
    videos_save_ = new SaveButton;
    videos_save_->why_not = [this] {
        return export_menu_rows().empty()             ? nothing_scored_text()
               : quick_formats(settings_).empty() ? no_format_text()
                                                    : QString();
    };
    videos_save_->describe = [this] { return quick_save_tooltip(rows_save_text(), quick_formats(settings_)); };
    videos_save_->save = [this] { save_rows_results(); };
    videos_save_->save_as = [this](QMenu& menu) { fill_videos_quick_menu(menu); };
    table_heading->addWidget(videos_save_, 0, Qt::AlignBottom);
    return table_heading;
}

// The test videos' table (distorted_table_), its metric columns' header
// with their tick boxes, and how it behaves.
void MainWindow::build_test_table() {
    std::vector<int> other_columns{col_remove, col_graph, col_check, col_info, col_black_bars, col_scaling, col_bitrate};
    for (const MetricColumn& item : metric_columns()) {
        other_columns.push_back(item.column);
    }
    distorted_table_ = new FillColumnTable(0, column_count, col_path, other_columns);
    distorted_table_->keep_a_row_selected();  // the Options panel always has a row to show
    // Seeded from the same defaults a new row gets, so the headers agree with
    // the Settings tab.
    QMap<int, bool> checkable;
    for (const MetricColumn& item : metric_columns()) {
        if (!metric_definition(item.key).is_plane()) {  // a plane comes with its metric's tick
            checkable.insert(item.column, default_metric_ticked(item.column));
        }
    }
    metric_header_ = new CheckableHeaderView(checkable, distorted_table_);
    distorted_table_->setHorizontalHeader(metric_header_);
    lay_out_metric_columns();
    // A metric's heading dragged to another place moves the metric in every
    // list; the marks, the tick box and File name are under the frozen
    // columns' own heading (freeze_leading_columns), which does not move.
    metric_header_->setSectionsMovable(true);
    connect(metric_header_, &QHeaderView::sectionMoved, this, [this](int column) { on_column_moved(column); });
    connect(&metric_order_signal(), &MetricOrderSignal::changed, this, [this] { lay_out_metric_columns(); });
    connect(metric_header_, &CheckableHeaderView::sectionToggled, this,
            [this](int column, bool checked) { on_metric_column_toggled(column, checked); });
    // The centred headings on two lines where that makes them narrower, so a
    // long name takes less of the row's width (measured in the header's bold,
    // as Qt measures a heading). The checkbox's room on both sides is in the
    // header's size (CheckableHeaderView::sectionSizeFromContents).
    QFont heading_font = metric_header_->font();
    heading_font.setBold(true);
    const QFontMetrics heading_metrics(heading_font);
    const auto two_lines = [&heading_metrics](const QString& text) { return two_line_heading(text, heading_metrics); };
    const auto header_of = [&two_lines](const char* key) {
        return two_lines(qt_text(metric_definition(key).table_header));
    };
    distorted_table_->setHorizontalHeaderLabels({
        "", app::tr("File name"), app::tr("Media info"), two_lines(app::tr("Black bars")), app::tr("Scaling"),
        two_lines(app::tr("Approx. bitrate")), header_of("psnr"), header_of("ssim"), header_of("vmaf"), header_of("xpsnr"),
        header_of("vmaf_neg"), header_of("ssimulacra2"), header_of("butteraugli"), header_of("cvvdp"),
        header_of("vmaf_v1"), header_of("vmaf_hdr"), header_of("psnr_yuv"), header_of("psnr_yuv_mse"),
        header_of("ssim_yuv"), header_of("ms_ssim"), header_of("ms_ssim_yuv"), header_of("psnr_hvs"),
        header_of("cambi"), header_of("ciede2000"), header_of("vmaf_yuv"), header_of("psnr_yuv_y"),
        header_of("psnr_yuv_u"), header_of("psnr_yuv_v"),
        header_of("ssim_yuv_y"), header_of("ssim_yuv_u"), header_of("ssim_yuv_v"), header_of("ms_ssim_yuv_y"),
        header_of("ms_ssim_yuv_u"), header_of("ms_ssim_yuv_v"), header_of("psnr_hvs_y"), header_of("psnr_hvs_u"),
        header_of("psnr_hvs_v"), header_of("vmaf_yuv_y"), header_of("vmaf_yuv_u"), header_of("vmaf_yuv_v"), "", "",
    });
    // A row's remove mark first, at the start of its row, as Metric Graphs'
    // lines have it; frozen with the tick box and File name.
    auto* remove_marks = new RemoveMarkDelegate(distorted_table_);
    remove_marks->tooltip = [this](const QModelIndex& index) {
        const QTableWidgetItem* name = distorted_table_->item(index.row(), col_path);
        return remove_selected_button_ != nullptr && !remove_selected_button_->isEnabled()
                   ? app::tr("Locked while metrics are calculated.")
                   : app::tr("Remove {name} from the list", {{"name", name != nullptr ? name->text() : QString()}});
    };
    remove_marks->off = [this](const QModelIndex&) {
        return remove_selected_button_ != nullptr && !remove_selected_button_->isEnabled();
    };
    distorted_table_->setItemDelegateForColumn(col_remove, remove_marks);
    // Beside it, the row's line sent back to Metric Graphs, where the user
    // removed it there: before, only Show removed videos (every row's) or a
    // run brought it back.
    auto* send_marks = new RemoveMarkDelegate(distorted_table_, RemoveMarkDelegate::Shape::send);
    send_marks->tooltip = [this](const QModelIndex& index) {
        const QTableWidgetItem* item = distorted_table_->item(index.row(), col_path);
        const QString name = item != nullptr ? item->text() : QString();
        const std::shared_ptr<CompletedRun> run =
            index.row() < static_cast<int>(rows_.size()) ? rows_[static_cast<std::size_t>(index.row())]->completed_run
                                                         : nullptr;
        return run == nullptr ? app::tr("{name} has no scores yet to show in Metric Graphs.", {{"name", name}})
               : graph_panel_->removed(run->graph_identity)
                   ? app::tr("Show {name} in Metric Graphs again", {{"name", name}})
                   : app::tr("{name} is already in Metric Graphs.", {{"name", name}});
    };
    send_marks->off = [this](const QModelIndex& index) { return !line_removed(index.row()); };
    distorted_table_->setItemDelegateForColumn(col_graph, send_marks);
    distorted_table_->unselecting_columns = {col_remove, col_graph};  // a press on a mark leaves the selection
    RemoveMarkDelegate::track_hover(distorted_table_);
    connect(distorted_table_, &QAbstractItemView::clicked, this, [this](const QModelIndex& index) {
        if (index.row() >= static_cast<int>(rows_.size())) {
            return;
        }
        if (index.column() == col_remove && remove_selected_button_->isEnabled()) {
            remove_distorted_rows({index.row()});
        } else if (index.column() == col_graph) {
            // Never a second line of one video: one already shown, or none
            // to show, says so in the title bar's indicator.
            const QTableWidgetItem* item = distorted_table_->item(index.row(), col_path);
            const QString name = item != nullptr ? item->text() : QString();
            const std::shared_ptr<CompletedRun>& run = rows_[static_cast<std::size_t>(index.row())]->completed_run;
            if (run == nullptr) {
                export_indicator_->show_note(
                    app::tr("{name} has no scores yet to show in Metric Graphs.", {{"name", name}}),
                    ExportIndicator::Note::info);
            } else if (!line_removed(index.row())) {
                export_indicator_->show_note(app::tr("{name} is already in Metric Graphs.", {{"name", name}}),
                                             ExportIndicator::Note::info);
            } else {
                open_or_update_graph({run}, false);
                export_indicator_->show_note(app::tr("{name} is back in Metric Graphs.", {{"name", name}}));
            }
        }
    });
    // What each metric measures, as Add/remove metrics says (metric_tip): a
    // plane's says whose plane it is. Wrapped here: WrapLongToolTips wraps
    // an item's tooltip by the item under the mouse, which a header has none of.
    for (const MetricColumn& item : metric_columns()) {
        distorted_table_->horizontalHeaderItem(item.column)->setToolTip(wrapped_tool_tip(metric_tip(item.key)));
    }
    distorted_table_->verticalHeader()->setVisible(false);
    // Kept as a column, but not shown: cropping is in the row's options.
    distorted_table_->setColumnHidden(col_black_bars, true);
    metric_header_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(metric_header_, &QWidget::customContextMenuRequested, this,
            [this](const QPoint& position) { show_metrics_panel(metric_header_->mapToGlobal(position)); });
    apply_metric_visibility();

    // A selected row's best and worst scores keep their green and red (Highlight best/worst results).
    distorted_table_->setItemDelegate(new MarkedCellDelegate(distorted_table_));
    distorted_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    distorted_table_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    distorted_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    distorted_table_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(distorted_table_, &QWidget::customContextMenuRequested, this,
            [this](const QPoint& position) { show_table_menu(position); });
    connect(distorted_table_, &QTableWidget::itemSelectionChanged, this, [this] { on_table_selection_changed(); });
    connect(distorted_table_, &QTableWidget::itemChanged, this,
            [this](QTableWidgetItem* item) { on_table_item_changed(item); });
    // Files dropped from a file manager are added, and rows dragged within
    // the table are reordered; a click on a heading (not on a metric's tick
    // box) sorts by that column, and another click reverses the order.
    distorted_table_->setDragEnabled(true);
    distorted_table_->setDragDropMode(QAbstractItemView::DragDrop);
    distorted_table_->setDropIndicatorShown(false);
    connect(distorted_table_, &FillColumnTable::filesDropped, this,
            [this](const QStringList& paths) { on_files_dropped_on_table(paths); });
    connect(distorted_table_, &FillColumnTable::rowsMoved, this,
            [this](const QList<int>& rows, int target) { on_rows_dragged(rows, target); });
    // A quick second click comes as a double-click, which the heading takes
    // for no click (Windows sends no second press): it is a click too. A
    // double-click on a column's edge is sectionHandleDoubleClicked, and on
    // a tick box ticks it again (CheckableHeaderView).
    for (const auto heard : {&QHeaderView::sectionClicked, &QHeaderView::sectionDoubleClicked}) {
        connect(metric_header_, heard, this, [this](int column) { on_header_clicked(column); });
    }

    // Every column can be resized by dragging, File name included, which
    // also fills whatever space is left over (FillColumnTable).
    QHeaderView* header = distorted_table_->horizontalHeader();
    header->setSectionResizeMode(QHeaderView::Interactive);
    // File name and Media info grow with the window, so their headings stay
    // above their content.
    for (const int column : {static_cast<int>(col_path), static_cast<int>(col_info)}) {
        if (QTableWidgetItem* item = distorted_table_->horizontalHeaderItem(column)) {
            item->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        }
    }
    header->setStretchLastSection(false);
    distorted_table_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    fit_table_columns();
    // The two marks, the tick box and File name stay in place while the
    // metrics scroll sideways, so a row's scores are always beside its name.
    distorted_table_->freeze_leading_columns(4);
}

void MainWindow::lay_out_metric_columns() {
    laying_out_columns_ = true;
    int visual = 0;
    const auto place = [this, &visual](int column) {
        if (metric_header_->visualIndex(column) != visual) {
            metric_header_->moveSection(metric_header_->visualIndex(column), visual);
        }
        ++visual;
    };
    for (const int column : {col_remove, col_graph, col_check, col_path, col_info, col_black_bars, col_scaling,
                             col_bitrate}) {
        place(column);
    }
    for (const MetricColumn& item : metric_columns()) {
        place(item.column);
    }
    laying_out_columns_ = false;
}

void MainWindow::on_column_moved(int column) {
    if (laying_out_columns_) {
        return;
    }
    const MetricColumn* moved = metric_column_of(column);
    if (moved == nullptr || metric_definition(moved->key).is_plane()) {
        lay_out_metric_columns();  // back to its place
        return;
    }
    // The metrics as their headings now stand, hidden ones where they were;
    // one dropped among the video's columns comes first.
    std::vector<std::pair<int, std::string>> placed;
    for (const MetricColumn& item : metric_columns()) {
        if (!metric_definition(item.key).is_plane()) {
            placed.emplace_back(metric_header_->visualIndex(item.column), item.key);
        }
    }
    std::sort(placed.begin(), placed.end());
    std::vector<std::string> order;
    for (auto& [visual, key] : placed) {
        order.push_back(std::move(key));
    }
    apply_metric_order(std::move(order));
    lay_out_metric_columns();  // its planes after it, the video's columns back in place
}

void MainWindow::apply_metric_order(std::vector<std::string> order) {
    if (ordered_metrics(order) == ordered_metrics({})) {
        order.clear();  // the registry's: a metric added later then comes in its own place
    }
    if (order == settings_.metric_order) {
        return;
    }
    settings_.metric_order = std::move(order);
    save_settings();
    set_metric_order(settings_.metric_order);  // every list told (metric_order_signal)
}

void MainWindow::fit_table_columns() {
    // Never narrower than the heading needs: an empty table would clip its
    // headings ("Video bitrate" at 65 px showed as "'ideo bitrat"). Measured
    // for hidden columns too: Qt's size hint of a hidden section is 0, so the
    // metrics hidden at first, shown later, kept a guessed width and their
    // headings were cut at both ends.
    std::map<int, int> starting_widths{
        {col_remove, 26}, {col_graph, 26}, {col_check, 28}, {col_info, 150}, {col_black_bars, 78}, {col_scaling, 90}, {col_bitrate, 65},
    };
    for (const MetricColumn& item : metric_columns()) {
        starting_widths[item.column] = 78;
    }
    for (const auto& [column, width] : starting_widths) {
        distorted_table_->setColumnWidth(column, std::max(width, metric_header_->content_size(column).width()));
    }
    // As tall as the tallest heading, hidden ones included: Qt's own height
    // counts only the shown ones, so showing a two-line heading would make
    // the header grow under the user.
    int height = 0;
    for (int column = 0; column < distorted_table_->columnCount(); ++column) {
        height = std::max(height, metric_header_->content_size(column).height());
    }
    metric_header_->setMinimumHeight(height);
}

// The buttons under the table, and Highlight best/worst. Returns the
// buttons a run freezes.
std::vector<QWidget*> MainWindow::add_file_buttons(QBoxLayout* buttons) {
    auto* add_button = new QPushButton(app::tr("Add files"));
    add_files_button_ = add_button;
    connect(add_button, &QPushButton::clicked, this, [this] { on_add_distorted(); });
    auto* add_resample_button = new QPushButton(app::tr("Add resolution test"));
    add_resample_button->setToolTip(
        app::tr("Calculates selected metrics for downscaling the reference to a lower resolution and scaling it back "
                "up -- no separate test file needed."));
    connect(add_resample_button, &QPushButton::clicked, this, [this] { on_add_resample_test(); });
    auto* remove_button = new QPushButton(app::tr("Remove selected"));
    remove_selected_button_ = remove_button;
    theme::set_role(remove_button, "quiet");  // VML Modern: read as text, not a button's frame
    connect(remove_button, &QPushButton::clicked, this, [this] { on_remove_distorted(); });
    remove_all_button_ = new QPushButton(app::tr("Remove all"));
    theme::set_role(remove_all_button_, "quiet");
    connect(remove_all_button_, &QPushButton::clicked, this, [this] { on_remove_all_distorted(); });
    start_over_button_ = new QPushButton(app::tr("Start over"));
    theme::set_role(start_over_button_, "quiet");
    start_over_button_->setToolTip(app::tr("Clears the reference and removes every test video, for a new "
                                           "comparison. Saved results are kept."));
    connect(start_over_button_, &QPushButton::clicked, this, [this] { on_start_over(); });
    // Remove all's and Start over's line when one is armed (confirm_click),
    // else the videos that do not last as long as the reference.
    footer_note_ = new QLabel;
    footer_note_->setObjectName("videosFooterNote");
    footer_note_->setTextFormat(Qt::PlainText);
    theme::set_role(footer_note_, "hint");  // the length note muted; an armed line in the danger colour over it
    footer_note_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);  // never widens the window
    click_twice_ = new ClickTwice(this);
    connect(click_twice_, &ClickTwice::changed, this, [this] { show_footer_note(); });
    // A saved run's file read into the table: a row of its own, as Add files
    // adds one. Frozen during a run with the files: a saved run can bring a
    // reference of its own, which would change what the running videos mean.
    load_button_ = new QPushButton(app::tr("Import run file..."));
    connect(load_button_, &QPushButton::clicked, this, [this] { on_load_saved_run(); });
    buttons->addWidget(add_button);
    buttons->addWidget(add_resample_button);
    buttons->addWidget(load_button_);
    buttons->addWidget(theme::separator(0));  // adding | removing (Modern's line)
    buttons->addWidget(remove_button);
    buttons->addWidget(remove_all_button_);
    buttons->addWidget(start_over_button_);
    buttons->addSpacing(8);
    buttons->addWidget(footer_note_, 1);
    // Any other of the footer's buttons is another decision: an armed one is
    // let go.
    for (QPushButton* other : {add_button, add_resample_button, load_button_, remove_button}) {
        connect(other, &QPushButton::clicked, click_twice_, &ClickTwice::disarm);
    }
    highlight_check_ = new QCheckBox(app::tr("Highlight best/worst results"));
    highlight_check_->setChecked(settings_.highlight_best_worst);
    connect(highlight_check_, &QCheckBox::toggled, this, [this](bool on) { on_highlight_toggled(on); });
    buttons->addWidget(highlight_check_);
    best_worst_timer_ = new QTimer(this);
    best_worst_timer_->setSingleShot(true);
    best_worst_timer_->setInterval(0);
    connect(best_worst_timer_, &QTimer::timeout, this, [this] { apply_best_worst(); });
    connect(distorted_table_->model(), &QAbstractItemModel::rowsRemoved, this, [this] { schedule_best_worst(); });
    // Any change to the table may be a change to what is ticked: a row's
    // tick, a metric's, a row added, removed or moved.
    QAbstractItemModel* model = distorted_table_->model();
    connect(model, &QAbstractItemModel::rowsInserted, this, [this] { check_ticks_soon(); });
    connect(model, &QAbstractItemModel::rowsRemoved, this, [this] { check_ticks_soon(); });
    connect(model, &QAbstractItemModel::rowsMoved, this, [this] { check_ticks_soon(); });
    connect(model, &QAbstractItemModel::dataChanged, this, [this] { check_ticks_soon(); });
    connect(model, &QAbstractItemModel::modelReset, this, [this] { check_ticks_soon(); });
    // The card's save icon saves the selected rows' results: grayed or not
    // as the selection changes.
    connect(distorted_table_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
            [this] { videos_save_->refresh(); });
    return {add_button, add_resample_button, load_button_, remove_button, remove_all_button_, start_over_button_};
}

QWidget* MainWindow::build_run_panel() {
    auto* panel = new QWidget;
    auto* layout = new QVBoxLayout(panel);

    auto* run_row = new QHBoxLayout;
    run_button_ = new QPushButton(app::tr("Calculate metrics"));
    theme::set_role(run_button_, "primary");  // the tab's own action (VML Modern)
    connect(run_button_, &QPushButton::clicked, this, [this] { on_run_clicked(); });
    pause_button_ = new QPushButton(app::tr("Pause"));
    pause_button_->setEnabled(false);
    pause_button_->setCheckable(true);
    connect(pause_button_, &QPushButton::clicked, this, [this] { on_pause_clicked(); });
    cancel_button_ = new QPushButton(app::tr("Cancel"));
    cancel_button_->setEnabled(false);
    connect(cancel_button_, &QPushButton::clicked, this, [this] { on_cancel_clicked(); });
    run_row->addWidget(run_button_);
    run_row->addWidget(pause_button_);
    run_row->addWidget(cancel_button_);
    // Here rather than in Settings, and still editable while a run is going:
    // a long queue is when someone notices their CPU is half idle.
    const unsigned cores = std::max(1u, platform::logical_cores());
    run_row->addSpacing(16);
    // A check box rather than a number: the choice is only one or two.
    parallel_jobs_check_ = new QCheckBox(app::tr("Calculate CPU metrics for 2 videos in parallel"));
    parallel_jobs_check_->setChecked(settings_.parallel_jobs >= max_parallel_jobs);
    parallel_jobs_check_->setToolTip(app::tr(
        "Runs CPU metric work for two videos simultaneously on this {cores}-core machine. libvmaf does not keep a "
        "many-core CPU busy on its own, so a second video largely fills the idle capacity rather than competing for "
        "it. libvmaf threads left on Auto are shared: each video gets half the cores.\n\nCan be changed while a run "
        "is in progress: ticking it starts another video straight away, unticking it lets the running ones finish "
        "first.\n\nOnly affects how fast results arrive, never what they are.",
        {{"cores", cores}}) + "\n\n" +
        // Issue #21: with VMAF on the GPU, PSNR, SSIM and XPSNR share its pass, in the GPU's one lane (plan_tasks).
        app::tr("With VMAF on the GPU, PSNR, SSIM and XPSNR are usually calculated in its pass, and the GPU takes one "
                "video at a time: this box then does not apply to them."));
    connect(parallel_jobs_check_, &QCheckBox::toggled, this, [this](bool on) { on_parallel_jobs_changed(on); });
    run_row->addWidget(parallel_jobs_check_);
    run_row->addStretch(1);
    layout->addLayout(run_row);

    status_label_ = new QLabel;
    layout->addWidget(status_label_);
    // One line of text per running video, built once and hidden: lines
    // appearing and disappearing mid-run would shift everything below them.
    for (int i = 0; i < max_videos_in_flight; ++i) {
        auto* line = new WrappingLabel;
        theme::style(line, "color: {run_line};");
        line->setVisible(false);
        layout->addWidget(line);
        job_lines_.add_label(line);
    }
    layout->addStretch(1);
    return panel;
}

// ------------------------------------------------------------- the settings

VmafOptions MainWindow::options_from_settings() const {
    // Only a starting point: each row's own settings are edited in the
    // Options panel.
    VmafOptions options;
    options.gpu_decode = settings_.default_gpu_decode;
    options.vmaf_on_gpu = settings_.default_vmaf_on_gpu;
    options.cambi_on_gpu = settings_.default_cambi_on_gpu;
    options.ciede2000_on_gpu = settings_.default_ciede2000_on_gpu;
    options.ms_ssim_on_gpu = settings_.default_ms_ssim_on_gpu;
    options.vship_build = default_vship_build(settings_.default_vship_build);
    settings_.take_default_metrics(options);
    return options;
}

std::set<std::string> MainWindow::extra_metrics_from_settings() const {
    std::set<std::string> keys;
    for (const char* key : {"ssimulacra2", "butteraugli", "cvvdp"}) {
        if (*settings_.default_compute(key)) {
            keys.insert(key);
        }
    }
    return keys;
}

void MainWindow::on_default_settings_changed() {
    use_crop_store();  // the saved results' folder may have changed
    default_options_ = options_from_settings();
    default_extra_metric_keys_ = extra_metrics_from_settings();
    // Unlike the other defaults, film grain's is for every video without a
    // choice of its own, not only new ones: until then it is to be asked.
    // During a run (Settings stays open), the rows and the panel are left as
    // they are until it ends (on_all_finished).
    defaults_changed_in_run_ = run_active_;
    if (!run_active_) {
        start_cache_lookup(apply_film_grain_default());
        show_panel_film_grain();
        if (panel_target_rows_.empty()) {
            write_panel_options(default_options_);  // the heading's ticks are what a new video starts with
        }
    }
}

// ------------------------------------------------------------ the update check

void MainWindow::fit_to_style() {
    fit_table_columns();
    bd_rate_panel_->fit_to_style();
}

void MainWindow::check_for_updates() { update_offer_->check(); }

bool MainWindow::cvvdp_default_automatic() const {
    return !cvvdp_preset_named(settings_.cvvdp_default_preset, settings_.cvvdp_presets_json);
}

CvvdpPreset MainWindow::new_row_cvvdp_preset() const {
    if (std::optional<CvvdpPreset> chosen =
            cvvdp_preset_named(settings_.cvvdp_default_preset, settings_.cvvdp_presets_json)) {
        return *chosen;
    }
    return automatic_cvvdp_preset(source_info_ ? &*source_info_ : nullptr);
}

void MainWindow::follow_reference_cvvdp() {
    // The display only: scaling to fill it is each row's own choice. The
    // preset's name too, so that a user's preset with the same display does
    // not stand for it (in the score's tooltip and the exported results).
    const CvvdpPreset& preset = automatic_cvvdp_preset(source_info_ ? &*source_info_ : nullptr);
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (row_data->cvvdp_automatic) {
            row_data->cvvdp.display = preset.settings.display;
            row_data->cvvdp_preset = preset.name;
        }
    }
    // The Options panel shows the display: a selected row's may have changed.
    if (!panel_target_rows_.empty()) {
        write_panel_options(rows_[static_cast<std::size_t>(panel_target_rows_.front())]->options);
    }
}

// ------------------------------------------------------- FFmpeg's presence

bool MainWindow::check_ffmpeg(bool prompt) {
    // Both tools must run, and ffmpeg be new enough. At the start (prompt),
    // a file picker to point at ffmpeg.exe is offered rather than only a
    // banner.
    const ToolsStatus status = check_tools(settings_);
    tools_status_ = status;
    // A remuxed or moved reference, and a remuxed test video, keep their
    // saved scores (file_fingerprint.h).
    set_video_fingerprinter(fingerprinter_using(status.ffmpeg.path, status.ffprobe.path));
    use_stream_fingerprinter(status.ffmpeg.path, status.ffprobe.path);
    if (prompt) {
        log_tools(status, detected_gpu_vendors_);
    }
    if (status.ok()) {
        ffmpeg_banner_->setVisible(false);
        locate_ffmpeg_button_->setVisible(false);
        if (download_ffmpeg_button_ != nullptr) {
            download_ffmpeg_button_->setVisible(false);
        }
        return true;
    }
    const QStringList problems = tool_problems(status);
#ifdef __APPLE__
    // The formula with zscale, which plain ffmpeg lacks (missing_zscale_text).
    const QString install = app::tr("  Install it with Homebrew: brew install ffmpeg-full.");
#else
    const QString install;
#endif
#ifdef _WIN32
    const QString click = app::tr("  Click \"Download FFmpeg\" to get it, or \"Locate ffmpeg.exe\" to point at it.");
    download_ffmpeg_button_->setVisible(true);
#else
    const QString click = app::with_program_names(app::tr("  Click \"Locate ffmpeg.exe\" to point at it."));
#endif
    ffmpeg_banner_->setText(problems.join("  ") + install + click);
    ffmpeg_banner_->setVisible(true);
    locate_ffmpeg_button_->setVisible(true);
    // A run with no screen (the tests) has nobody to answer a dialog.
    if (!prompt || qEnvironmentVariable("QT_QPA_PLATFORM") == "offscreen") {
        return false;
    }
    const QString found_where = app::tr("\n\nffmpeg was looked for at:\n  {path}", {{"path", status.ffmpeg.path}}) +
                                app::tr("\nffprobe at:\n  {path}", {{"path", status.ffprobe.path}});
    Message message;
    message.kind = MessageKind::warning;
    message.title = app::tr("ffmpeg not usable");
#ifdef _WIN32
    message.text = problems.join("\n") + found_where +
                   app::tr("\n\nThe app can download FFmpeg for you (about 200 MB, from the FFmpeg-Builds "
                           "project on GitHub), or you can point it at an ffmpeg.exe you already have.");
    message.buttons = {{app::tr("Download FFmpeg"), MessageButton::Role::primary},
                       {app::tr("Locate ffmpeg.exe"), MessageButton::Role::plain},
                       {app::tr("Not now"), MessageButton::Role::plain}};
    const int answer = dialogs_.message(message).button;
    if (answer == 0) {
        on_download_ffmpeg();
    } else if (answer == 1) {
        return on_locate_ffmpeg();
    }
#else
    message.text = problems.join("\n") + install + found_where +
                   app::with_program_names(app::tr("\n\nWould you like to point the app at ffmpeg.exe now?"));
    message.buttons = {{app::tr("Yes"), MessageButton::Role::primary}, {app::tr("No"), MessageButton::Role::plain}};
    if (dialogs_.message(message).button == 0) {
        return on_locate_ffmpeg();
    }
#endif
    return false;
}

void MainWindow::on_download_ffmpeg() {
#ifdef _WIN32
    if (ffmpeg_download_ != nullptr) {
        return;  // one is running
    }
    ffmpeg_download_ = new FfmpegDownload(downloaded_ffmpeg_folder(), this);
    const auto finished = [this] {
        ffmpeg_download_->deleteLater();
        ffmpeg_download_ = nullptr;
    };
    connect(ffmpeg_download_, &FfmpegDownload::downloaded, this, [this, finished](const QString& folder) {
        finished();
        on_ffmpeg_downloaded(path_of(folder));
    });
    connect(ffmpeg_download_, &FfmpegDownload::failed, this, [this, finished](const QString& message) {
        finished();
        dialogs_.alert(app::tr("FFmpeg download failed"), message);
    });
    connect(ffmpeg_download_, &FfmpegDownload::cancelled, this, [this, finished] {
        finished();
        notify_info(app::tr("FFmpeg download cancelled"));
    });
    ffmpeg_download_->start();
#endif
}

void MainWindow::on_ffmpeg_downloaded(const std::filesystem::path& folder) {
    // Checked before it is kept: a download that does not run leaves the
    // setting as it was.
    Settings trial = settings_;
    trial.ffmpeg_dir = platform::path_text(folder);
    const ToolsStatus status = check_tools(trial);
    if (!status.ok()) {
        const QStringList problems = tool_problems(status);
        dialogs_.warn(app::tr("Still not usable"), problems.join("\n"));
        // Of no use, and the app's own: not left for the next start.
        std::error_code error;
        std::filesystem::remove(folder / "ffmpeg.exe", error);
        std::filesystem::remove(folder / "ffprobe.exe", error);
        return;
    }
    settings_.ffmpeg_dir = trial.ffmpeg_dir;
    if (settings_panel_ != nullptr) {
        settings_panel_->show_ffmpeg_folder(QString::fromStdString(settings_.ffmpeg_dir));
    }
    tools_status_ = status;
    // A remuxed or moved reference, and a remuxed test video, keep their
    // saved scores (file_fingerprint.h).
    set_video_fingerprinter(fingerprinter_using(status.ffmpeg.path, status.ffprobe.path));
    use_stream_fingerprinter(status.ffmpeg.path, status.ffprobe.path);
    log_tools(status, detected_gpu_vendors_);
    ffmpeg_found(status, folder);
    if (const std::optional<std::string> problem = settings_.save()) {
        show_status(tr_message(QString::fromStdString(*problem)));  // used until the app closes
    }
}

void MainWindow::ffmpeg_found(const ToolsStatus& status, const std::filesystem::path& directory) {
    ffmpeg_banner_->setVisible(false);
    locate_ffmpeg_button_->setVisible(false);
    if (download_ffmpeg_button_ != nullptr) {
        download_ffmpeg_button_->setVisible(false);
    }
    notify_done(app::tr("Using ffmpeg {version} from {directory}",
                        {{"version", format_version(status.ffmpeg.version)}, {"directory", text_of(directory)}}));
}

bool MainWindow::on_locate_ffmpeg() {
    // Asks for ffmpeg.exe itself, not its folder, which a file picker makes
    // clearer; then looks for ffprobe beside it.
#ifdef _WIN32
    const QString ffmpeg_exe = "ffmpeg.exe";
    const QString ffprobe_exe = "ffprobe.exe";
#else
    const QString ffmpeg_exe = "ffmpeg";
    const QString ffprobe_exe = "ffprobe";
#endif
    const QString chosen = dialogs_.open_file_at(
        app::tr("Select {ffmpeg_exe}", {{"ffmpeg_exe", ffmpeg_exe}}), QString(),
        app::tr("{ffmpeg_exe} ({ffmpeg_exe});;All files (*)", {{"ffmpeg_exe", ffmpeg_exe}}));
    if (chosen.isEmpty()) {
        return false;
    }
    const std::filesystem::path directory = path_of(chosen).parent_path();
    std::error_code error;
    if (!std::filesystem::is_regular_file(directory / platform::path_from_text(ffprobe_exe.toStdString()), error)) {
        dialogs_.inform(app::tr("ffprobe needed too"),
                        app::tr("{ffprobe_exe} wasn't found next to {ffmpeg_exe}.\n\nThe app needs both. Please select "
                                "{ffprobe_exe} as well (it normally ships in the same folder).",
                                {{"ffprobe_exe", ffprobe_exe}, {"ffmpeg_exe", ffmpeg_exe}}));
        const QString probe_chosen = dialogs_.open_file_at(
            app::tr("Select {ffprobe_exe}", {{"ffprobe_exe", ffprobe_exe}}), text_of(directory),
            app::tr("{ffprobe_exe} ({ffprobe_exe});;All files (*)", {{"ffprobe_exe", ffprobe_exe}}));
        if (probe_chosen.isEmpty()) {
            return false;
        }
        if (!same_path(path_of(probe_chosen).parent_path(), directory)) {
            dialogs_.warn(app::tr("Different folders"),
                          app::tr("{ffmpeg_exe} and {ffprobe_exe} need to be in the same folder for the app to "
                                  "find both.",
                                  {{"ffmpeg_exe", ffmpeg_exe}, {"ffprobe_exe", ffprobe_exe}}));
            return false;
        }
    }
    // Kept as the Settings tab's folder: the one place it is kept.
    settings_.ffmpeg_dir = platform::path_text(directory);
    if (settings_panel_ != nullptr) {
        settings_panel_->show_ffmpeg_folder(QString::fromStdString(settings_.ffmpeg_dir));
    }
    if (const std::optional<std::string> problem = settings_.save()) {
        show_status(tr_message(QString::fromStdString(*problem)));
        return false;
    }
    const ToolsStatus status = check_tools(settings_);
    tools_status_ = status;
    // A remuxed or moved reference, and a remuxed test video, keep their
    // saved scores (file_fingerprint.h).
    set_video_fingerprinter(fingerprinter_using(status.ffmpeg.path, status.ffprobe.path));
    use_stream_fingerprinter(status.ffmpeg.path, status.ffprobe.path);
    if (!status.ok()) {
        const QStringList problems = tool_problems(status);
        dialogs_.warn(app::tr("Still not usable"), problems.join("\n"));
        check_ffmpeg(false);
        return false;
    }
    ffmpeg_found(status, directory);
    return true;
}

}  // namespace vml::app
