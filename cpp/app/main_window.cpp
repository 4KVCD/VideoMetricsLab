#include "main_window.h"

#include <QAbstractItemView>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkProxyFactory>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPushButton>
#include <QSplitter>
#include <QStyleHints>
#include <QTableWidgetItem>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <chrono>
#include <thread>
#include <tuple>

#include "bd_rate_panel.h"
#include "bitrate_panel.h"
#include "cvvdp_display_dialog.h"
#include "file_writes.h"
#include "frame_compare_panel.h"
#include "graph_panel.h"
#include "i18n.h"
#include "options_panel.h"
#include "run_worker.h"
#include "series_colors.h"
#include "settings_panel.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/ffmpeg_tools.h"
#include "vml/hardware_decoding.h"
#include "vml/metrics.h"
#include "vml/python_text.h"
#include "vml/platform/system.h"
#include "vml/score_identity.h"
#include "vml/vship.h"
#include "widgets.h"
#include "window_helpers.h"

// The window's text is app::tr's (i18n.h), always written so: Q_OBJECT gives
// the window a tr() of Qt's own, which inside its functions would hide the
// app's and show the English.
namespace vml::app {

namespace {

const app_log::Logger logger("vml.main_window");

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

const std::vector<MetricColumn>& metric_columns() {
    // VMAF NEG beside VMAF v0.6.1, whose variant it is; VMAF v1 after them.
    static const std::vector<MetricColumn> columns{
        {col_vmaf, "vmaf"},       {col_vmaf_neg, "vmaf_neg"},       {col_vmaf_v1, "vmaf_v1"},
        {col_psnr, "psnr"},       {col_ssim, "ssim"},               {col_xpsnr, "xpsnr"},
        {col_ssimulacra2, "ssimulacra2"}, {col_butteraugli, "butteraugli"}, {col_cvvdp, "cvvdp"},
    };
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
    logger.info(std::string("Settings: parallel CPU metrics ") + (settings_.parallel_jobs > 1 ? "on" : "off") +
                ", GPU decode " + (settings_.default_gpu_decode ? "on" : "off") + ", VVC decoded on the " +
                (settings_.vvc_decode_on_gpu ? "GPU" : "CPU") + ", VMAF v0.6.1 and NEG on " +
                (settings_.default_vmaf_on_gpu ? "GPU" : "CPU") + ", GPU metrics " +
                (settings_.gpu_metrics_together ? "in one pass" : "one at a time") + " with " +
                std::string(vship_build_label(default_vship_build(settings_.default_vship_build))) +
                ", saved results " + (settings_.use_cache ? "reused" : "not reused") + " (" +
                (settings_.cache_dir.empty() ? std::string("default folder") : settings_.cache_dir) + ")");
    if (settings_.remember_window_size) {
        resize(settings_.window_width, settings_.window_height);
    } else {
        resize(1280, 800);
    }
    // Each video's own settings; the Options panel shows and edits the
    // selected rows' settings, and there is no one global setting.
    default_options_ = options_from_settings();
    default_extra_metric_keys_ = extra_metrics_from_settings();
    for (const std::string& key : settings_.hidden_metrics) {
        if (std::any_of(metric_columns().begin(), metric_columns().end(),
                        [&key](const MetricColumn& item) { return key == item.key; })) {
            hidden_metrics_.insert(key);
        }
    }
    if (hidden_metrics_.size() >= metric_columns().size()) {
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
    connect(file_writes_, &FileWrites::write_failed, this, [this](const QString& description, const QString& error) {
        show_status(app::tr("Could not {description}: {error}", {{"description", description}, {"error", error}}));
    });
    // Saving runs can be asked for again once the saves asked for are written.
    connect(file_writes_, &FileWrites::became_idle, this, [this] {
        if (file_writes_->idle()) {
            save_button_->setEnabled(true);
        }
    });

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
    // Don't close while background work still writes into the widgets: it
    // would crash the app and leave the process running. The work is told
    // to stop (a run ends the programs it started), and the window closes
    // itself once it has.
    if (!closing_) {
        closing_ = true;
        if (gpu_check_watcher_ != nullptr) {
            stop_waiting_for_gpu_checks();  // a run asked for is not started now
        }
        if (run_waits_for_saved_scores_) {
            stop_waiting_for_saved_scores();
        }
        frame_compare_panel_->cancel();
        bitrate_panel_->cancel();
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
    if (!live_workers().empty() || !written) {
        // Held on the status line, where the run would put its counts.
        run_hold_ = RunHold::message;
        run_hold_message_ = app::tr("Finishing up; the window will close on its own.");
        show_status(run_hold_message_);
        QTimer::singleShot(200, this, [this] { close_when_idle(); });
        event->ignore();
        return;
    }
    // Remember the window size and the videos, if asked to.
    if (settings_.remember_window_size) {
        settings_.window_width = width();
        settings_.window_height = height();
    }
    if (settings_.remember_window_size || settings_.remember_videos) {
        store_open_videos();
        settings_.save();
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::store_open_videos() {
    // The files themselves, in the table's order. Resolution tests and
    // opposite-direction rows are made from other rows and are not kept.
    if (!settings_.remember_videos) {
        return;
    }
    settings_.remembered_reference = shown_source_path().toStdString();
    settings_.remembered_tests.clear();
    for (const std::shared_ptr<RowData>& row : rows_) {
        const std::string path = platform::path_text(row->path);
        if (!row->scale_direction_pinned && !row->options.resample_test &&
            std::find(settings_.remembered_tests.begin(), settings_.remembered_tests.end(), path) ==
                settings_.remembered_tests.end()) {
            settings_.remembered_tests.push_back(path);
        }
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

void MainWindow::save_settings() {
    if (const std::optional<std::string> error = settings_.save()) {
        show_status(tr_message(QString::fromStdString(*error)));
    }
}

// ------------------------------------------------------------- construction

void MainWindow::build_ui() {
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
    locate_ffmpeg_button_ = new QPushButton(app::with_program_names(app::tr("Locate ffmpeg.exe...")));
    connect(locate_ffmpeg_button_, &QPushButton::clicked, this, [this] { on_locate_ffmpeg(); });
    locate_ffmpeg_button_->setVisible(false);
    banner_row->addWidget(ffmpeg_banner_, 1);
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
    connect(options_panel_, &OptionsPanel::scale_direction_chosen, this,
            [this](int index) { on_scale_direction_combo_changed(index); });
    connect(options_panel_, &OptionsPanel::film_grain_chosen, this, [this](bool with) { on_film_grain_chosen(with); });
    run_cvvdp_dialog_ = [](CvvdpDisplayDialog& dialog) { return dialog.exec(); };
    dialogs_ = standard_dialogs(this);
    splitter->addWidget(options_panel_);
    splitter->addWidget(build_run_panel());
    splitter->setSizes({380, 300, 120});
    tabs_->addTab(new ScrollingPage(videos_page), app::tr("Videos"));

    video_colors_ = new VideoColors(this);
    graph_panel_ = new GraphPanel(video_colors_);
    graph_panel_->set_preferred_metric(settings_.graph_metric);
    connect(graph_panel_, &GraphPanel::metric_changed, this, [this](const QString& key) {
        settings_.graph_metric = key.toStdString();  // the metric chosen, shown again next time
        save_settings();
    });
    connect(graph_panel_, &GraphPanel::results_export_requested, this,
            [this](ResultsFormat format) { export_results(format); });
    tabs_->addTab(new ScrollingPage(graph_panel_), app::tr("Metric Graphs"));
    frame_compare_panel_ = new FrameComparePanel([this] { return tools_status_.ffmpeg.path; },
                                                 preview_color_mode_from(settings_.frame_preview_color_mode),
                                                 settings_.compare_decoded_videos);
    connect(frame_compare_panel_, &FrameComparePanel::color_mode_changed, this, [this](const QString& mode) {
        settings_.frame_preview_color_mode = mode.toStdString();  // the colour mode chosen, next time too
        save_settings();
    });
    tabs_->addTab(frame_compare_panel_, app::tr("Video Compare"));
    // Its own list of videos: a finished run adds its own (on_job_finished).
    bitrate_panel_ = new BitratePanel([this] { return find_tool("ffprobe", settings_); }, video_colors_);
    tabs_->addTab(new ScrollingPage(bitrate_panel_), app::tr("Bitrate Viewer"));
    // Its encodes from the runs Metric Graphs lists, brought up to date first.
    bd_rate_panel_ = new BdRatePanel([this] {
        sync_graph();
        return graph_panel_->runs();
    });
    tabs_->addTab(new ScrollingPage(bd_rate_panel_), app::tr("BD-rate"));
    settings_panel_ = new SettingsPanel(settings_, *file_writes_, [this] { return tools_status_; }, dialogs_);
    connect(settings_panel_, &SettingsPanel::defaults_changed, this, [this] { on_default_settings_changed(); });
    connect(settings_panel_, &SettingsPanel::decoded_videos_changed, frame_compare_panel_,
            &FrameComparePanel::set_decoded_videos);
    connect(settings_panel_, &SettingsPanel::ffmpeg_folder_changed, this, [this] { check_ffmpeg(false); });
    connect(settings_panel_, &SettingsPanel::saved_results_cleared, this, [this] { on_saved_results_cleared(); });
    tabs_->addTab(new ScrollingPage(settings_panel_), app::tr("Settings"));
    connect(tabs_, &QTabWidget::currentChanged, this, [this](int index) { on_tab_changed(index); });
}

QWidget* MainWindow::build_files_panel() {
    files_box_ = new QGroupBox(app::tr("Videos"));
    auto* files_layout = new QVBoxLayout(files_box_);

    add_reference_row(files_layout);
    // One block, heading row and table, with no spacing between them.
    auto* table_block = new QVBoxLayout;
    table_block->setSpacing(0);
    table_block->addLayout(build_table_heading());
    build_test_table();
    table_block->addWidget(distorted_table_, 1);
    files_layout->addLayout(table_block, 1);
    const std::vector<QWidget*> file_buttons = add_file_buttons(files_layout);

    // Frozen during a run one control at a time. Don't disable the whole box:
    // a disabled QGroupBox disables its children, the table's scroll bar
    // among them, so a long queue could not be scrolled while it runs.
    file_action_widgets_ = {source_browse_button_};
    file_action_widgets_.insert(file_action_widgets_.end(), file_buttons.begin(), file_buttons.end());
    file_action_widgets_.push_back(metrics_button_);
    return files_box_;
}

// The reference's row: its path, Browse..., and what it is.
void MainWindow::add_reference_row(QVBoxLayout* layout) {
    layout->addWidget(new QLabel(app::tr("Reference video:")));
    auto* source_row = new QHBoxLayout;
    // A path can be pasted or typed in (Enter, or leaving the box, reads it),
    // and a file dropped on it from a file manager.
    source_edit_ = new QLineEdit;
    source_edit_->setPlaceholderText(app::tr("Drop a video here, paste its path, or click Browse..."));
    connect(source_edit_, &QLineEdit::editingFinished, this, [this] { on_source_path_entered(); });
    source_edit_->installEventFilter(this);
    source_browse_button_ = new QPushButton(app::tr("Browse..."));
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
    connect(metrics_button_, &QPushButton::clicked, this,
            [this] { show_metrics_menu(metrics_button_->mapToGlobal(metrics_button_->rect().bottomLeft())); });
    auto* table_heading = new QHBoxLayout;
    auto* table_hint = new QLabel(
        app::tr("Test videos to compare against the reference. Check rows to calculate; select rows to edit their "
                "settings below. Metric header shortcuts apply to all rows.") +
        " " +
        app::tr("Drop video files here to add them, drag rows to reorder them, and click a column heading to sort."));
    // Wrapped where it does not fit: on one line, in a longer language, it
    // set the window's minimum width.
    table_hint->setWordWrap(true);
    table_heading->addWidget(table_hint, 1, Qt::AlignBottom);
    table_heading->addWidget(metrics_button_, 0, Qt::AlignBottom);
    return table_heading;
}

// The test videos' table (distorted_table_), its metric columns' header
// with their tick boxes, and how it behaves.
void MainWindow::build_test_table() {
    std::vector<int> other_columns{col_check, col_info, col_black_bars, col_scaling, col_bitrate};
    for (const MetricColumn& item : metric_columns()) {
        other_columns.push_back(item.column);
    }
    distorted_table_ = new FillColumnTable(0, column_count, col_path, other_columns);
    // Seeded from the same defaults a new row gets, so the headers agree with
    // the Settings tab.
    QMap<int, bool> checkable;
    for (const MetricColumn& item : metric_columns()) {
        checkable.insert(item.column, default_metric_ticked(item.column));
    }
    metric_header_ = new CheckableHeaderView(checkable, distorted_table_);
    distorted_table_->setHorizontalHeader(metric_header_);
    int visual = 6;
    for (const MetricColumn& item : metric_columns()) {
        metric_header_->moveSection(metric_header_->visualIndex(item.column), visual++);
    }
    connect(metric_header_, &CheckableHeaderView::sectionToggled, this,
            [this](int column, bool checked) { on_metric_column_toggled(column, checked); });
    const auto header_of = [](const char* key) {
        return QStringLiteral("   ") + qt_text(metric_definition(key).table_header);
    };
    distorted_table_->setHorizontalHeaderLabels({
        "", app::tr("File name"), app::tr("Media info"), app::tr("Black bars"), app::tr("Scaling"),
        app::tr("Video bitrate"), header_of("psnr"), header_of("ssim"), header_of("vmaf"), header_of("xpsnr"),
        header_of("vmaf_neg"), header_of("ssimulacra2"), header_of("butteraugli"), header_of("cvvdp"),
        header_of("vmaf_v1"),
    });
    distorted_table_->verticalHeader()->setVisible(false);
    // Kept as a column, but not shown: cropping is in the row's options.
    distorted_table_->setColumnHidden(col_black_bars, true);
    metric_header_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(metric_header_, &QWidget::customContextMenuRequested, this,
            [this](const QPoint& position) { show_metrics_menu(metric_header_->mapToGlobal(position)); });
    apply_metric_visibility();

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
    connect(metric_header_, &QHeaderView::sectionClicked, this, [this](int column) { on_header_clicked(column); });

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
    // Never narrower than the heading needs: an empty table would clip its
    // headings ("Video bitrate" at 65 px showed as "'ideo bitrat").
    std::map<int, int> starting_widths{
        {col_check, 28}, {col_info, 150}, {col_black_bars, 78}, {col_scaling, 90}, {col_bitrate, 65},
    };
    for (const MetricColumn& item : metric_columns()) {
        const bool wide = item.column == col_psnr || item.column == col_xpsnr || item.column == col_ssimulacra2 ||
                          item.column == col_butteraugli;
        starting_widths[item.column] = wide ? 110 : 78;
    }
    for (const auto& [column, width] : starting_widths) {
        distorted_table_->setColumnWidth(column, std::max(width, header->sectionSizeHint(column)));
    }
}

// The buttons under the table, and Highlight best/worst. Returns the
// buttons a run freezes.
std::vector<QWidget*> MainWindow::add_file_buttons(QVBoxLayout* layout) {
    auto* buttons = new QHBoxLayout;
    auto* add_button = new QPushButton(app::tr("Add files..."));
    connect(add_button, &QPushButton::clicked, this, [this] { on_add_distorted(); });
    auto* add_resample_button = new QPushButton(app::tr("Add resolution test..."));
    add_resample_button->setToolTip(
        app::tr("Calculates selected metrics for downscaling the reference to a lower resolution and scaling it back "
                "up -- no separate test file needed."));
    connect(add_resample_button, &QPushButton::clicked, this, [this] { on_add_resample_test(); });
    auto* remove_button = new QPushButton(app::tr("Remove selected"));
    connect(remove_button, &QPushButton::clicked, this, [this] { on_remove_distorted(); });
    remove_all_button_ = new QPushButton(app::tr("Remove all"));
    connect(remove_all_button_, &QPushButton::clicked, this, [this] { on_remove_all_distorted(); });
    buttons->addWidget(add_button);
    buttons->addWidget(add_resample_button);
    buttons->addWidget(remove_button);
    buttons->addWidget(remove_all_button_);
    buttons->addStretch(1);
    highlight_check_ = new QCheckBox(app::tr("Highlight best/worst results"));
    highlight_check_->setChecked(settings_.highlight_best_worst);
    connect(highlight_check_, &QCheckBox::toggled, this, [this](bool on) { on_highlight_toggled(on); });
    buttons->addWidget(highlight_check_);
    layout->addLayout(buttons);
    best_worst_timer_ = new QTimer(this);
    best_worst_timer_->setSingleShot(true);
    best_worst_timer_->setInterval(0);
    connect(best_worst_timer_, &QTimer::timeout, this, [this] { apply_best_worst(); });
    connect(distorted_table_->model(), &QAbstractItemModel::rowsRemoved, this, [this] { schedule_best_worst(); });
    return {add_button, add_resample_button, remove_button, remove_all_button_};
}

QWidget* MainWindow::build_run_panel() {
    auto* panel = new QWidget;
    auto* layout = new QVBoxLayout(panel);

    auto* run_row = new QHBoxLayout;
    run_button_ = new QPushButton(app::tr("Calculate metrics"));
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
        {{"cores", cores}}));
    connect(parallel_jobs_check_, &QCheckBox::toggled, this, [this](bool on) { on_parallel_jobs_changed(on); });
    run_row->addWidget(parallel_jobs_check_);
    run_row->addStretch(1);

    load_button_ = new QPushButton(app::tr("Load analysis results..."));
    connect(load_button_, &QPushButton::clicked, this, [this] { on_load_saved_run(); });
    // Frozen during a run with the files: a saved run can bring a reference
    // of its own, which would change what the running videos mean.
    file_action_widgets_.push_back(load_button_);
    save_button_ = new QPushButton(app::tr("Save selected results..."));
    connect(save_button_, &QPushButton::clicked, this, [this] { on_save_selected(); });
    show_graph_button_ = new QPushButton(app::tr("Open metric graphs"));
    show_graph_button_->setToolTip(app::tr("Opens Metric Graphs with the current results and graph settings."));
    connect(show_graph_button_, &QPushButton::clicked, this, [this] { on_show_graph_clicked(); });
    run_row->addWidget(load_button_);
    run_row->addWidget(save_button_);
    run_row->addWidget(show_graph_button_);
    layout->addLayout(run_row);

    status_label_ = new QLabel;
    layout->addWidget(status_label_);
    // One line of text per running video, built once and hidden: lines
    // appearing and disappearing mid-run would shift everything below them.
    for (int i = 0; i < max_videos_in_flight; ++i) {
        auto* line = new ElidedLabel;
        theme::style(line, "color: {run_line};");
        line->setVisible(false);
        layout->addWidget(line);
        job_progress_labels_.push_back(line);
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
    options.vship_build = default_vship_build(settings_.default_vship_build);
    options.extra_features = settings_.default_extra_features();
    options.compute_xpsnr = settings_.default_compute_xpsnr;
    options.compute_vmaf = settings_.default_compute_vmaf;
    options.compute_vmaf_neg = settings_.default_compute_vmaf_neg;
    options.compute_vmaf_v1 = settings_.default_compute_vmaf_v1;
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
    if (!run_active_) {
        start_cache_lookup(apply_film_grain_default());
        show_panel_film_grain();
    }
}

// ------------------------------------------------------------ the update check

void MainWindow::check_for_updates() {
    if (!settings_.check_for_updates) {
        return;
    }
    // Through the system's proxy, if it names one, as the Python app's
    // urllib asks.
    QNetworkProxyFactory::setUseSystemConfiguration(true);
    auto* network = new QNetworkAccessManager(this);
    QNetworkRequest request(QUrl(QString::fromUtf8(releases_url.data(), static_cast<qsizetype>(releases_url.size()))));
    request.setRawHeader("Accept", "application/vnd.github+json");
    // GitHub refuses requests without a User-Agent.
    request.setHeader(QNetworkRequest::UserAgentHeader, "VideoMetricsLab/" VML_VERSION);
    request.setTransferTimeout(update_check_timeout_seconds * 1000);
    QNetworkReply* reply = network->get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, network] {
        const QString error = reply->error() == QNetworkReply::NoError ? QString() : reply->errorString();
        const QByteArray body = reply->readAll();
        reply->deleteLater();
        network->deleteLater();
        on_update_answer(body.toStdString(), error);
    });
}

void MainWindow::on_update_answer(const std::string& body, const QString& error) {
    std::optional<Release> release;
    try {
        if (!error.isEmpty()) {
            throw UpdateCheckError(error.toStdString());
        }
        release = release_from_answer(body);
    } catch (const UpdateCheckError& failure) {
        logger.info(std::string("Update check: GitHub could not be asked (") + failure.what() + ")");
        return;
    }
    if (!is_newer(release->version, VML_VERSION)) {
        logger.info("Update check: VideoMetricsLab " VML_VERSION " is the latest release");
        return;
    }
    if (!offers_package_here(*release)) {
        logger.info("Update check: VideoMetricsLab " + release->version +
                    " is available, but not yet for this system (this is " VML_VERSION ")");
        return;
    }
    logger.info("Update check: VideoMetricsLab " + release->version + " is available (this is " VML_VERSION ")");
    offer_update(*release);
}

void MainWindow::offer_update(const Release& release) {
    // Nothing is shown for a release the user chose to skip.
    if (release.version == settings_.skipped_update_version) {
        return;
    }
    auto* box = new QMessageBox(this);
    box->setAttribute(Qt::WA_DeleteOnClose);
    box->setIcon(QMessageBox::Information);
    box->setWindowTitle(app::tr("Update available"));
    box->setText(app::tr("{app} {version} is available. You have {current}.",
                         {{"app", "VideoMetricsLab"},
                          {"version", QString::fromStdString(release.version)},
                          {"current", VML_VERSION}}));
    box->setInformativeText(app::tr("Updating does not affect your saved results or settings: they are stored "
                                    "separately from the app, in {cache_dir} and {user_data_dir}.",
                                    {{"cache_dir", text_of(cache_folder())},
                                     {"user_data_dir", text_of(Settings::file().parent_path())}}));
    if (!python_strip(release.notes).empty()) {
        box->setDetailedText(QString::fromStdString(release.notes));
    }
    QPushButton* download = box->addButton(app::tr("Download"), QMessageBox::AcceptRole);
    QPushButton* skip = box->addButton(app::tr("Skip this version"), QMessageBox::DestructiveRole);
    box->addButton(app::tr("Later"), QMessageBox::RejectRole);
    box->setDefaultButton(download);
    connect(box, &QMessageBox::finished, this, [this, box, download, skip, release] {
        if (box->clickedButton() == download) {
            // This system's package itself, where the release names it (the
            // Python app opened the page).
            QDesktopServices::openUrl(QUrl(QString::fromStdString(release.download_url.value_or(release.page_url))));
        } else if (box->clickedButton() == skip) {
            settings_.skipped_update_version = release.version;
            save_settings();
        }
    });
    update_box_ = box;
    box->open();
}

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
    if (prompt) {
        log_tools(status, detected_gpu_vendors_);
    }
    if (status.ok()) {
        ffmpeg_banner_->setVisible(false);
        return true;
    }
    QStringList problems;
    for (const std::string& problem : status.problems()) {
        problems.append(QString::fromStdString(problem));
    }
#ifdef __APPLE__
    // The formula with zscale, which plain ffmpeg lacks (missing_zscale_text).
    const QString install = app::tr("  Install it with Homebrew: brew install ffmpeg-full.");
#else
    const QString install;
#endif
    ffmpeg_banner_->setText(problems.join("  ") + install +
                            app::with_program_names(app::tr("  Click \"Locate ffmpeg.exe...\" to point at it.")));
    ffmpeg_banner_->setVisible(true);
    locate_ffmpeg_button_->setVisible(true);
    // A run with no screen (the tests) has nobody to answer a dialog.
    if (prompt && qEnvironmentVariable("QT_QPA_PLATFORM") != "offscreen") {
        const auto answer = QMessageBox::warning(
            this, app::tr("ffmpeg not usable"),
            problems.join("\n") + install +
                app::tr("\n\nffmpeg was looked for at:\n  {path}", {{"path", status.ffmpeg.path}}) +
                app::tr("\nffprobe at:\n  {path}", {{"path", status.ffprobe.path}}) +
                app::with_program_names(app::tr("\n\nWould you like to point the app at ffmpeg.exe now?")),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        if (answer == QMessageBox::Yes) {
            return on_locate_ffmpeg();
        }
    }
    return false;
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
    const QString chosen = QFileDialog::getOpenFileName(
        this, app::tr("Select {ffmpeg_exe}", {{"ffmpeg_exe", ffmpeg_exe}}), QString(),
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
        const QString probe_chosen = QFileDialog::getOpenFileName(
            this, app::tr("Select {ffprobe_exe}", {{"ffprobe_exe", ffprobe_exe}}), text_of(directory),
            app::tr("{ffprobe_exe} ({ffprobe_exe});;All files (*)", {{"ffprobe_exe", ffprobe_exe}}));
        if (probe_chosen.isEmpty()) {
            return false;
        }
        if (!same_path(path_of(probe_chosen).parent_path(), directory)) {
            QMessageBox::warning(this, app::tr("Different folders"),
                                 app::tr("{ffmpeg_exe} and {ffprobe_exe} need to be in the same folder for the app to "
                                         "find both.",
                                         {{"ffmpeg_exe", ffmpeg_exe}, {"ffprobe_exe", ffprobe_exe}}));
            return false;
        }
    }
    // Kept as the Settings tab's folder: the one place it is kept.
    settings_.ffmpeg_dir = platform::path_text(directory);
    settings_panel_->show_ffmpeg_folder(QString::fromStdString(settings_.ffmpeg_dir));
    if (const std::optional<std::string> problem = settings_.save()) {
        show_status(tr_message(QString::fromStdString(*problem)));
        return false;
    }
    const ToolsStatus status = check_tools(settings_);
    tools_status_ = status;
    if (!status.ok()) {
        QStringList problems;
        for (const std::string& problem : status.problems()) {
            problems.append(QString::fromStdString(problem));
        }
        QMessageBox::warning(this, app::tr("Still not usable"), problems.join("\n"));
        check_ffmpeg(false);
        return false;
    }
    ffmpeg_banner_->setVisible(false);
    locate_ffmpeg_button_->setVisible(false);
    show_status(app::tr("Using ffmpeg {version} from {directory}",
                        {{"version", format_version(status.ffmpeg.version)}, {"directory", text_of(directory)}}));
    return true;
}

}  // namespace vml::app
