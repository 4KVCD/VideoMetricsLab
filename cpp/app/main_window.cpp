#include "main_window.h"

#include <QAbstractItemView>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QItemSelection>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QMouseEvent>
#include <QPushButton>
#include <QSplitter>
#include <QStyleHints>
#include <QTableWidgetItem>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>
#include <variant>

#include "cvvdp_display_dialog.h"
#include "file_writes.h"
#include "formatting.h"
#include "i18n.h"
#include "options_panel.h"
#include "run_worker.h"
#include "theme.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/geometry.h"
#include "vml/hardware_decoding.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/time_format.h"
#include "vml/vship.h"
#include "widgets.h"
#include "window_helpers.h"

// The window's text is app::tr's (i18n.h), always written so: Q_OBJECT gives
// the window a tr() of Qt's own, which inside its functions would hide the
// app's and show the English.
namespace vml::app {

namespace {

// The videos in progress at once: two with the CPU's metrics, and one more
// on the GPU (the Python app's job_runner MAX_VIDEOS_IN_FLIGHT).
constexpr int max_videos_in_flight = max_parallel_jobs + 1;

// A metric cell's score, as compared by Highlight best/worst results:
// nothing where it shows none to compare.
constexpr int score_role = Qt::UserRole;

// "1920x1080".
QString size_text(int width, int height) { return QStringLiteral("%1x%2").arg(width).arg(height); }

const MetricColumn* metric_column_of(int column) {
    for (const MetricColumn& item : metric_columns()) {
        if (item.column == column) {
            return &item;
        }
    }
    return nullptr;
}

// Settings' default_compute_<key> for each metric.
bool Settings::* default_compute_field(std::string_view key) {
    static const std::map<std::string_view, bool Settings::*> fields{
        {"psnr", &Settings::default_compute_psnr},
        {"ssim", &Settings::default_compute_ssim},
        {"xpsnr", &Settings::default_compute_xpsnr},
        {"vmaf", &Settings::default_compute_vmaf},
        {"vmaf_neg", &Settings::default_compute_vmaf_neg},
        {"vmaf_v1", &Settings::default_compute_vmaf_v1},
        {"ssimulacra2", &Settings::default_compute_ssimulacra2},
        {"butteraugli", &Settings::default_compute_butteraugli},
        {"cvvdp", &Settings::default_compute_cvvdp},
    };
    const auto found = fields.find(key);
    return found == fields.end() ? nullptr : found->second;
}

// A menu of check boxes that stays open while boxes are ticked, so several
// metrics can be switched in one visit.
class StayOpenMenu : public QMenu {
public:
    using QMenu::QMenu;

protected:
    void mouseReleaseEvent(QMouseEvent* event) override {
        QAction* action = actionAt(event->position().toPoint());
        if (action != nullptr && action->isCheckable() && action->isEnabled()) {
            action->trigger();
            return;
        }
        QMenu::mouseReleaseEvent(event);
    }
};

// A file name split into its runs of digits and of other text, for sorting
// "clip 9" before "clip 10" (the Python app's re.split(r"(\d+)", name)): the
// text casefolded, the digits compared as numbers.
std::vector<QString> natural_parts(const QString& name) {
    std::vector<QString> parts{QString()};
    bool digits = false;
    for (const QChar c : name) {
        if (c.isDigit() != digits) {
            digits = c.isDigit();
            parts.emplace_back();
        }
        parts.back() += c;
    }
    for (std::size_t i = 0; i < parts.size(); i += 2) {
        parts[i] = parts[i].toCaseFolded();
    }
    return parts;
}

// A run of digits as the number Python's int() reads in it, of any length
// and in any script's digits: its digits' values, without leading zeros.
std::vector<int> digit_values(const QString& digits) {
    std::vector<int> values;
    for (const QChar c : digits) {
        if (!values.empty() || c.digitValue() != 0) {
            values.push_back(c.digitValue());
        }
    }
    return values;
}

// Python's comparison of the parts' lists: the text parts (even places) as
// text, the digit parts (odd places) as whole numbers.
bool natural_less(const std::vector<QString>& a, const std::vector<QString>& b) {
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (i % 2 == 0) {
            if (const int order = QString::compare(a[i], b[i]); order != 0) {
                return order < 0;
            }
            continue;
        }
        const std::vector<int> x = digit_values(a[i]);
        const std::vector<int> y = digit_values(b[i]);
        if (x.size() != y.size()) {
            return x.size() < y.size();
        }
        if (x != y) {
            return x < y;
        }
    }
    return a.size() < b.size();
}

// What a column sorts a row by: one kind for every row of a column.
using SortKey = std::variant<std::vector<QString>, std::tuple<long long, double, double>, long long, int, QString,
                             double>;

bool sort_less(const SortKey& a, const SortKey& b) {
    if (const auto* x = std::get_if<std::vector<QString>>(&a)) {
        return natural_less(*x, std::get<std::vector<QString>>(b));
    }
    if (const auto* x = std::get_if<QString>(&a)) {
        return QString::compare(*x, std::get<QString>(b)) < 0;
    }
    return a < b;
}

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

// ======================================================================= window

MainWindow::MainWindow() {
    setWindowTitle(QStringLiteral("VideoMetricsLab %1").arg(VML_APP_VERSION));
    // Settings first: the ffmpeg location and what new rows default to both
    // come from them, so they must be applied before the tool check or any
    // row is added.
    settings_ = Settings::load();
    if (settings_.remember_window_size) {
        resize(settings_.window_width, settings_.window_height);
    } else {
        resize(1280, 800);
    }
    // Each video's own settings; the Options panel shows and edits the
    // selected rows' settings, and there is no one global setting.
    default_options_ = options_from_settings();
    default_extra_metric_keys_ = extra_metrics_from_settings();
    default_metric_backends_ = {
        {"ssimulacra2", settings_.default_ssimulacra2_backend == "cpu" ? "cpu" : "gpu"},
        {"butteraugli", settings_.default_butteraugli_backend == "cpu" ? "cpu" : "gpu"},
    };
    default_cvvdp_ = cvvdp_from_settings();
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
    clock_ = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
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
    check_ffmpeg(true);  // the start's check: both tools present, ffmpeg new enough
    restore_open_videos();
    on_table_selection_changed();
}

MainWindow::~MainWindow() {
    // Hidden first, while the window is whole: hiding moves the focus, and
    // the reference box reads what it holds when the focus leaves it.
    hide();
    // Each stops after the video it is reading: one ffprobe at most.
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
    return workers;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    // Don't close while background work still writes into the widgets: it
    // would crash the app and leave the process running. The work is told
    // to stop (a run ends the programs it started), and the window closes
    // itself once it has.
    if (!closing_) {
        closing_ = true;
        for (ProbeWorker* worker : probe_workers_) {
            worker->cancel();
        }
        if (worker_ != nullptr) {
            worker_->cancel();
        }
    }
    // Nor during a write of saved scores, which closing would lose.
    const bool written = file_writes_->wait_until_idle(std::chrono::milliseconds(500));
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
    // respond meanwhile.
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
    auto* watcher = new QFutureWatcher<std::pair<QString, QStringList>>(this);
    connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
        const auto [present_reference, present] = watcher->result();
        watcher->deleteLater();
        reopen_videos(present_reference, present);
    });
    watcher->setFuture(QtConcurrent::run([reference, tests] {
        const QString present_reference = !reference.isEmpty() && QFileInfo(reference).isFile() ? reference : QString();
        QStringList present;
        for (const QString& path : tests) {
            if (QFileInfo(path).isFile()) {
                present.append(path);
            }
        }
        return std::pair{present_reference, present};
    }));
}

void MainWindow::reopen_videos(const QString& reference, const QStringList& tests) {
    // Added unless the window has gained videos of its own since.
    if (!rows_.empty() || source_info_ || source_reading_ || run_active_) {
        return;
    }
    if (!reference.isEmpty()) {
        choose_source(path_of(reference));
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
    locate_ffmpeg_button_ = new QPushButton(app::tr("Locate ffmpeg.exe..."));
    connect(locate_ffmpeg_button_, &QPushButton::clicked, this, [this] { on_locate_ffmpeg(); });
    locate_ffmpeg_button_->setVisible(false);
    banner_row->addWidget(ffmpeg_banner_, 1);
    banner_row->addWidget(locate_ffmpeg_button_);
    root->addLayout(banner_row);

    // Tabs, not separate windows. Each tab but Video Compare scrolls when the
    // window is too small for it (ScrollingPage).
    tabs_ = new QuietTabWidget;
    root->addWidget(tabs_, 1);

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
    connect(options_panel_, &OptionsPanel::metric_backend_changed, this,
            [this](const QString& key) { on_metric_backend_changed(key); });
    connect(options_panel_, &OptionsPanel::cvvdp_preset_chosen, this,
            [this](int index) { on_cvvdp_preset_chosen(index); });
    connect(options_panel_, &OptionsPanel::cvvdp_resize_toggled, this,
            [this](bool checked) { on_cvvdp_resize_toggled(checked); });
    connect(options_panel_, &OptionsPanel::cvvdp_add_requested, this, [this] { on_cvvdp_add_preset(); });
    connect(options_panel_, &OptionsPanel::cvvdp_edit_requested, this, [this] { on_cvvdp_edit_display(); });
    connect(options_panel_, &OptionsPanel::cvvdp_delete_requested, this, [this] { on_cvvdp_delete_preset(); });
    connect(options_panel_, &OptionsPanel::scale_direction_chosen, this,
            [this](int index) { on_scale_direction_combo_changed(index); });
    run_cvvdp_dialog_ = [](CvvdpDisplayDialog& dialog) { return dialog.exec(); };
    inform_ = [this](const QString& title, const QString& text) { QMessageBox::information(this, title, text); };
    alert_ = [this](const QString& title, const QString& text) { QMessageBox::critical(this, title, text); };
    ask_ = [this](const QString& title, const QString& text) {
        return QMessageBox::question(this, title, text) == QMessageBox::Yes;
    };
    open_file_ = [this](const QString& title, const QString& filter) {
        return QFileDialog::getOpenFileName(this, title, QString(), filter);
    };
    save_file_ = [this](const QString& title, const QString& suggested, const QString& filter) {
        return QFileDialog::getSaveFileName(this, title, suggested, filter);
    };
    choose_folder_ = [this](const QString& title) { return QFileDialog::getExistingDirectory(this, title); };
    splitter->addWidget(options_panel_);
    splitter->addWidget(build_run_panel());
    splitter->setSizes({380, 300, 120});
    tabs_->addTab(new ScrollingPage(videos_page), app::tr("Videos"));

    const auto placeholder = [](const QString& what) {
        auto* label = new QLabel(what + QStringLiteral(" is not ported to C++ yet."));
        label->setAlignment(Qt::AlignCenter);
        return label;
    };
    tabs_->addTab(new ScrollingPage(placeholder(QStringLiteral("Metric Graphs"))), app::tr("Metric Graphs"));
    tabs_->addTab(placeholder(QStringLiteral("Video Compare")), app::tr("Video Compare"));
    tabs_->addTab(new ScrollingPage(placeholder(QStringLiteral("The Bitrate Viewer"))), app::tr("Bitrate Viewer"));
    tabs_->addTab(new ScrollingPage(placeholder(QStringLiteral("Settings"))), app::tr("Settings"));
}

QWidget* MainWindow::build_files_panel() {
    files_box_ = new QGroupBox(app::tr("Videos"));
    auto* files_layout = new QVBoxLayout(files_box_);

    files_layout->addWidget(new QLabel(app::tr("Reference video:")));
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
    files_layout->addLayout(source_row);
    source_info_label_ = new QLabel(app::tr("No reference selected."));
    theme::style(source_info_label_, "color: {muted};");
    files_layout->addWidget(source_info_label_);

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
    // One block, heading row and table, with no spacing between them.
    auto* table_block = new QVBoxLayout;
    table_block->setSpacing(0);
    table_block->addLayout(table_heading);
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
    table_block->addWidget(distorted_table_, 1);
    files_layout->addLayout(table_block, 1);

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
    files_layout->addLayout(buttons);
    best_worst_timer_ = new QTimer(this);
    best_worst_timer_->setSingleShot(true);
    best_worst_timer_->setInterval(0);
    connect(best_worst_timer_, &QTimer::timeout, this, [this] { apply_best_worst(); });
    connect(distorted_table_->model(), &QAbstractItemModel::rowsRemoved, this, [this] { schedule_best_worst(); });

    // Frozen during a run one control at a time. Don't disable the whole box:
    // a disabled QGroupBox disables its children, the table's scroll bar
    // among them, so a long queue could not be scrolled while it runs.
    file_action_widgets_ = {source_browse_button_, add_button, add_resample_button, remove_button,
                            remove_all_button_, metrics_button_};
    return files_box_;
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
    show_graph_button_->setEnabled(false);  // Metric Graphs are not ported to C++ yet
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
        if (settings_.*default_compute_field(key)) {
            keys.insert(key);
        }
    }
    return keys;
}

CvvdpSettings MainWindow::cvvdp_from_settings() const {
    // The preset chosen in Settings, or the built-in default if that preset
    // is gone or unset. A preset is a display; scaling to fill it is each
    // video's own choice and starts off.
    std::optional<CvvdpPreset> preset;
    if (!settings_.cvvdp_default_preset.empty()) {
        preset = cvvdp_preset_named(settings_.cvvdp_default_preset, settings_.cvvdp_presets_json);
    }
    CvvdpSettings settings;
    settings.display = (preset ? preset->settings : builtin_cvvdp_presets().front().settings).display;
    return settings;
}

// ------------------------------------------------------- FFmpeg's presence

bool MainWindow::check_ffmpeg(bool prompt) {
    // Both tools must run, and ffmpeg be new enough. At the start (prompt),
    // a file picker to point at ffmpeg.exe is offered rather than only a
    // banner.
    const ToolsStatus status = check_tools(settings_);
    if (status.ok()) {
        ffmpeg_banner_->setVisible(false);
        return true;
    }
    QStringList problems;
    for (const std::string& problem : status.problems()) {
        problems.append(QString::fromStdString(problem));
    }
    ffmpeg_banner_->setText(problems.join("  ") + app::tr("  Click \"Locate ffmpeg.exe...\" to point at it."));
    ffmpeg_banner_->setVisible(true);
    locate_ffmpeg_button_->setVisible(true);
    // A run with no screen (the tests) has nobody to answer a dialog.
    if (prompt && qEnvironmentVariable("QT_QPA_PLATFORM") != "offscreen") {
        const auto answer = QMessageBox::warning(
            this, app::tr("ffmpeg not usable"),
            problems.join("\n") +
                app::tr("\n\nffmpeg was looked for at:\n  {path}", {{"path", status.ffmpeg.path}}) +
                app::tr("\nffprobe at:\n  {path}", {{"path", status.ffprobe.path}}) +
                app::tr("\n\nWould you like to point the app at ffmpeg.exe now?"),
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
    if (!std::filesystem::exists(directory / platform::path_from_text(ffprobe_exe.toStdString()), error)) {
        inform_(app::tr("ffprobe needed too"),
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
    if (const std::optional<std::string> problem = settings_.save()) {
        show_status(tr_message(QString::fromStdString(*problem)));
        return false;
    }
    const ToolsStatus status = check_tools(settings_);
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
        settings_.save();
    }
}

void MainWindow::start_source_probe(const std::filesystem::path& path) {
    // Read without freezing the window. A newer choice replaces one still
    // being read.
    if (source_probe_worker_ != nullptr && source_probe_worker_->isRunning()) {
        source_probe_worker_->cancel();
    }
    const int generation = ++source_probe_generation_;
    auto* worker = new ProbeWorker({text_of(path)}, find_tool("ffprobe", settings_), this);
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
        source_info_label_->setText(
            source_info_ ? media_info_string(*source_info_) + ", " + bitrate_string(*source_info_) + "  (" +
                               QString::fromStdString(format_hms(source_info_->duration, 1)) + ")"
                         : app::tr("No reference selected."));
        QMessageBox::critical(this, app::tr("Could not read video"), tr_message(probed.error));
        return;
    }
    apply_source_info(path_of(probed.path), *probed.info);
}

void MainWindow::apply_source_info(const std::filesystem::path& path, const VideoInfo& info) {
    source_info_ = info;
    source_edit_->setText(text_of(path));
    source_info_label_->setText(media_info_string(info) + ", " + bitrate_string(info) + "  (" +
                                QString::fromStdString(format_hms(info.duration, 1)) + ")");
    check_rows_against_new_source(path);
}

void MainWindow::check_rows_against_new_source(const std::filesystem::path& path) {
    // The rows, after `path` became the reference: chosen, or taken from a
    // saved run. A resolution test is made from the reference, while a test
    // video is only compared with it, so a resolution test cannot outlive a
    // change of reference.
    const std::vector<QString> dropped = remove_rows_owned_by_the_previous_source();
    // Scores belong to a pair of videos: a new reference takes away every
    // score measured against another, which would otherwise stand in the
    // table as the new pair's, and keep a run from calculating it, and the
    // state of every row without scores (how its run against the old one
    // ended, failed or cancelled). Those measured against the very reference
    // chosen stay: choosing it again confirms them.
    std::vector<int> stale;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (!row_data.completed_run || !same_file(row_data.completed_run->result.source, path)) {
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
        inform_(app::tr("Resolution tests removed"),
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
    for (int row = static_cast<int>(rows_.size()) - 1; row >= 0; --row) {
        if (!rows_[static_cast<std::size_t>(row)]->options.resample_test) {
            continue;
        }
        removed.push_back(name_of(rows_[static_cast<std::size_t>(row)]->path));
        distorted_table_->removeRow(row);
        rows_.erase(rows_.begin() + row);
    }
    if (!removed.empty()) {
        on_table_selection_changed();
    }
    return removed;
}

void MainWindow::invalidate_completed_results(const std::vector<int>& rows) {
    // The rows' scores go, with their state and its detail: the tooltip
    // would otherwise say "Not calculated" over the old result's "240 scored
    // frames; metrics: ...", or over a failure's reason.
    for (const int row : rows) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        row_data.analysis_status.reset();
        row_data.status_detail.clear();
        row_data.completed_run.reset();
        set_row_metrics(row);
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

// --------------------------------------------------------- the test videos

int MainWindow::add_table_row(const std::filesystem::path& path) {
    // Rows are added at the end, so the table is not in the order a heading's
    // sort arrow says any more.
    clear_sort_indicator();
    const int row = distorted_table_->rowCount();
    distorted_table_->insertRow(row);

    auto* check_item = new QTableWidgetItem;
    check_item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    check_item->setCheckState(Qt::Checked);
    distorted_table_->setItem(row, col_check, check_item);

    auto* path_item = new QTableWidgetItem(name_of(path));
    path_item->setToolTip(text_of(path));  // the full path on hover
    distorted_table_->setItem(row, col_path, path_item);
    distorted_table_->setItem(row, col_info, new QTableWidgetItem(app::tr("Probing...")));
    auto* bars_item = new QTableWidgetItem;
    bars_item->setTextAlignment(Qt::AlignCenter);
    distorted_table_->setItem(row, col_black_bars, bars_item);
    distorted_table_->setItem(row, col_scaling, new QTableWidgetItem(QString()));
    distorted_table_->setItem(row, col_bitrate, new QTableWidgetItem(QString()));
    for (const MetricColumn& item : metric_columns()) {
        auto* cell = new QTableWidgetItem(QString());
        cell->setTextAlignment(Qt::AlignCenter);
        // Coloured by the best/worst highlighting alone from here on.
        cell->setBackground(QColor(0, 0, 0, 0));
        distorted_table_->setItem(row, item.column, cell);
    }

    // New rows start with a copy of the defaults, so that several similar
    // files need not each be set up from scratch. Each row has its own copy,
    // so editing one row never changes another.
    auto row_data = std::make_shared<RowData>();
    row_data->path = path;
    row_data->options = default_options_;
    row_data->extra_metric_keys = default_extra_metric_keys_;
    row_data->metric_backends = default_metric_backends_;
    row_data->cvvdp = default_cvvdp_;
    row_data->cvvdp_preset = settings_.cvvdp_default_preset;
    rows_.push_back(row_data);
    set_row_metrics(row);
    return row;
}

void MainWindow::set_row_metrics(int row) {
    // Each metric cell shows its score, or a tick box for calculating it. A
    // metric that has been measured shows only the number: there is no
    // choice left to make about it, and a tick box beside a score invites
    // unticking it as though that would undo the measurement. A metric
    // without a score shows a tick box in the column its result will go in,
    // so choosing what to calculate is one click on the row. Scores stay when
    // the ticks change: ticking another metric changes what is asked for,
    // not what was measured.
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    CompletedRun* run = row_data.completed_run.get();
    // setCheckState/setData below emit itemChanged, which is also how a real
    // click arrives: without this the window would read its own repaint back
    // as the user asking for a change.
    syncing_table_ = true;
    for (const MetricColumn& metric : metric_columns()) {
        QTableWidgetItem* item = distorted_table_->item(row, metric.column);
        if (item == nullptr) {
            continue;
        }
        const bool enabled = row_metric_enabled(row_data, metric.column);
        const std::optional<double> score = run != nullptr ? score_of(*run, metric.column) : std::nullopt;
        if (score) {
            // Measured: the score replaces the tick box. Taking the check
            // state's value away is what removes the box: Qt draws one for
            // any item that has the role set at all.
            item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
            item->setData(Qt::CheckStateRole, QVariant());
            const MetricDefinition& definition = metric_definition(metric.key);
            const FrameMetricResult* frame_metric = run->result.metric_results.frame(metric.key);
            QString text = qt_text(definition.format_value(*score));
            const bool other_scale = frame_metric != nullptr && row_data.metric_backends.contains(metric.key) &&
                                     !backend_matches(row_data, metric.key, frame_metric->provenance);
            if (other_scale) {
                // Visible without hovering: a score from the other
                // implementation is on a different scale from the rest.
                text += " (" + QString::fromStdString(frame_metric->provenance.compute_backend).toUpper() + ")";
            }
            item->setText(text);
            // Compared with the others' unless on another scale.
            item->setData(score_role, other_scale ? QVariant() : QVariant(*score));
            item->setToolTip(definition.kind == MetricKind::sequence
                                 ? cvvdp_note(row_data)
                                 : app::tr("Mean of calculated frame scores.") +
                                       identical_frames_note(*run, metric.column) +
                                       backend_note(row_data, metric.key, frame_metric));
            QFont font;
            font.setBold(true);
            item->setFont(font);
            theme::foreground(item, std::nullopt);
            continue;
        }
        if (const std::optional<QString> unavailable = metric_unavailable_reason(row_data, metric.key)) {
            // No tick box: there is nothing to choose. The row's own selection
            // is left alone for when it applies again.
            item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
            item->setData(Qt::CheckStateRole, QVariant());
            item->setText("n/a");
            item->setToolTip(*unavailable);
            item->setForeground(theme::color("faint"));
            item->setData(score_role, QVariant());
            item->setFont(QFont());
            continue;
        }
        item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        item->setCheckState(enabled ? Qt::Checked : Qt::Unchecked);
        const bool failed = enabled && (row_data.analysis_status == RowState::failed ||
                                         row_data.analysis_status == RowState::partly_failed);
        item->setText(failed ? app::tr("Failed") : QString());
        QString tooltip;
        if (failed) {
            QString reason = row_data.metric_failures.contains(metric.key) ? row_data.metric_failures.at(metric.key)
                                                                           : QString();
            if (reason.isEmpty() && row_data.analysis_status == RowState::failed) {
                reason = row_data.status_detail.section("\n\n", 0, 0);  // the whole video failed: its reason
            }
            tooltip = (reason.isEmpty()
                           ? app::tr("This metric failed on the last run.")
                           : app::tr("Failed on the last run: {reason}", {{"reason", tr_message(reason)}})) +
                      app::tr("\n\nLeft ticked, it is calculated again on the next run; untick to skip it.\nThe full "
                              "details are in the log (Settings > Storage > Log files).");
        } else {
            tooltip = enabled ? app::tr("Ticked: calculated on the next run. Untick to skip it.")
                              : app::tr("Not selected. Tick to calculate this metric.");
        }
        if (std::string_view(metric.key) == "cvvdp") {
            tooltip += cvvdp_elsewhere_note(row_data);
        }
        item->setToolTip(tooltip);
        theme::foreground(item, failed ? std::optional(theme::color("failed")) : std::nullopt);
        item->setData(score_role, QVariant());
        item->setFont(QFont());
    }
    syncing_table_ = false;
    schedule_best_worst();
    refresh_row_state(row);
    set_row_black_bars(row);
    set_row_scaling(row);
}

RowState MainWindow::row_state(const RowData& row_data) const {
    // How the row's analysis stands, in one phrase. Above all what the
    // metric cells cannot say: which stage a live job is at, and why a
    // finished job is not shown.
    if (row_data.analysis_status) {
        return *row_data.analysis_status;
    }
    if (requested_metrics(row_data).empty()) {
        return RowState::no_metrics;
    }
    if (has_requested_results(row_data)) {
        return RowState::complete;
    }
    return row_data.completed_run ? RowState::partial : RowState::not_calculated;
}

void MainWindow::refresh_row_state(int row) {
    // The row's state on its file name: the full path, then the state and
    // any detail, with a colour for the two cases nothing else on the row
    // shows.
    QTableWidgetItem* item = distorted_table_->item(row, col_path);
    if (item == nullptr) {
        return;
    }
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const RowState state = row_state(row_data);
    QStringList lines{text_of(row_data.path), row_state_shown(state)};
    if (!row_data.status_detail.isEmpty()) {
        lines.append(tr_message(row_data.status_detail));
    }
    item->setToolTip(lines.join("\n\n"));
    if (state == RowState::failed) {
        item->setForeground(theme::color("failed"));
    } else if (finished_not_shown(state)) {
        // Done and saved, but on purpose not shown: without a mark the row
        // would look untouched.
        item->setForeground(theme::color("stale"));
    } else {
        theme::foreground(item, std::nullopt);
    }
}

QString MainWindow::crop_detail(const QString& label, const VideoInfo& info, const std::optional<CropBox>& crop) {
    // One video's crop, spelled out for the Black bars cell's tooltip.
    if (!crop) {
        return app::tr("{label}: not checked for black bars; no crop was applied.", {{"label", label}});
    }
    if (crop->is_noop(info.width, info.height)) {
        return app::tr("{label}: no black bars. Compared in full at {size}.",
                       {{"label", label}, {"size", size_text(info.width, info.height)}});
    }
    QStringList sides;
    for (const auto& [side, pixels] : {std::pair{"top {pixels} px", std::max(0, crop->y)},
                                       std::pair{"bottom {pixels} px", std::max(0, info.height - crop->y - crop->h)},
                                       std::pair{"left {pixels} px", std::max(0, crop->x)},
                                       std::pair{"right {pixels} px", std::max(0, info.width - crop->x - crop->w)}}) {
        if (pixels != 0) {
            sides.append(app::tr(side, {{"pixels", pixels}}));
        }
    }
    return app::tr("{label}: black bars cropped off -- {sides}.\nCompared at {cropped} instead of {full}.",
                   {{"label", label},
                    {"sides", sides.join(", ")},
                    {"cropped", size_text(crop->w, crop->h)},
                    {"full", size_text(info.width, info.height)}});
}

void MainWindow::set_row_black_bars(int row, bool probe_failed) {
    // Whether the test video has black bars, and nothing more; the pixel
    // counts, and what was cropped off the reference, are in the tooltip.
    // Auto crop is worked out during a run, and can differ between rows, as
    // each has its own duration limit: the cell says it is pending until a
    // result gives the crops the run used.
    QTableWidgetItem* item = distorted_table_->item(row, col_black_bars);
    if (item == nullptr) {
        return;
    }
    const auto show = [item](const QString& text, const QString& tooltip, bool muted) {
        item->setText(text);
        item->setToolTip(tooltip);
        theme::foreground(item, muted ? std::optional(theme::color("dimmed")) : std::nullopt);
    };
    if (probe_failed) {
        show(app::tr("Unknown"), app::tr("Black bars could not be checked because the video could not be read."), true);
        return;
    }
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    if (!row_data.completed_run) {
        if (row_data.options.crop_mode == CropMode::none) {
            show(app::tr("Off"),
                 app::tr("Black-bar detection is disabled for this row; the full frames will be compared."), true);
        } else {
            show(app::tr("Pending"), app::tr("Black bars will be detected when this row is run."), true);
        }
        return;
    }
    const ComparisonResult& result = row_data.completed_run->result;
    // A resolution test has no test video of its own: both of its videos come
    // from the reference, so the reference's bars are the answer.
    const bool resample = row_data.options.resample_test.has_value();
    const std::optional<CropBox>& subject_crop = resample ? result.source_crop : result.distorted_crop;
    const VideoInfo& subject_info = resample ? result.source_info : result.distorted_info;
    if (!subject_crop) {
        show(app::tr("Off"), app::tr("Black-bar detection was disabled for this run; no crop was applied."), true);
        return;
    }
    QString tooltip = crop_detail(app::tr("Reference"), result.source_info, result.source_crop);
    if (!resample) {
        tooltip = crop_detail(app::tr("Test video"), result.distorted_info, result.distorted_crop) + "\n\n" + tooltip;
    }
    show(subject_crop->is_noop(subject_info.width, subject_info.height) ? app::tr("No") : app::tr("Yes"), tooltip,
         false);
}

std::pair<QString, QString> MainWindow::resize_mismatch(int row) const {
    // (short tag, full explanation) saying which of the two videos is
    // resized to match the other, when their sizes differ. The two are
    // compared after their black bars come off, as the run compares them: a
    // 1920x1080 letterboxed reference and a 1920x804 encode of the same film
    // are the same picture, not a resize. After a run, it uses the crops and
    // direction the run used, which stay right if the row's settings are
    // edited afterwards; before one, the row's own setting.
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const CompletedRun* run = row_data.completed_run.get();
    if (run == nullptr && (!source_info_ || !row_data.video_info)) {
        return {};
    }
    const VideoInfo& source = run != nullptr ? run->result.source_info : *source_info_;
    const VideoInfo& distorted = run != nullptr ? run->result.distorted_info : *row_data.video_info;
    const Size reference = run != nullptr ? content_size(source, run->result.source_crop)
                                          : Size{source.width, source.height};
    const Size test = run != nullptr ? content_size(distorted, run->result.distorted_crop)
                                     : Size{distorted.width, distorted.height};
    if (reference == test) {
        return {QString(), app::tr("Reference and test video are compared at the same {size} -- no scaling needed.",
                              {{"size", size_text(reference.first, reference.second)}})};
    }
    if (run == nullptr && row_data.options.crop_mode != CropMode::none &&
        (reference.first == test.first || reference.second == test.second)) {
        // One dimension already matches and only the other differs: the shape
        // of black bars. Whether any scaling is needed is known once they are
        // detected, when the row runs.
        return {app::tr("Pending"),
                app::tr("The reference is {reference} and the test video {test}, a difference in one dimension only -- "
                        "the shape of black bars on one of them.\n\nBars are detected when this row runs, and are "
                        "removed before the two are compared, so whether any scaling is needed is known then.",
                        {{"reference", size_text(source.width, source.height)},
                         {"test", size_text(distorted.width, distorted.height)}})};
    }
    // A "Test both" companion row always goes its own way, set as it is
    // made; a saved result from before results kept the direction says the
    // default, whatever made it.
    const ScaleDirection direction = run != nullptr && !row_data.scale_direction_pinned
                                         ? run->result.scale_direction
                                         : row_data.options.scale_direction;
    const bool cropped_either =
        reference != Size{source.width, source.height} || test != Size{distorted.width, distorted.height};
    const QString cropped = cropped_either ? app::tr(" (after black bars)") : QString();
    if (direction == ScaleDirection::distorted_to_source) {
        return {app::tr("Test upscaled to source"),
                app::tr("Test video upscaled {from_size} -> {to_size}{cropped} to match the reference.",
                        {{"from_size", size_text(test.first, test.second)},
                         {"to_size", size_text(reference.first, reference.second)},
                         {"cropped", cropped}})};
    }
    return {app::tr("\u2193 source"),
            app::tr("Reference downscaled {from_size} -> {to_size}{cropped} to match the test video.",
                    {{"from_size", size_text(reference.first, reference.second)},
                     {"to_size", size_text(test.first, test.second)},
                     {"cropped", cropped}})};
}

void MainWindow::fit_columns(std::initializer_list<int> columns) {
    // Fitted once, when the window is next idle: fitting measures every
    // row's cell, and fitting for each row as it is drawn would take 90,000
    // measurements a column for 300 rows.
    if (columns_to_fit_.empty()) {
        QTimer::singleShot(0, this, [this] { fit_pending_columns(); });
    }
    columns_to_fit_.insert(columns.begin(), columns.end());
}

void MainWindow::fit_pending_columns() {
    const std::set<int> columns = std::exchange(columns_to_fit_, {});
    for (const int column : columns) {
        distorted_table_->resizeColumnToContents(column);
    }
}

void MainWindow::set_row_scaling(int row) {
    QTableWidgetItem* item = distorted_table_->item(row, col_scaling);
    if (item == nullptr) {
        return;
    }
    const auto [tag, explanation] = resize_mismatch(row);
    item->setText(tag);
    item->setToolTip(explanation);
    fit_columns({col_scaling});
}

void MainWindow::set_row_info(int row, const std::optional<VideoInfo>& info, const QString& error) {
    QTableWidgetItem* item = distorted_table_->item(row, col_info);
    QTableWidgetItem* scaling_item = distorted_table_->item(row, col_scaling);
    if (!error.isEmpty() || !info) {
        set_row_status(row, RowState::failed, error);
        item->setText(app::tr("Probe failed"));
        item->setToolTip(error);
        item->setForeground(theme::color("failed"));
        distorted_table_->item(row, col_bitrate)->setText(QString());
        scaling_item->setText(QString());
        scaling_item->setToolTip(QString());
        set_row_black_bars(row, true);
    } else {
        item->setText(media_info_string(*info));
        // Back to normal text: the placeholder shown while probing greys the
        // cell, and leaving it grey makes a probed row look disabled.
        theme::foreground(item, std::nullopt);
        item->setToolTip(QString::fromStdString(format_hms(info->duration, 1)));
        distorted_table_->item(row, col_bitrate)->setText(bitrate_string(*info));
        distorted_table_->item(row, col_bitrate)->setToolTip(bitrate_note(*info));
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        row_data.video_info = info;
        set_row_scaling(row);
        if (row_data.analysis_status == RowState::reading) {
            row_data.analysis_status.reset();
            set_row_metrics(row);
        }
        set_row_black_bars(row);
    }
    // Snug to what is in them, while staying draggable in between updates.
    fit_columns({col_info, col_bitrate, col_scaling});
}

void MainWindow::set_row_status(int row, RowState state, const QString& detail) {
    RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    row_data.analysis_status = state;
    row_data.status_detail = detail;
    refresh_row_state(row);
}

void MainWindow::on_add_distorted() {
    const QStringList paths =
        QFileDialog::getOpenFileNames(this, app::tr("Select test videos"), video_dialog_dir("test"));
    if (paths.isEmpty()) {
        return;
    }
    remember_video_dir("test", path_of(paths.front()));
    std::vector<std::filesystem::path> chosen;
    for (const QString& path : paths) {
        chosen.push_back(path_of(path));
    }
    add_test_videos(chosen);
}

void MainWindow::on_files_dropped_on_table(const QStringList& paths) {
    if (run_active_ || paths.isEmpty()) {
        return;
    }
    remember_video_dir("test", path_of(paths.front()));
    std::vector<std::filesystem::path> dropped;
    for (const QString& path : paths) {
        dropped.push_back(path_of(path));
    }
    add_test_videos(dropped);
}

void MainWindow::add_test_videos(const std::vector<std::filesystem::path>& paths) {
    // A row for each video not already in the table, in order. The rows
    // appear at once; reading each file and looking up its saved scores
    // happen on workers, as for a handful of long videos they take a couple
    // of seconds.
    std::vector<std::filesystem::path> added;
    for (const std::filesystem::path& path : paths) {
        const auto same = [&path](const std::filesystem::path& other) { return same_path(path, other); };
        if (std::any_of(rows_.begin(), rows_.end(), [&same](const auto& row) { return same(row->path); }) ||
            std::any_of(added.begin(), added.end(), same)) {
            continue;
        }
        added.push_back(path);
    }
    if (added.empty()) {
        return;
    }
    for (const std::filesystem::path& path : added) {
        const int row = add_table_row(path);
        set_row_status(row, RowState::reading);
    }
    start_media_probe(added);
    start_cache_lookup(added);
}

void MainWindow::on_add_resample_test() {
    // Adds a resolution test for the width the user picks.
    if (!source_info_) {
        QMessageBox::warning(this, app::tr("No reference"), app::tr("Please select a reference video first."));
        return;
    }
    std::vector<ResampleTarget> targets;
    for (const ResampleTarget& target : resample_target_choices()) {
        if (target.width < source_info_->width) {
            targets.push_back(target);
        }
    }
    if (targets.empty()) {
        inform_(app::tr("No smaller resolution available"),
                app::tr("The smallest resolution test is {smallest} pixels wide, which is not below this "
                        "reference's {width}-pixel width.",
                        {{"smallest", resample_target_choices().back().width}, {"width", source_info_->width}}));
        return;
    }
    QStringList labels;
    for (const ResampleTarget& target : targets) {
        labels.append(QString::fromStdString(target.label));
    }
    bool chosen = false;
    const QString label =
        QInputDialog::getItem(this, app::tr("Add resolution test"), app::tr("Downscale to (then scale back up):"),
                              labels, std::min(1, static_cast<int>(labels.size()) - 1), false, &chosen);
    if (!chosen) {
        return;
    }
    add_resample_test(targets[static_cast<std::size_t>(labels.indexOf(label))]);
}

void MainWindow::add_resample_test(const ResampleTarget& target) {
    // A row for a resolution test of the reference at `target`.
    const std::filesystem::path path = synthetic_resample_distorted_path(source_info_->path, target);
    if (row_index_of_path(path)) {
        inform_(app::tr("Already added"), app::tr("A {label} resolution test for this reference is already in the list.",
                                                  {{"label", QString::fromStdString(target.label)}}));
        return;
    }
    const int row = add_table_row(path);
    RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    // A resolution test decodes only the reference; its test video is made in
    // the filter graph. So the reference is the file the saved scores follow,
    // and the target is part of the options.
    row_data.media_path = source_info_->path;
    row_data.options.resample_test = target;
    row_data.video_info = *source_info_;
    set_resample_row_info(row, target);
    // Drawn again now it is a resolution test: drawn when added, its
    // SSIMULACRA2, Butteraugli and CVVDP cells showed tick boxes, not n/a.
    set_row_metrics(row);
    start_cache_lookup({path});
}

void MainWindow::set_resample_row_info(int row, const ResampleTarget& target) {
    // Shows what a resolution test scales the reference down to and back up
    // to.
    const VideoInfo& info = *source_info_;
    const int down_h = resample_down_height(target.width, {info.width, info.height});
    QTableWidgetItem* item = distorted_table_->item(row, col_info);
    item->setText(app::tr("Downscale to {down}, upscale back to {back}",
                          {{"down", size_text(target.width, down_h)}, {"back", size_text(info.width, info.height)}}));
    item->setToolTip(QString::fromStdString(format_hms(info.duration, 1)));
    distorted_table_->item(row, col_bitrate)->setText("N/A");
    set_row_black_bars(row);
    fit_columns({col_info, col_bitrate});
}

void MainWindow::start_media_probe(const std::vector<std::filesystem::path>& paths) {
    // Cancels nothing, on purpose: a probe answers a question about one file,
    // and the answer stays true whatever the user does next.
    if (paths.empty()) {
        return;
    }
    QStringList texts;
    for (const std::filesystem::path& path : paths) {
        texts.append(text_of(path));
    }
    auto* worker = new ProbeWorker(texts, find_tool("ffprobe", settings_), this);
    probe_workers_.push_back(worker);
    connect(worker, &ProbeWorker::probed, this, [this](const Probed& probed) { on_probed(probed); });
    connect(worker, &ProbeWorker::finished_all, this, [this, worker] { on_probe_finished(worker); });
    show_reading(ntr("Reading {count} video...", "Reading {count} videos...", static_cast<long long>(paths.size())));
    worker->start();
}

void MainWindow::on_probed(const Probed& probed) {
    const std::optional<int> row = row_index_of_path(path_of(probed.path));
    if (!row) {
        return;  // removed while the probe was in flight
    }
    set_row_info(*row, probed.info, probed.info ? QString() : probed.error);
}

void MainWindow::on_probe_finished(ProbeWorker* worker) {
    retire(worker);
    if (std::any_of(probe_workers_.begin(), probe_workers_.end(), [](ProbeWorker* w) { return w->isRunning(); })) {
        return;  // another lane is still reading
    }
    if (!run_active_) {
        show_ready();
    }
    on_table_selection_changed();
}

void MainWindow::apply_row_order(const std::vector<int>& order) {
    // The table's cells and the rows move together, and the same videos stay
    // selected. Every other piece of state finds its row by identity or by
    // path, never by position.
    std::vector<int> sorted_order = order;
    std::sort(sorted_order.begin(), sorted_order.end());
    std::vector<int> identity(order.size());
    std::iota(identity.begin(), identity.end(), 0);
    if (sorted_order != identity || order == identity || order.size() != rows_.size()) {
        return;
    }
    QTableWidget* table = distorted_table_;
    std::set<const RowData*> selected;
    for (const QModelIndex& index : table->selectionModel()->selectedRows()) {
        selected.insert(rows_[static_cast<std::size_t>(index.row())].get());
    }
    const QModelIndex current = table->currentIndex();
    const RowData* current_row = current.isValid() ? rows_[static_cast<std::size_t>(current.row())].get() : nullptr;
    const int columns = table->columnCount();
    table->blockSignals(true);
    table->selectionModel()->blockSignals(true);
    syncing_table_ = true;
    std::vector<std::vector<QTableWidgetItem*>> cells(order.size());
    for (int row = 0; row < static_cast<int>(order.size()); ++row) {
        for (int column = 0; column < columns; ++column) {
            cells[static_cast<std::size_t>(row)].push_back(table->takeItem(row, column));
        }
    }
    std::vector<std::shared_ptr<RowData>> reordered;
    for (int position = 0; position < static_cast<int>(order.size()); ++position) {
        const int old = order[static_cast<std::size_t>(position)];
        for (int column = 0; column < columns; ++column) {
            if (QTableWidgetItem* item = cells[static_cast<std::size_t>(old)][static_cast<std::size_t>(column)]) {
                table->setItem(position, column, item);
            }
        }
        reordered.push_back(rows_[static_cast<std::size_t>(old)]);
    }
    rows_ = std::move(reordered);
    table->clearSelection();
    QItemSelection selection;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        if (selected.contains(rows_[static_cast<std::size_t>(row)].get())) {
            selection.select(table->model()->index(row, 0), table->model()->index(row, columns - 1));
        }
    }
    table->selectionModel()->select(selection, QItemSelectionModel::Select);
    if (current_row != nullptr) {
        const auto found = std::find_if(rows_.begin(), rows_.end(),
                                        [current_row](const auto& row) { return row.get() == current_row; });
        table->selectionModel()->setCurrentIndex(
            table->model()->index(static_cast<int>(found - rows_.begin()), current.column()),
            QItemSelectionModel::NoUpdate);
    }
    syncing_table_ = false;
    table->selectionModel()->blockSignals(false);
    table->blockSignals(false);
    table->viewport()->update();
    on_table_selection_changed();
}

void MainWindow::on_rows_dragged(const QList<int>& rows, int target) {
    // Rows dragged within the table go, in their order, before `target`.
    if (run_active_ || rows.isEmpty()) {
        return;
    }
    const std::set<int> moved(rows.begin(), rows.end());
    std::vector<int> rest;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        if (!moved.contains(row)) {
            rest.push_back(row);
        }
    }
    const int at =
        target - static_cast<int>(std::count_if(rows.begin(), rows.end(), [target](int row) { return row < target; }));
    std::vector<int> order(rest.begin(), rest.begin() + at);
    order.insert(order.end(), moved.begin(), moved.end());
    order.insert(order.end(), rest.begin() + at, rest.end());
    std::vector<int> identity(order.size());
    std::iota(identity.begin(), identity.end(), 0);
    if (order != identity) {
        clear_sort_indicator();
    }
    apply_row_order(order);
}

void MainWindow::on_header_clicked(int column) {
    // Sorts the rows by the column whose heading was clicked; the same
    // heading again reverses it. A row without a value goes last whichever
    // way.
    if (run_active_ || rows_.empty()) {
        return;
    }
    const Qt::SortOrder order = column == sort_column_ && sort_order_ == Qt::AscendingOrder ? Qt::DescendingOrder
                                                                                              : Qt::AscendingOrder;
    std::vector<std::optional<SortKey>> keys;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        const std::optional<VideoInfo>& info = row_data.video_info;
        std::optional<SortKey> key;
        if (column == col_path) {
            key = natural_parts(name_of(row_data.path));
        } else if (column == col_info) {
            if (info) {
                key = std::tuple{static_cast<long long>(info->width) * info->height, info->fps, info->duration};
            }
        } else if (column == col_bitrate) {
            if (info && info->bit_rate != 0) {
                key = info->bit_rate;
            }
        } else if (column == col_check) {
            const QTableWidgetItem* item = distorted_table_->item(row, col_check);
            key = item != nullptr && item->checkState() == Qt::Checked ? 0 : 1;
        } else if (column == col_scaling) {
            const QTableWidgetItem* item = distorted_table_->item(row, col_scaling);
            if (item != nullptr && !item->text().isEmpty()) {
                key = item->text().toCaseFolded();
            }
        } else if (metric_column_of(column) != nullptr && row_data.completed_run) {
            // NaN would scramble the sort.
            if (const std::optional<double> score = score_of(*row_data.completed_run, column);
                score && !std::isnan(*score)) {
                key = *score;
            }
        }
        keys.push_back(std::move(key));
    }
    std::vector<int> present;
    std::vector<int> missing;
    for (int row = 0; row < static_cast<int>(keys.size()); ++row) {
        (keys[static_cast<std::size_t>(row)] ? present : missing).push_back(row);
    }
    if (present.empty()) {
        return;  // nothing in that column to sort by
    }
    std::stable_sort(present.begin(), present.end(), [&keys, order](int a, int b) {
        const SortKey& x = *keys[static_cast<std::size_t>(a)];
        const SortKey& y = *keys[static_cast<std::size_t>(b)];
        return order == Qt::DescendingOrder ? sort_less(y, x) : sort_less(x, y);
    });
    sort_column_ = column;
    sort_order_ = order;
    QHeaderView* header = distorted_table_->horizontalHeader();
    header->setSortIndicator(column, order);
    header->setSortIndicatorShown(true);
    present.insert(present.end(), missing.begin(), missing.end());
    apply_row_order(present);
}

void MainWindow::clear_sort_indicator() {
    if (!sort_column_) {
        return;
    }
    sort_column_.reset();
    distorted_table_->horizontalHeader()->setSortIndicatorShown(false);
}

std::optional<int> MainWindow::row_index_of_path(const std::filesystem::path& path) const {
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        if (same_path(rows_[static_cast<std::size_t>(row)]->path, path)) {
            return row;
        }
    }
    return std::nullopt;
}

void MainWindow::on_remove_distorted() {
    std::set<int, std::greater<>> rows;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        rows.insert(index.row());
    }
    for (const int row : rows) {
        distorted_table_->removeRow(row);
        rows_.erase(rows_.begin() + row);
    }
    on_table_selection_changed();
}

void MainWindow::on_remove_all_distorted() {
    if (rows_.empty()) {
        return;
    }
    // Confirmed, because it can discard a lot of completed work from the
    // table at once. The saved scores themselves are kept.
    const auto answer = QMessageBox::question(
        this, app::tr("Remove all videos"),
        ntr("Remove {count} video from the list?\n\nSaved results are kept -- re-adding a video shows its scores "
            "again.",
            "Remove all {count} videos from the list?\n\nSaved results are kept -- re-adding a video shows its "
            "scores again.",
            static_cast<long long>(rows_.size())),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return;
    }
    distorted_table_->setRowCount(0);
    rows_.clear();
    on_table_selection_changed();
}

void MainWindow::on_highlight_toggled(bool on) {
    settings_.highlight_best_worst = on;
    save_settings();
    apply_best_worst();
}

void MainWindow::schedule_best_worst() { best_worst_timer_->start(); }

void MainWindow::apply_best_worst() {
    // Colours each metric's best score green and its worst red, among the
    // test videos' (MetricDefinition::best_and_worst), while Highlight
    // best/worst results is ticked: only where two or more videos have a
    // score to compare. A score from the other implementation than its row
    // asks for, marked "(CPU)" or "(GPU)", is on another scale and is not
    // compared (set_row_metrics gives it no score_role). Unticked, no cell
    // has a colour.
    const bool on = highlight_check_->isChecked();
    const QColor none(0, 0, 0, 0);
    // setData emits itemChanged, as a click's edit does.
    const bool syncing = std::exchange(syncing_table_, true);
    for (const MetricColumn& column : metric_columns()) {
        std::vector<QTableWidgetItem*> items;
        std::vector<std::optional<double>> scores;
        for (int row = 0; row < distorted_table_->rowCount(); ++row) {
            if (QTableWidgetItem* item = distorted_table_->item(row, column.column)) {
                const QVariant score = item->data(score_role);
                items.push_back(item);
                scores.push_back(score.isValid() ? std::optional(score.toDouble()) : std::nullopt);
            }
        }
        const std::vector<ScoreMark> marks = on ? metric_definition(column.key).best_and_worst(scores)
                                                : std::vector<ScoreMark>(items.size(), ScoreMark::none);
        for (std::size_t index = 0; index < items.size(); ++index) {
            items[index]->setData(Qt::BackgroundRole, marks[index] == ScoreMark::best    ? theme::color("best")
                                                      : marks[index] == ScoreMark::worst ? theme::color("worst")
                                                                                         : none);
        }
    }
    syncing_table_ = syncing;
}

// ------------------------------------------------------------- the metrics

bool MainWindow::default_metric_ticked(int column) const {
    // Whether new rows start with this metric ticked. SSIMULACRA2,
    // Butteraugli and CVVDP are not in VmafOptions: reading only the options,
    // their headings would show unticked even when the metric is on.
    const MetricColumn* metric = metric_column_of(column);
    return metric_enabled(default_options_, column) || default_extra_metric_keys_.contains(metric->key);
}

bool MainWindow::metric_enabled(const VmafOptions& options, int column) {
    const char* key = metric_column_of(column)->key;
    return metric_definition(key).ffmpeg_binding.has_value() && options.metric_enabled(key);
}

void MainWindow::set_metric_option(VmafOptions& options, int column, bool checked) {
    options.set_metric_enabled(metric_column_of(column)->key, checked);
}

std::vector<std::string> MainWindow::selected_metrics(const RowData& row_data) {
    // The row's tick boxes, metrics whose column is hidden included, in the
    // registry's order.
    const std::vector<std::string> ffmpeg = row_data.options.requested_metrics();
    std::vector<std::string> selected;
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        if (std::find(ffmpeg.begin(), ffmpeg.end(), key) != ffmpeg.end() || row_data.extra_metric_keys.contains(key)) {
            selected.push_back(key);
        }
    }
    return selected;
}

std::vector<std::string> MainWindow::requested_metrics(const RowData& row_data) const {
    // What a run of this row calculates: its ticks, less hidden metrics and
    // metrics the row's kind of comparison cannot produce. Hiding a metric
    // leaves the row's tick untouched, so showing it again restores it.
    std::vector<std::string> requested;
    for (const std::string& key : selected_metrics(row_data)) {
        if (!hidden_metrics_.contains(key) && !metric_unavailable_reason(row_data, key)) {
            requested.push_back(key);
        }
    }
    return requested;
}

std::optional<QString> MainWindow::metric_unavailable_reason(const RowData& row_data, const std::string& key) {
    // No perceptual metric takes a resolution test. CVVDP needs every frame,
    // as it judges each one together with the frames before it, and a
    // supported GPU, as the app has no CPU version of it.
    if (row_data.options.resample_test && metric_definition(key).backend_id == "perceptual") {
        return app::tr("{label} is not available for resolution round-trip tests; the other selected metrics are still "
                       "calculated.",
                       {{"label", qt_text(metric_definition(key).label)}});
    }
    if (key == "cvvdp") {
        if (row_data.options.n_subsample > 1) {
            return app::tr("CVVDP judges each frame together with the frames before it, so it needs every frame: it is "
                           "not available while libvmaf frame subsample is above 1.");
        }
        if (!vship_available()) {
            return app::tr("CVVDP is calculated on the GPU only, and no GPU that Vship can use was found.");
        }
    }
    return std::nullopt;
}

bool MainWindow::row_metric_enabled(const RowData& row_data, int column) {
    const std::vector<std::string> selected = selected_metrics(row_data);
    return std::find(selected.begin(), selected.end(), metric_column_of(column)->key) != selected.end();
}

void MainWindow::apply_metric_visibility() {
    for (const MetricColumn& item : metric_columns()) {
        distorted_table_->setColumnHidden(item.column, hidden_metrics_.contains(item.key));
    }
}

void MainWindow::show_metrics_menu(const QPoint& position) {
    if (run_active_) {
        return;
    }
    auto* menu = new StayOpenMenu(this);
    menu->setAttribute(Qt::WA_DeleteOnClose);
    menu->setToolTipsVisible(true);
    const std::size_t shown = metric_columns().size() - hidden_metrics_.size();
    for (const MetricColumn& item : metric_columns()) {
        QAction* action = menu->addAction(qt_text(metric_definition(item.key).label));
        action->setCheckable(true);
        action->setChecked(!hidden_metrics_.contains(item.key));
        // The last visible metric cannot be hidden: a table with no metrics
        // could calculate nothing.
        if (shown == 1 && !hidden_metrics_.contains(item.key)) {
            action->setEnabled(false);
            action->setToolTip(app::tr("At least one metric must stay shown."));
        }
        const std::string key = item.key;
        connect(action, &QAction::toggled, this,
                [this, key, menu](bool checked) { on_metric_visibility_toggled(key, checked, menu); });
    }
    menu->exec(position);
}

void MainWindow::on_metric_visibility_toggled(const std::string& key, bool shown, QMenu* menu) {
    // Hides or shows a metric's column, keeping at least one shown, and saves
    // the choice.
    if (shown) {
        hidden_metrics_.erase(key);
    } else {
        if (hidden_metrics_.size() + 1 >= metric_columns().size()) {
            return;
        }
        hidden_metrics_.insert(key);
    }
    settings_.hidden_metrics.clear();
    for (const MetricColumn& item : metric_columns()) {
        if (hidden_metrics_.contains(item.key)) {
            settings_.hidden_metrics.emplace_back(item.key);
        }
    }
    settings_.save();
    apply_metric_visibility();
    if (menu != nullptr) {
        int visible = 0;
        for (const QAction* action : menu->actions()) {
            visible += action->isChecked() ? 1 : 0;
        }
        for (QAction* action : menu->actions()) {
            action->setEnabled(!(visible == 1 && action->isChecked()));
        }
    }
    std::vector<int> every_row;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        set_row_metrics(row);
        every_row.push_back(row);
    }
    if (shown) {
        // Scores saved for the metric while it was hidden come back.
        reload_saved_scores(every_row);
    }
}

void MainWindow::apply_metric_selection(const std::vector<int>& rows, int column, bool checked, bool set_default) {
    // Turns one metric on or off for `rows`. `set_default` carries the choice
    // to rows added later: true for the header shortcut, a statement about
    // the whole table; false for a tick in one row's own cell.
    if (syncing_panel_ || run_active_) {
        return;
    }
    const std::string key = metric_column_of(column)->key;
    const bool ffmpeg = metric_definition(key).ffmpeg_binding.has_value();
    for (const int row : rows) {
        RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        if (!ffmpeg) {
            if (checked) {
                row_data.extra_metric_keys.insert(key);
            } else {
                row_data.extra_metric_keys.erase(key);
            }
        } else {
            set_metric_option(row_data.options, column, checked);
        }
        row_data.analysis_status.reset();
        set_row_metrics(row);
    }
    if (set_default) {
        if (!ffmpeg) {
            if (checked) {
                default_extra_metric_keys_.insert(key);
            } else {
                default_extra_metric_keys_.erase(key);
            }
        } else {
            set_metric_option(default_options_, column, checked);
        }
        // Saved as the default too, the same setting as Settings > Default
        // metrics.
        settings_.*default_compute_field(key) = checked;
        save_settings();
    }
    reload_saved_scores(rows);
    on_table_selection_changed();
}

void MainWindow::on_table_item_changed(QTableWidgetItem* item) {
    // A metric tick box in the table was clicked. It applies to every
    // selected row when the clicked row is one of them, so ticking PSNR
    // across a selection is one click; otherwise to the clicked row alone.
    if (syncing_table_) {
        return;
    }
    const int column = item->column();
    if (metric_column_of(column) == nullptr) {
        return;
    }
    const int row = item->row();
    if (row >= static_cast<int>(rows_.size())) {
        return;  // a row still being built: its boxes are filled in after
    }
    if (!(item->flags() & Qt::ItemIsUserCheckable)) {
        return;  // a cell showing a score or live progress, not a tick box
    }
    const bool checked = item->checkState() == Qt::Checked;
    if (checked == row_metric_enabled(*rows_[static_cast<std::size_t>(row)], column)) {
        // itemChanged also fires for text, colour and font edits: only a box
        // that now disagrees with the row it stands for is a click.
        return;
    }
    if (run_active_) {
        set_row_metrics(row);  // nothing may change mid-run
        return;
    }
    std::set<int> selected;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        selected.insert(index.row());
    }
    const std::vector<int> rows = selected.contains(row) ? std::vector<int>(selected.begin(), selected.end())
                                                         : std::vector<int>{row};
    apply_metric_selection(rows, column, checked, false);
}

void MainWindow::on_metric_column_toggled(int column, bool checked) {
    // The header's shortcuts apply to every row.
    std::vector<int> rows(rows_.size());
    std::iota(rows.begin(), rows.end(), 0);
    apply_metric_selection(rows, column, checked, true);
}

// -------------------------------------------------------------- the scores

std::vector<std::string> MainWindow::reusable_metric_keys(const RowData& row_data) const {
    // The requested metrics the row already has and a run can keep
    // (metric_cache.h's reusable_keys, the command line's rule too).
    if (!row_data.completed_run) {
        return {};
    }
    std::vector<ColourTags> infos;
    for (const std::optional<VideoInfo>* info : {&source_info_, &row_data.video_info}) {
        if (*info) {
            infos.push_back({(*info)->pix_fmt, (*info)->color_transfer});
        }
    }
    return reusable_keys(row_data.completed_run->result,
                         analysis_request(row_data.options, requested_metrics(row_data), row_data.metric_backends,
                                          row_data.cvvdp)
                             .metrics,
                         row_data.metric_backends, [](const std::string& key) { return gpu_can_score(key); }, infos);
}

MetricResultSet MainWindow::reusable_results(const RowData& row_data) const {
    // Those metrics' scores, for a run to keep.
    MetricResultSet kept;
    for (const std::string& key : reusable_metric_keys(row_data)) {
        const MetricResultSet& scores = row_data.completed_run->result.metric_results;
        if (const SequenceMetricResult* sequence = scores.sequence(key)) {
            kept.add(*sequence);
        } else {
            kept.add(*scores.frame(key));
        }
    }
    return kept;
}

bool MainWindow::has_requested_results(const RowData& row_data) const {
    // Whether the row already has a result for every metric it asks for.
    const std::vector<std::string> requested = requested_metrics(row_data);
    if (requested.empty()) {
        return false;
    }
    const std::vector<std::string> kept = reusable_metric_keys(row_data);
    return std::all_of(requested.begin(), requested.end(), [&kept](const std::string& key) {
        return std::find(kept.begin(), kept.end(), key) != kept.end();
    });
}

bool MainWindow::backend_matches(const RowData& row_data, const std::string& key,
                                 const MetricProvenance& provenance) {
    // Whether a score was made the way the row asks for it (metric_cache.h's
    // made_as_chosen); a metric without a GPU or CPU choice always is.
    const auto choice = row_data.metric_backends.find(key);
    return made_as_chosen(key, choice == row_data.metric_backends.end() ? std::nullopt : std::optional(choice->second),
                          provenance.compute_backend, [](const std::string& metric) { return gpu_can_score(metric); });
}

QString MainWindow::backend_note(const RowData& row_data, const std::string& key, const FrameMetricResult* metric) {
    // The tooltip's word on a SSIMULACRA2 or Butteraugli score calculated on
    // the other device than the row's choice, the rare case marked "(CPU)" or
    // "(GPU)" in the cell. A score calculated where chosen needs none.
    const auto choice = row_data.metric_backends.find(key);
    if (choice == row_data.metric_backends.end() || metric == nullptr ||
        backend_matches(row_data, key, metric->provenance)) {
        return {};
    }
    return app::tr("\n\nCalculated on the {produced}, but this row is set to {wanted}; the two give different numbers, "
                   "so the next run recalculates it on the {wanted}.",
                   {{"produced", QString::fromStdString(metric->provenance.compute_backend).toUpper()},
                    {"wanted", choice->second == "gpu" ? "GPU" : "CPU"}});
}

std::optional<double> MainWindow::score_of(CompletedRun& run, int column) {
    // A metric column's score: CVVDP's one for the whole video, the others'
    // mean of their frames. Worked out once a run: each time, a two-hour
    // video's took 0.6 ms a metric, at each of the refreshes of its row.
    if (const auto known = run.scores.find(column); known != run.scores.end()) {
        return known->second;
    }
    const std::string key = metric_column_of(column)->key;
    std::optional<double> score;
    if (const SequenceMetricResult* sequence = run.result.metric_results.sequence(key)) {
        score = sequence->score;
    } else if (const FrameMetricResult* frame = run.result.metric_results.frame(key)) {
        score = frame->aggregate();
    }
    run.scores.emplace(column, score);
    return score;
}

QString MainWindow::identical_frames_note(CompletedRun& run, int column) {
    // Says when frames were identical to the reference and scored infinity:
    // without it, the mean silently describes fewer frames than the run
    // measured. Counted once a run.
    auto counted = run.identical.find(column);
    if (counted == run.identical.end()) {
        std::pair<long long, long long> found{0, 0};
        if (const FrameMetricResult* frame = run.result.metric_results.frame(metric_column_of(column)->key)) {
            found = {std::count_if(frame->values.begin(), frame->values.end(),
                                   [](float value) { return std::isinf(value) && value > 0; }),
                     static_cast<long long>(frame->values.size())};
        }
        counted = run.identical.emplace(column, found).first;
    }
    const auto [identical, count] = counted->second;
    if (identical == 0) {
        return {};
    }
    return app::tr("\n\n{identical} of {count} frames were identical to the reference and scored infinity; they "
                   "contribute zero distortion and are included in the frame count for the XPSNR sequence average.",
                   {{"identical", identical}, {"count", count}});
}

QString MainWindow::cvvdp_note(const RowData& row_data) const {
    // A CVVDP score's tooltip: what the number means, and which display it
    // is for.
    const std::optional<CvvdpPreset> preset =
        matching_cvvdp_preset(row_data.cvvdp, settings_.cvvdp_presets_json, row_data.cvvdp_preset);
    return app::tr("CVVDP of the whole video, in JOD (just-objectionable differences): 10 means no visible "
                   "difference, and one JOD lower means 75% of viewers would pick the reference as better.\n\nFor "
                   "the display {preset}: {display}",
                   {{"preset", preset ? preset_label(*preset) : app::tr("(custom)")},
                    {"display", display_description(row_data.cvvdp.display)}}) +
           (row_data.cvvdp.resize_to_display ? app::tr(", video scaled to fill it") : QString()) +
           app::tr(".\n\nThe JOD of each second is plotted in Metric Graphs.");
}

}  // namespace vml::app
