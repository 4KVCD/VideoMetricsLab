#include "settings_panel.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QFontMetrics>
#include <QRawFont>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QStyle>
#include <QStyleOptionComboBox>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

#include "file_writes.h"
#include "formatting.h"
#include "i18n.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/crop_detect.h"
#include "vml/cvvdp.h"
#include "vml/metric_cache.h"
#include "vml/platform/system.h"
#include "vml/vship.h"

// The panel's text is app::tr's: inside a Q_OBJECT class, tr() is Qt's.
namespace vml::app {

namespace {

const app_log::Logger logger("vml.settings_panel");

// The "Default metrics" check boxes, in the order they are shown, the
// Videos table's and Metric Graphs': the metric's key and its label.
const std::vector<std::pair<std::string, QString>> default_metric_boxes{
    {"vmaf", "VMAF v0.6.1"},
    {"vmaf_neg", "VMAF NEG"},
    {"vmaf_v1", "VMAF v1"},
    {"psnr", "PSNR"},
    {"ssim", "SSIM"},
    {"xpsnr", "XPSNR"},
    {"ssimulacra2", "SSIMULACRA2"},
    {"butteraugli", "Butteraugli"},
    {"cvvdp", "CVVDP"},
};

QString text_of(const std::filesystem::path& path) { return QString::fromStdString(platform::path_text(path)); }

QString text_of(const std::string& text) { return QString::fromStdString(text); }

}  // namespace

SettingsPanel::SettingsPanel(Settings& settings, FileWrites& file_writes, std::function<ToolsStatus()> tools,
                             const Dialogs& dialogs, QWidget* parent)
    : QWidget(parent),
      settings_(settings),
      file_writes_(file_writes),
      tools_(std::move(tools)),
      dialogs_(dialogs),
      counting_(new QFutureWatcher<CacheSummary>(this)) {
    connect(counting_, &QFutureWatcher<CacheSummary>::finished, this, [this] { show_cache_summary(); });
    connect(&file_writes_, &FileWrites::became_idle, this, [this] { on_file_writes_idle(); });
    auto* layout = new QVBoxLayout(this);
    for (QGroupBox* box : {build_ffmpeg_box(), build_storage_box(), build_defaults_box(), build_gpu_box(),
                           build_compare_box(), build_window_box()}) {
        layout->addWidget(box);
    }
    layout->addStretch(1);
    status = new QLabel;
    theme::style(status, "color: {muted};");
    layout->addWidget(status);
    // The tools and saved results are filled in when the tab is shown
    // (refresh_status).
}

SettingsPanel::~SettingsPanel() = default;

QGroupBox* SettingsPanel::build_ffmpeg_box() {
    auto* tools_box = new QGroupBox("ffmpeg");
    auto* tools_form = new QFormLayout(tools_box);
    ffmpeg_edit = new QLineEdit(text_of(settings_.ffmpeg_dir));
    ffmpeg_edit->setPlaceholderText(app::tr("blank = search PATH"));
    connect(ffmpeg_edit, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    auto* browse_ffmpeg = new QPushButton(app::tr("Browse..."));
    connect(browse_ffmpeg, &QPushButton::clicked, this,
            [this] { pick_directory(app::with_program_names(app::tr("Select the folder containing ffmpeg.exe")), ffmpeg_edit); });
    auto* ffmpeg_row = new QHBoxLayout;
    ffmpeg_row->addWidget(ffmpeg_edit, 1);
    ffmpeg_row->addWidget(browse_ffmpeg);
    tools_form->addRow(app::tr("ffmpeg folder:"), ffmpeg_row);
    ffmpeg_status = new QLabel;
    tools_form->addRow("", ffmpeg_status);
    return tools_box;
}

QGroupBox* SettingsPanel::build_storage_box() {
    auto* storage_box = new QGroupBox(app::tr("Storage"));
    auto* storage_form = new QFormLayout(storage_box);
    cache_edit = new QLineEdit(text_of(settings_.cache_dir));
    cache_edit->setPlaceholderText(app::tr("blank = shared user-profile folder"));
    connect(cache_edit, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    auto* browse_cache = new QPushButton(app::tr("Browse..."));
    connect(browse_cache, &QPushButton::clicked, this,
            [this] { pick_directory(app::tr("Where should saved results be kept?"), cache_edit); });
    auto* open_cache = new QPushButton(app::tr("Open"));
    connect(open_cache, &QPushButton::clicked, this, [this] {
        const std::filesystem::path folder = cache_folder();
        std::error_code ignored;
        std::filesystem::create_directories(folder, ignored);
        QDesktopServices::openUrl(QUrl::fromLocalFile(text_of(folder)));
    });
    auto* cache_row = new QHBoxLayout;
    cache_row->addWidget(cache_edit, 1);
    cache_row->addWidget(browse_cache);
    cache_row->addWidget(open_cache);
    storage_form->addRow(app::tr("Saved results:"), cache_row);

    cache_summary = new QLabel;
    auto* clear_cache = new QPushButton(app::tr("Clear saved results"));
    connect(clear_cache, &QPushButton::clicked, this, [this] { clear_saved_results(); });
    auto* summary_row = new QHBoxLayout;
    summary_row->addWidget(cache_summary, 1);
    summary_row->addWidget(clear_cache);
    storage_form->addRow("", summary_row);

    export_edit = new QLineEdit(text_of(settings_.export_dir));
    export_edit->setPlaceholderText(app::tr("blank = ask each time"));
    connect(export_edit, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    auto* browse_export = new QPushButton(app::tr("Browse..."));
    connect(browse_export, &QPushButton::clicked, this,
            [this] { pick_directory(app::tr("Where should exports be written?"), export_edit); });
    auto* export_row = new QHBoxLayout;
    export_row->addWidget(export_edit, 1);
    export_row->addWidget(browse_export);
    storage_form->addRow(app::tr("Export folder:"), export_row);

    use_cache = new QCheckBox(app::tr("Reuse a saved result when a video is added again"));
    use_cache->setChecked(settings_.use_cache);
    connect(use_cache, &QCheckBox::toggled, this, [this] { on_edited(); });
    storage_form->addRow("", use_cache);

    // Where each session's log goes: runs, steps and every failure in full,
    // kept after the window that showed them is closed.
    log_label = new QLabel(text_of(app_log::log_dir()));
    log_label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    log_label->setToolTip(
        app::tr("Each session's setup, every run and its steps, and every failure with its full text. "
                "Send these files along when reporting a problem."));
    auto* open_logs = new QPushButton(app::tr("Open"));
    connect(open_logs, &QPushButton::clicked, this, [] {
        const std::filesystem::path folder = app_log::log_dir();
        std::error_code ignored;
        std::filesystem::create_directories(folder, ignored);
        QDesktopServices::openUrl(QUrl::fromLocalFile(text_of(folder)));
    });
    export_log_button = new QPushButton(app::tr("Export log..."));
    export_log_button->setToolTip(
        app::tr("Save the log files as one .zip file, to attach when reporting a problem."));
    connect(export_log_button, &QPushButton::clicked, this, [this] { on_export_log(); });
    copy_log_button = new QPushButton(app::tr("Copy log"));
    copy_log_button->setToolTip(app::tr(
        "Copy the latest session that calculated metrics to the clipboard, to paste into a post or chat."));
    connect(copy_log_button, &QPushButton::clicked, this, [this] { on_copy_log(); });
    auto* log_row = new QHBoxLayout;
    log_row->addWidget(log_label, 1);
    log_row->addWidget(open_logs);
    log_row->addWidget(copy_log_button);
    log_row->addWidget(export_log_button);
    storage_form->addRow(app::tr("Log files:"), log_row);
    return storage_box;
}

QGroupBox* SettingsPanel::build_defaults_box() {
    auto* defaults_box = new QGroupBox(app::tr("Defaults for newly added videos"));
    auto* defaults_layout = new QVBoxLayout(defaults_box);
    auto* hint = new QLabel(app::tr("These are only a starting point -- each video's own settings are "
                                    "edited in the Videos tab."));
    theme::style(hint, "color: {muted}; font-style: italic;");
    defaults_layout->addWidget(hint);
    default_gpu = new QCheckBox(app::tr("Use GPU decoding"));
    default_gpu->setChecked(settings_.default_gpu_decode);
    connect(default_gpu, &QCheckBox::toggled, this, [this] { on_edited(); });
    defaults_layout->addWidget(default_gpu);

    auto* metrics_row = new QHBoxLayout;
    metrics_row->addWidget(new QLabel(app::tr("Default metrics:")));
    for (const auto& [key, label] : default_metric_boxes) {
        auto* box = new QCheckBox(label);
        box->setChecked(*settings_.default_compute(key));
        connect(box, &QCheckBox::toggled, this, [this] { on_edited(); });
        default_metrics[key] = box;
        metrics_row->addWidget(box);
    }
    metrics_row->addStretch(1);
    defaults_layout->addLayout(metrics_row);

    auto* cvvdp_row = new QHBoxLayout;
    cvvdp_row->addWidget(new QLabel(app::tr("CVVDP display:")));
    cvvdp_default = new QComboBox;
    cvvdp_default->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    cvvdp_default->setToolTip(app::tr("The display CVVDP models for newly added videos. Automatic: the SDR or the "
                                      "HDR display, as the reference video is. Saving a preset of your own in the "
                                      "Options panel makes it the default; choose here to change that."));
    fill_cvvdp_default_combo();
    connect(cvvdp_default, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    cvvdp_row->addWidget(cvvdp_default);
    cvvdp_row->addStretch(1);
    defaults_layout->addLayout(cvvdp_row);
    return defaults_box;
}

QGroupBox* SettingsPanel::build_gpu_box() {
    auto* gpu_box = new QGroupBox(app::tr("GPU metrics"));
    auto* gpu_layout = new QVBoxLayout(gpu_box);
    if (vship51_bundled()) {
        // The entries' data: Settings' default_vship_build values.
        auto* build_row = new QHBoxLayout;
        build_row->addWidget(new QLabel(app::tr("New videos use:")));
        vship_build_default = new QComboBox;
        vship_build_default->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        vship_build_default->addItem("GPU - Vship-fast", QString::fromUtf8(vship_build_value(VshipBuild::fast)));
        vship_build_default->addItem("GPU - Vship", QString::fromUtf8(vship_build_value(VshipBuild::stock)));
        vship_build_default->setCurrentIndex(
            default_vship_build(settings_.default_vship_build) == VshipBuild::stock ? 1 : 0);
        vship_build_default->setToolTip(app::tr(
            "The Vship that calculates SSIMULACRA2, Butteraugli and CVVDP for newly added videos, and on the command "
            "line without --vship: Vship-fast, the app's own build, or stock Vship 5.1.2. Each video's own is set in "
            "the Options panel (Vship compute)."));
        connect(vship_build_default, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
        build_row->addWidget(vship_build_default);
        build_row->addStretch(1);
        gpu_layout->addLayout(build_row);
    }
    gpu_together = new QCheckBox(app::tr("Calculate SSIMULACRA2, Butteraugli and CVVDP together, in one pass per video"));
    gpu_together->setToolTip(app::tr(
        "Each video is decoded once for all three instead of once for each. This is faster only on very powerful "
        "GPUs, and needs a lot of GPU memory. A metric that fails in the shared pass, out of GPU memory for example, "
        "is calculated again on its own.\n\nThe scores are the same either way. Applies from the next run."));
    gpu_together->setChecked(settings_.gpu_metrics_together);
    connect(gpu_together, &QCheckBox::toggled, this, [this] { on_edited(); });
    gpu_layout->addWidget(gpu_together);
    auto* gpu_note = new QLabel(
        app::tr("Unticked, the three are calculated one at a time, which needs much less GPU memory and is hardly any "
                "slower on most GPUs."));
    gpu_note->setWordWrap(true);
    theme::style(gpu_note, "color: {muted};");
    gpu_layout->addWidget(gpu_note);
    return gpu_box;
}

QGroupBox* SettingsPanel::build_compare_box() {
    auto* compare_box = new QGroupBox(app::tr("Video Compare"));
    auto* compare_layout = new QVBoxLayout(compare_box);
    // A dropdown, not a spin box: 1 to 9 is one click away.
    decoded_videos = new QComboBox;
    for (int count = 1; count <= 9; ++count) {
        decoded_videos->addItem(QString::number(count), count);
    }
    decoded_videos->setCurrentIndex(std::max(0, decoded_videos->findData(settings_.compare_decoded_videos)));
    // As wide as its widest entry and no wider: a form row would otherwise
    // stretch a one-digit dropdown across the window.
    decoded_videos->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    decoded_videos->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    decoded_videos->setToolTip(app::tr("How many test videos keep decoding while one is shown, so the "
                                       "left and right arrows switch to a video that is already running: "
                                       "the selected one, then the one to its right, its left, the next "
                                       "right, the next left, and so on. Each is a running GPU decoder "
                                       "(about 250 MB of RAM for 4K), plus the source. Takes effect "
                                       "immediately, even during playback."));
    connect(decoded_videos, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    // The dropdown first, at the left edge, with its label beside it, as the
    // check boxes above sit. A form row would put the field wherever the
    // label column ends.
    auto* decoded_row = new QHBoxLayout;
    decoded_row->addWidget(decoded_videos);
    decoded_row->addWidget(new QLabel(app::tr("Number of test videos decoded simultaneously for fast comparison "
                                              "switching")));
    decoded_row->addStretch(1);
    compare_layout->addLayout(decoded_row);
    return compare_box;
}

QGroupBox* SettingsPanel::build_window_box() {
    auto* window_box = new QGroupBox(app::tr("Window"));
    auto* window_layout = new QVBoxLayout(window_box);
    // Each language in its own name, so it can be found by someone who
    // cannot read the one the window is in.
    language = new QComboBox;
    const QString windows = windows_language();
    QString windows_name = "English";
    for (const Language& item : languages()) {
        if (item.code == windows) {
            windows_name = item.name;
        }
    }
#ifdef __APPLE__
    language->addItem(app::tr("Same as macOS ({language})", {{"language", windows_name}}), "");
#else
    language->addItem(app::tr("Same as Windows ({language})", {{"language", windows_name}}), "");
#endif
    for (const Language& item : languages()) {
        language->addItem(item.name, item.code);
    }
    language->setCurrentIndex(std::max(0, language->findData(text_of(settings_.language))));
    // As wide as its widest name, measuring only the names the window's own
    // font has the letters for, the one shown, and "Same as Windows", which
    // is in the window's language. Measuring every name, as AdjustToContents
    // does, makes Qt look for fonts for Chinese, Japanese, Korean and Thai,
    // which reads every font on the PC: some 22 MB for as long as the app
    // runs, though the list may never be opened. Those names are short.
    // (QFontMetrics::inFontUcs4 would look for the fonts too; the raw font is
    // the window's font alone.)
    language->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    language->setMinimumContentsLength(1);
    const QFontMetrics metrics = language->fontMetrics();
    const QRawFont own = QRawFont::fromFont(language->font());
    int widest = 0;
    for (int index = 0; index < language->count(); ++index) {
        const QString name = language->itemText(index);
        const QList<uint> letters = name.toUcs4();
        if (index == 0 || index == language->currentIndex() ||
            std::all_of(letters.begin(), letters.end(), [&own](uint letter) { return own.supportsCharacter(letter); })) {
            widest = std::max(widest, metrics.horizontalAdvance(name));
        }
    }
    QStyleOptionComboBox option;
    option.initFrom(language);
    language->setMinimumWidth(
        language->style()->sizeFromContents(QStyle::CT_ComboBox, &option, QSize(widest, metrics.height()), language)
            .width());
    language->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    language->setToolTip(app::tr("The language of the window. Takes effect when the app is next started. The log, "
                                 "saved results and exported files stay in English."));
    connect(language, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    auto* language_row = new QHBoxLayout;
    language_row->addWidget(new QLabel(app::tr("Language:")));
    language_row->addWidget(language);
    language_row->addStretch(1);
    window_layout->addLayout(language_row);
    // Light or dark, at once; "Same as Windows" (macOS) also follows the
    // system changing while the app is open.
    theme = new QComboBox;
#ifdef __APPLE__
    theme->addItem(app::tr("Same as macOS"), "");
#else
    theme->addItem(app::tr("Same as Windows"), "");
#endif
    theme->addItem(app::tr("Light"), "light");
    theme->addItem(app::tr("Dark"), "dark");
    theme->setCurrentIndex(std::max(0, theme->findData(text_of(settings_.theme))));
    theme->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    theme->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    connect(theme, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    auto* theme_row = new QHBoxLayout;
    theme_row->addWidget(new QLabel(app::tr("Theme:")));
    theme_row->addWidget(theme);
    theme_row->addStretch(1);
    window_layout->addLayout(theme_row);
    remember_size = new QCheckBox(app::tr("Reopen at the size the window was last closed at"));
    remember_size->setChecked(settings_.remember_window_size);
    connect(remember_size, &QCheckBox::toggled, this, [this] { on_edited(); });
    window_layout->addWidget(remember_size);
    remember_videos = new QCheckBox(app::tr("Reopen the videos that were open when the app was last closed"));
    remember_videos->setToolTip(
        app::tr("The reference and the test videos, those still there. Resolution tests are not reopened."));
    remember_videos->setChecked(settings_.remember_videos);
    connect(remember_videos, &QCheckBox::toggled, this, [this] { on_edited(); });
    window_layout->addWidget(remember_videos);
    check_updates = new QCheckBox(app::tr("Check GitHub for a newer version when the app starts"));
    check_updates->setToolTip(app::tr("Once, at startup, asks GitHub's public list of releases whether there is a "
                                      "newer one, and says so only if there is. Nothing is sent but the request; "
                                      "GitHub sees your address, as it would for any page."));
    check_updates->setChecked(settings_.check_for_updates);
    connect(check_updates, &QCheckBox::toggled, this, [this] { on_edited(); });
    window_layout->addWidget(check_updates);
    return window_box;
}

// ------------------------------------------------------------- for the window

void SettingsPanel::refresh_status() {
    const ToolsStatus tools = tools_();
    const std::vector<std::string> problems = tools.problems();
    QStringList said;
    for (const std::string& problem : problems) {
        said.append(tr_message(text_of(problem)));
    }
    ffmpeg_status->setText(problems.empty() ? app::tr("ffmpeg {version} and ffprobe found.",
                                                      {{"version", text_of(format_version(tools.ffmpeg.version))}})
                                            : said.join(' '));
    theme::style(ffmpeg_status, problems.empty() ? "color: {good};" : "color: {failed};");

    // Counted on a thread of its own; a count asked for later replaces one
    // still running, whose answer is then never shown.
    counted_folder_ = cache_folder();
    counting_->setFuture(QtConcurrent::run([folder = counted_folder_] { return vml::cache_summary(folder); }));
}

void SettingsPanel::fill_cvvdp_default_combo() {
    const QSignalBlocker blocker(cvvdp_default);
    cvvdp_default->clear();
    cvvdp_default->addItem(automatic_cvvdp_label(), QString(""));  // saved as "", which no preset is called
    for (const CvvdpPreset& preset : cvvdp_presets(settings_.cvvdp_presets_json)) {
        cvvdp_default->addItem(preset_label(preset), text_of(preset.name));
    }
    // A preset that is gone is Automatic, as the window takes it.
    cvvdp_default->setCurrentIndex(std::max(0, cvvdp_default->findData(text_of(settings_.cvvdp_default_preset))));
}

void SettingsPanel::show_ffmpeg_folder(const QString& directory) { ffmpeg_edit->setText(directory); }

void SettingsPanel::show_default_metric(const std::string& key, bool checked) {
    if (const auto found = default_metrics.find(key); found != default_metrics.end()) {
        const QSignalBlocker blocker(found->second);
        found->second->setChecked(checked);
    }
}

void SettingsPanel::clear_saved_results() {
    const std::filesystem::path directory = cache_folder();
    // The count the tab shows, made off the window's thread
    // (refresh_status): counting a large folder here would freeze the
    // window. While it is still being made, the question is asked anyway.
    const bool counted = counting_->isFinished() && counting_->future().resultCount() > 0 &&
                         counted_folder_ == directory;
    if (counted && counting_->result().comparisons == 0 && file_writes_.idle()) {
        status->setText(app::tr("There are no saved results to clear."));
        return;
    }
    if (cleared_) {
        status->setText(app::tr("Saved results are already being cleared."));
        return;
    }
    if (!dialogs_.ask(app::tr("Clear saved results"),
                      app::tr("Delete saved results from {directory}?\n\nVideos already scored will have to be "
                              "recomputed.",
                              {{"directory", text_of(directory)}}))) {
        return;
    }
    // Saved scores are written and deleted through the same queue, one after
    // another. If a run just finished, its scores therefore land before they
    // are cleared, rather than coming back after the user cleared them all.
    // `directory` is the folder the question just named: if the setting
    // changes before the queue gets to this, it must not delete a folder the
    // user was never asked about.
    cleared_ = std::make_shared<int>(0);
    file_writes_.submit("clear saved results", [directory, removed = cleared_] {
        // The black bars kept with the saved results go too: cleared, they
        // start afresh.
        forget_crop_answers();
        std::error_code error;
        const std::filesystem::path black_bars = directory / black_bars_file;
        std::filesystem::remove(black_bars, error);
        if (error) {
            app_log::write(app_log::Level::warning, "vml.crop_detect",
                           "Could not delete " + platform::path_text(black_bars) + ": " + error.message());
        }
        *removed = clear_all(directory);
    });
    status->setText(app::tr("Clearing saved results..."));
}

// --------------------------------------------------------------- inside the tab

std::filesystem::path SettingsPanel::cache_folder() const {
    return settings_.cache_dir_path().value_or(default_cache_dir());
}

void SettingsPanel::show_cache_summary() {
    const CacheSummary summary = counting_->result();
    cache_summary->setText(ntr("{count} saved result, {size:.1f} MB in {directory}",
                               "{count} saved results, {size:.1f} MB in {directory}", summary.comparisons,
                               {{"size", static_cast<double>(summary.bytes) / 1048576.0},
                                {"directory", text_of(counted_folder_)}}));
}

void SettingsPanel::on_file_writes_idle() {
    // Once the clearing has run: a write queued before it and finished just
    // as it was asked for says the queue is idle too, too early.
    if (!cleared_ || !file_writes_.idle()) {
        return;
    }
    const int removed = *cleared_;
    cleared_.reset();
    status->setText(ntr("Removed {count} saved result.", "Removed {count} saved results.", removed));
    refresh_status();
    emit saved_results_cleared();
}

void SettingsPanel::on_edited() {
    const Settings before = settings_;  // to see what this edit changed
    read_controls();
    emit decoded_videos_changed(settings_.compare_decoded_videos);
    apply_global_changes(before);
    // Only the starting point for new rows; rows already in the table keep
    // what they were given.
    emit defaults_changed();
    const std::optional<std::string> error = settings_.save();
    if (settings_.ffmpeg_dir != before.ffmpeg_dir) {
        emit ffmpeg_folder_changed();  // after the save: the tools are looked for in the folder saved
    }
    status->setText(error                                     ? tr_message(text_of(*error))
                    : settings_.language != before.language ? app::tr("Settings saved. The new language shows when "
                                                                       "the app is next started.")
                                                             : app::tr("Settings saved."));
    if (settings_.ffmpeg_dir != before.ffmpeg_dir || settings_.cache_dir != before.cache_dir) {
        refresh_status();
    }
}

void SettingsPanel::read_controls() {
    settings_.ffmpeg_dir = ffmpeg_edit->text().toStdString();
    settings_.cache_dir = cache_edit->text().toStdString();
    settings_.export_dir = export_edit->text().toStdString();
    settings_.use_cache = use_cache->isChecked();
    settings_.default_gpu_decode = default_gpu->isChecked();
    for (const auto& [key, box] : default_metrics) {
        *settings_.default_compute(key) = box->isChecked();
    }
    settings_.cvvdp_default_preset = cvvdp_default->currentData().toString().toStdString();
    settings_.compare_decoded_videos = decoded_videos->currentData().toInt();
    settings_.remember_window_size = remember_size->isChecked();
    settings_.remember_videos = remember_videos->isChecked();
    if (!settings_.remember_videos) {
        // The paths are kept only while asked for.
        settings_.remembered_reference.clear();
        settings_.remembered_tests.clear();
    }
    settings_.check_for_updates = check_updates->isChecked();
    settings_.language = language->currentData().toString().toStdString();
    settings_.theme = theme->currentData().toString().toStdString();
    settings_.gpu_metrics_together = gpu_together->isChecked();
    if (vship_build_default != nullptr) {
        settings_.default_vship_build = vship_build_default->currentData().toString().toStdString();
    }
}

void SettingsPanel::apply_global_changes(const Settings& before) {
    if (settings_.theme != before.theme) {
        theme::apply_theme(text_of(settings_.theme));  // the window's changeEvent follows
    }
}

void SettingsPanel::pick_directory(const QString& title, QLineEdit* edit) {
    const QString directory = dialogs_.choose_folder(title, edit->text());
    if (!directory.isEmpty()) {
        edit->setText(directory);
        on_edited();
    }
}

void SettingsPanel::on_copy_log() {
    std::optional<std::pair<std::string, bool>> shared;
    try {
        shared = app_log::log_text_to_share();
    } catch (const std::runtime_error& error) {
        status->setText(app::tr("Could not read the log: {error}", {{"error", QString::fromUtf8(error.what())}}));
        return;
    }
    if (!shared) {
        status->setText(app::tr("There is no log to copy yet."));
        return;
    }
    const QString text = QString::fromStdString(shared->first);
    QApplication::clipboard()->setText(text);
    const long long lines = text.count('\n') + 1;
    status->setText(shared->second ? ntr("Copied the log to the clipboard ({count} line, the middle left out -- "
                                         "Export log saves all of it).",
                                         "Copied the log to the clipboard ({count} lines, the middle left out -- "
                                         "Export log saves all of it).",
                                         lines)
                                   : ntr("Copied the log to the clipboard ({count} line).",
                                         "Copied the log to the clipboard ({count} lines).", lines));
}

void SettingsPanel::on_export_log() {
    const std::filesystem::path start = settings_.export_dir.empty()
                                            ? platform::home_directory() / "Documents"
                                            : platform::path_from_text(settings_.export_dir);
    const std::string name =
        "VideoMetricsLab log " + QDateTime::currentDateTime().toString("yyyy-MM-dd HHmm").toStdString() + ".zip";
    const QString path = dialogs_.save_file(app::tr("Export log"), text_of(start / platform::path_from_text(name)),
                                            app::tr("Zip archive (*.zip)"));
    if (path.isEmpty()) {
        return;
    }
    std::vector<std::string> names;
    try {
        names = app_log::export_logs(platform::path_from_text(path.toStdString()));
    } catch (const std::exception& error) {
        status->setText(app::tr("Could not export the log: {error}", {{"error", QString::fromUtf8(error.what())}}));
        return;
    }
    if (names.empty()) {
        status->setText(app::tr("There is no log to export yet."));
        return;
    }
    logger.info("Log exported to " + path.toStdString());
    status->setText(app::tr("Log exported to {path}", {{"path", path}}));
}

}  // namespace vml::app
