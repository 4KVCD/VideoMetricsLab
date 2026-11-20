#include "settings_panel.h"

#include <QAbstractButton>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QButtonGroup>
#include <QColorDialog>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QFontMetrics>
#include <QRawFont>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QScopedValueRollback>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QStyle>
#include <QStyleOptionComboBox>
#include <QTabWidget>
#include <QTextDocument>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>

#include "file_writes.h"
#include "formatting.h"
#include "i18n.h"
#include "segment_rules_editor.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/crop_detect.h"
#include "vml/cvvdp.h"
#include "vml/gpu_choice.h"
#include "vml/gpu_frames.h"
#include "vml/metric_cache.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"
#include "vml/vship.h"
#include "vship_names.h"
#include "widgets.h"
#include "window_helpers.h"

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
    // Shown while metrics are calculated (set_run_active).
    // A padlock before it in Classic; in VML Modern a tinted strip (the
    // notice role), its text alone (set_run_active).
    run_lock_note = new QLabel;
    run_lock_note->setObjectName("runLockNote");
    theme::set_role(run_lock_note, "notice");
    run_lock_note->setWordWrap(true);
    theme::style(run_lock_note, "color: {muted};");
    run_lock_note->hide();
    layout->addWidget(run_lock_note);
    // Sub-tabs of related boxes rather than one long column (Notifications:
    // settings_notifications.cpp). A page of one
    // box is named after it (Storage, Weak segments), so "Settings > Storage
    // > ..." still names a setting; its box drops the title the tab shows.
    sections = new QTabWidget;
    sections->setObjectName("settingsSections");
    sections->setDocumentMode(true);
    keep_focus_on_tab_click(sections);  // a section clicked focuses none of its boxes
    theme::set_role(sections, "segmented");  // VML Modern: a row of pills
    theme::set_role(sections->tabBar(), "segmented");
    const auto add_section = [this](const QString& name, std::initializer_list<QGroupBox*> boxes) {
        auto* page = new QWidget;
        auto* page_layout = new QVBoxLayout(page);
        for (QGroupBox* box : boxes) {
            if (boxes.size() == 1) {
                box->setTitle(QString());
            }
            page_layout->addWidget(box);
        }
        page_layout->addStretch(1);
        sections->addTab(page, name);
    };
    add_section(app::tr("General"), {build_ffmpeg_box(), build_window_box(), build_compare_box()});
    sections->addTab(build_appearance_page(), app::tr("Appearance"));
    add_section(app::tr("Metrics"), {build_defaults_box(), build_gpu_box()});
    add_section(app::tr("Weak segments"), {build_weak_segments_box()});
    add_section(app::tr("Storage"), {build_storage_box()});
    sections->addTab(build_notifications_page(), app::tr("Notifications"));
    layout->addWidget(sections, 1);
    status = new QLabel;
    theme::style(status, "color: {muted};");
    layout->addWidget(status);
    new WheelPastValueBoxes(this);  // the wheel scrolls this page, never changes a setting
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
    auto* browse_ffmpeg = new QPushButton(app::tr("Browse"));
    connect(browse_ffmpeg, &QPushButton::clicked, this,
            [this] { pick_directory(app::with_program_names(app::tr("Select the folder containing ffmpeg.exe")), ffmpeg_edit); });
    auto* ffmpeg_row = new QHBoxLayout;
    ffmpeg_row->addWidget(ffmpeg_edit, 1);
    ffmpeg_row->addWidget(browse_ffmpeg);
    tools_form->addRow(app::tr("ffmpeg folder:"), ffmpeg_row);
    ffmpeg_edit->setObjectName("ffmpegFolder");
    run_locked_.insert(run_locked_.end(), {ffmpeg_edit, browse_ffmpeg});
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
    auto* browse_cache = new QPushButton(app::tr("Browse"));
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
    cache_edit->setObjectName("savedResultsFolder");
    run_locked_.insert(run_locked_.end(), {cache_edit, browse_cache});

    cache_summary = new QLabel;
    auto* clear_cache = new QPushButton(app::tr("Clear saved results"));
    theme::set_role(clear_cache, "danger");
    connect(clear_cache, &QPushButton::clicked, this, [this] { clear_saved_results(); });
    auto* summary_row = new QHBoxLayout;
    summary_row->addWidget(cache_summary, 1);
    summary_row->addWidget(clear_cache);
    storage_form->addRow("", summary_row);
    clear_cache->setObjectName("clearSavedResults");
    run_locked_.push_back(clear_cache);

    export_edit = new QLineEdit(text_of(settings_.export_dir));
    export_edit->setPlaceholderText(app::tr("blank = ask each time"));
    connect(export_edit, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    auto* browse_export = new QPushButton(app::tr("Browse"));
    connect(browse_export, &QPushButton::clicked, this,
            [this] { pick_directory(app::tr("Where should exports be written?"), export_edit); });
    auto* export_row = new QHBoxLayout;
    export_row->addWidget(export_edit, 1);
    export_row->addWidget(browse_export);
    storage_form->addRow(app::tr("Export folder:"), export_row);

    // The colours Metric Graphs' and the Bitrate tab's Export graph as PNG draw in:
    // light unless chosen otherwise, whatever the window shows.
    const QString colors_tip =
        app::tr("The colors of the PNG images Metric Graphs and the Bitrate Viewer export. A Background or Text "
                "color, when chosen, takes the place of the theme's; Reset color in its dialog goes back to it. "
                "The lines keep their own colors.");
    export_graph_theme = new QComboBox;
    export_graph_theme->addItem(app::tr("Same as the window"), "window");
    export_graph_theme->addItem(app::tr("Light"), "light");
    export_graph_theme->addItem(app::tr("Dark"), "dark");
    const int chosen_theme = export_graph_theme->findData(text_of(settings_.export_graph_theme));
    export_graph_theme->setCurrentIndex(chosen_theme >= 0 ? chosen_theme : export_graph_theme->findData("light"));
    export_graph_theme->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    export_graph_theme->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    export_graph_theme->setToolTip(colors_tip);
    connect(export_graph_theme, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    export_background_button = new QPushButton;
    export_background_button->setProperty("chosen", text_of(settings_.export_graph_background));
    export_background_button->setToolTip(colors_tip);
    connect(export_background_button, &QPushButton::clicked, this, [this] {
        choose_export_color(export_background_button, app::tr("Exported graphs' background"));
    });
    export_text_button = new QPushButton;
    export_text_button->setProperty("chosen", text_of(settings_.export_graph_text));
    export_text_button->setToolTip(colors_tip);
    connect(export_text_button, &QPushButton::clicked, this,
            [this] { choose_export_color(export_text_button, app::tr("Exported graphs' text")); });
    auto* export_colors_row = new QHBoxLayout;
    export_colors_row->addWidget(export_graph_theme);
    export_colors_row->addWidget(export_background_button);
    export_colors_row->addWidget(export_text_button);
    export_colors_row->addStretch(1);
    storage_form->addRow(app::tr("Exported graphs:"), export_colors_row);
    show_export_colors();

    storage_form->addRow(app::tr("Exports include:"), build_export_choices());
    storage_form->addRow(app::tr("After each run:"), build_after_run_export());


    use_cache = new QCheckBox(app::tr("Reuse a saved result when a video is added again"));
    use_cache->setChecked(settings_.use_cache);
    connect(use_cache, &QCheckBox::toggled, this, [this] { on_edited(); });
    storage_form->addRow("", use_cache);
    recognise_remuxes = new QCheckBox(app::tr("Recognize test videos remuxed after scoring (reads each one again)"));
    recognise_remuxes->setToolTip(Qt::convertFromPlainText(app::tr(
        "A test video muxed or remuxed after it was scored (its raw stream into an MKV, an MKV into an MP4) keeps "
        "its saved scores: it is known by its coded pictures.\n\nThat takes reading each test video once more "
        "after its scores are saved, as long as reading the file takes (a few minutes for a film on a network "
        "share), in the background and never during a run; and a video whose stream begins as a scored one's is "
        "read whole to tell whether it is that video, after the other videos' saved scores are found.\n\n"
        "Off: test videos are known by their files only."),
        Qt::WhiteSpaceNormal));
    recognise_remuxes->setChecked(settings_.recognise_test_remuxes);
    connect(recognise_remuxes, &QCheckBox::toggled, this, [this] { on_edited(); });
    storage_form->addRow("", recognise_remuxes);
    check_frame_counts = new QCheckBox(app::tr("Read entire files to check frame counts when needed (can be slow)"));
    // As rich text, which Qt wraps (a plain tooltip is one line however long)
    check_frame_counts->setToolTip(Qt::convertFromPlainText(app::tr(
        "Sometimes a test video ends up with a few more or fewer frames than the reference, for example when an "
        "encoder duplicates frames to keep a constant frame rate. Frames are then compared with the wrong "
        "reference frames, and every score from that point on is wrong.\n\nThe app notices when this may have "
        "happened and checks the frame counts that MP4 files, and MKV files made by mkvmerge, store. Many other "
        "files, including those made by FFmpeg and HandBrake, don't store one.\n\nWhen this is on, such a file is "
        "read from start to finish to count its frames. This happens only for a video that looks out of sync, "
        "after it has been scored, but it can add several minutes per video for a film on a network drive. When "
        "it's off, the video is only marked as possibly out of sync.\n\nTakes effect from the next run."),
        Qt::WhiteSpaceNormal));
    check_frame_counts->setChecked(settings_.check_frame_counts);
    connect(check_frame_counts, &QCheckBox::toggled, this, [this] { on_edited(); });
    // Said where it is seen, right under it: the time it can add comes after a video's scores
    auto* frame_count_note = new QLabel(app::tr(
        "When on, a video that looks out of sync can take several extra minutes to finish after it's scored, "
        "while its files are read to count their frames."));
    frame_count_note->setWordWrap(true);
    // At the top of the height a wrapping label is given, not in its middle
    frame_count_note->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    theme::style(frame_count_note, "color: {muted};");
    auto* frame_count_column = new QVBoxLayout;
    frame_count_column->setContentsMargins(0, 0, 0, 0);
    frame_count_column->setSpacing(2);
    frame_count_column->addWidget(check_frame_counts);
    frame_count_column->addWidget(frame_count_note);
    storage_form->addRow("", frame_count_column);

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
    export_log_button = new QPushButton(app::tr("Export log"));
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

QWidget* SettingsPanel::build_export_choices() {
    // A box for each kind of export, so that another kind is one more box;
    // a box for each format the results and weak segments are written in;
    // then as a .zip file or loose files. The right-click menus' exports and
    // the one after each run alike.
    auto* exports = new QWidget;
    auto* row = new QHBoxLayout(exports);
    row->setContentsMargins(0, 0, 0, 0);
    const std::map<std::string, std::pair<QString, QString>> described{
        {"results",
         {app::tr("Results"),
          app::tr("Scores and statistics, as Export results saves them, in the formats ticked: a video's alone "
                  "(<video> - results.json), or all the videos' in one file.")}},
        {"frames",
         {app::tr("Per-frame scores"),
          app::tr("Every frame's score of each metric, in the formats ticked: a file for each video "
                  "(<video> - frames.json). A long video's are big.")}},
        {"graphs",
         {app::tr("Graphs"),
          app::tr("Each video's graph of every metric, and its bitrate with the reference's, as PNG images "
                  "(<video> - <metric>.png); with several videos, all of them together too. They're saved once "
                  "the bitrate has been read, which the Bitrate Viewer does after each run.")}},
        {"weak_segments",
         {app::tr("Weak segments"),
          app::tr("Weak segments by the profile in use (Settings > Weak segments; on the Saved Results tab, the "
                  "rules picked there), in the formats ticked: a video's (<video>.vmaf.segments.json), or all "
                  "the videos' in one report. Nothing is saved while the rules have no condition.")}},
    };
    for (const std::string& key : export_item_keys()) {
        const auto& [name, tip] = described.at(key);
        auto* box = new QCheckBox(name);
        box->setToolTip(Qt::convertFromPlainText(tip, Qt::WhiteSpaceNormal));
        box->setChecked(std::find(settings_.export_items.begin(), settings_.export_items.end(), key) !=
                        settings_.export_items.end());
        connect(box, &QCheckBox::toggled, this, [this] { on_edited(); });
        export_item_boxes[key] = box;
        row->addWidget(box);
    }
    row->addSpacing(12);
    row->addWidget(new QLabel(app::tr("In:")));
    const std::map<std::string, std::pair<QString, QString>> formats{
        {"json",
         {QStringLiteral("JSON"),
          app::tr("Results, per-frame scores and weak segments as JSON files. With neither JSON nor CSV "
                  "ticked, only graphs are exported.")}},
        {"csv",
         {QStringLiteral("CSV"),
          app::tr("Results, per-frame scores and weak segments as CSV files, for spreadsheets. With neither JSON "
                  "nor CSV ticked, only graphs are exported.")}},
    };
    for (const std::string& key : export_format_keys()) {
        const auto& [name, tip] = formats.at(key);
        auto* box = new QCheckBox(name);
        box->setToolTip(Qt::convertFromPlainText(tip, Qt::WhiteSpaceNormal));
        box->setChecked(std::find(settings_.export_formats.begin(), settings_.export_formats.end(), key) !=
                        settings_.export_formats.end());
        connect(box, &QCheckBox::toggled, this, [this] { on_edited(); });
        export_format_boxes[key] = box;
        row->addWidget(box);
    }
    row->addSpacing(12);
    export_form = new QComboBox;
    export_form->addItem(app::tr("A .zip file"), true);
    export_form->addItem(app::tr("Loose files"), false);
    export_form->setCurrentIndex(settings_.export_zip ? 0 : 1);
    export_form->setToolTip(Qt::convertFromPlainText(
        app::tr("A .zip file: a video's in <video> - VideoMetricsLab.zip, all the videos' in one .zip named after "
                "the reference. Loose files: each file on its own, side by side, in the folder chosen."),
        Qt::WhiteSpaceNormal));
    export_form->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    export_form->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    connect(export_form, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    row->addWidget(new QLabel(app::tr("As:")));
    row->addWidget(export_form);
    row->addStretch(1);
    return exports;
}

QWidget* SettingsPanel::build_after_run_export() {
    // Whether each video is exported (as Exports include say) once its run
    // ends, and where.
    auto* after_run = new QWidget;
    auto* row = new QHBoxLayout(after_run);
    row->setContentsMargins(0, 0, 0, 0);
    after_run_box = new QCheckBox(app::tr("Export each video"));
    after_run_box->setChecked(settings_.after_run_export);
    after_run_box->setToolTip(Qt::convertFromPlainText(
        app::tr("Once each video's run ends, exports what Exports include, as they say."), Qt::WhiteSpaceNormal));
    after_run_place = new QComboBox;
    after_run_place->addItem(app::tr("Next to each test video"), QStringLiteral("beside"));
    after_run_place->addItem(app::tr("In the export folder"), QStringLiteral("folder"));
    after_run_place->setCurrentIndex(settings_.after_run_place == "folder" ? 1 : 0);
    after_run_place->setEnabled(settings_.after_run_export);
    after_run_place->setToolTip(Qt::convertFromPlainText(
        app::tr("In the export folder: next to the video while no export folder is set. Existing files of the "
                "same names are replaced."),
        Qt::WhiteSpaceNormal));
    after_run_place->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    after_run_place->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    connect(after_run_box, &QCheckBox::toggled, this, [this](bool on) {
        after_run_place->setEnabled(on);
        on_edited();
    });
    connect(after_run_place, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    row->addWidget(after_run_box);
    row->addSpacing(12);
    row->addWidget(new QLabel(app::tr("Save to:")));
    row->addWidget(after_run_place);
    row->addStretch(1);
    return after_run;
}

QGroupBox* SettingsPanel::build_weak_segments_box() {
    auto* box = new QGroupBox(app::tr("Weak segments"));
    auto* layout = new QVBoxLayout(box);
    auto* note = new QLabel(app::tr("Where a video's scores count as weak, in named profiles for different kinds "
                                    "of video. The profile picked here is the one in use: Metric Graphs > Worst "
                                    "frames > Weak segments starts from it, and the weak segments report, the "
                                    "exports after each run and the command line's --segments saved follow it."));
    note->setWordWrap(true);
    theme::style(note, "color: {muted};");
    layout->addWidget(note);
    auto* profile_row = new QHBoxLayout;
    profile_row->addWidget(new QLabel(app::tr("Profile:")));
    weak_profile = new QComboBox;
    weak_profile->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    profile_row->addWidget(weak_profile);
    auto* new_profile = new QPushButton(app::tr("New"));
    new_profile->setToolTip(app::tr("A new profile, starting from this one's rules."));
    connect(new_profile, &QPushButton::clicked, this, [this] { new_weak_profile(); });
    profile_row->addWidget(new_profile);
    auto* from_preset = new QPushButton(app::tr("From preset"));
    from_preset->setToolTip(app::tr("Loads a starting point for a kind of video into this profile, replacing its "
                                    "rules, to adjust as needed. To keep the rules shown too, make a profile with "
                                    "New first."));
    auto* presets_menu = new QMenu(from_preset);
    presets_menu->setToolTipsVisible(true);
    for (const SegmentPreset& preset : segment_presets()) {
        QAction* action = presets_menu->addAction(app::tr(preset.name));
        action->setToolTip(app::tr(preset.about));
        connect(action, &QAction::triggered, this,
                [this, key = preset.key] { load_weak_preset(key); });
    }
    from_preset->setMenu(presets_menu);
    profile_row->addWidget(from_preset);
    weak_profile_rename = new QPushButton(app::tr("Rename"));
    connect(weak_profile_rename, &QPushButton::clicked, this, [this] { rename_weak_profile(); });
    profile_row->addWidget(weak_profile_rename);
    weak_profile_delete = new QPushButton(app::tr("Delete"));
    connect(weak_profile_delete, &QPushButton::clicked, this, [this] { delete_weak_profile(); });
    profile_row->addWidget(weak_profile_delete);
    profile_row->addStretch(1);
    layout->addLayout(profile_row);
    std::vector<std::string> keys;
    for (const MetricDefinition& definition : metrics()) {
        keys.emplace_back(definition.key);
    }
    weak_segments = new SegmentRulesEditor(std::move(keys), false);
    fill_weak_profiles();
    connect(weak_profile, &QComboBox::currentIndexChanged, this, [this] {
        if (filling_profiles_) {
            return;
        }
        // The profile picked is the one in use; its rules are shown.
        settings_.weak_segment_profile = weak_profile->currentData().toString().toStdString();
        fill_weak_profiles();
        on_edited();
        emit weak_segment_rules_changed();  // changed before on_edited looked
    });
    connect(weak_segments, &SegmentRulesEditor::changed, this, [this] { on_edited(); });
    layout->addWidget(weak_segments);

    // Exported with the rest: Settings > Storage > Exports include.
    auto* after_run_note = new QLabel(app::tr("Exported when ticked in Settings > Storage > Exports include."));
    theme::style(after_run_note, "color: {muted};");
    weak_segments->add_option_row(app::tr("Exports:"), after_run_note);
    return box;
}

void SettingsPanel::show_weak_segment_rules() { fill_weak_profiles(); }

void SettingsPanel::fill_weak_profiles() {
    const QScopedValueRollback filling(filling_profiles_, true);
    const SegmentProfile in_use = segment_profile_in_use(settings_);
    weak_profile->clear();
    for (const SegmentProfile& profile : segment_profiles(settings_)) {
        weak_profile->addItem(profile.name.empty() ? app::tr("Default") : text_of(profile.name),
                              text_of(profile.name));
    }
    weak_profile->setCurrentIndex(std::max(weak_profile->findData(text_of(in_use.name)), 0));
    weak_profile_rename->setEnabled(!in_use.name.empty());  // Default keeps its name
    weak_profile_delete->setEnabled(!in_use.name.empty());  // and is always there
    weak_segments->set_remembered(in_use.rules.conditions);
    weak_segments->set_rules(in_use.rules);
}

void SettingsPanel::new_weak_profile() {
    const std::optional<QString> name =
        dialogs_.ask_text(app::tr("New weak segment profile"), app::tr("Profile name:"), QString());
    if (!name) {
        return;
    }
    const std::string text = std::string(python_strip(name->toStdString()));
    if (!segment_profile_name_free(segment_profiles(settings_), text)) {
        dialogs_.alert(app::tr("Profile not saved"),
                       app::tr("A profile needs a name of its own, without commas, and not Default or another "
                               "profile's."));
        return;
    }
    store_segment_profile(settings_, text, weak_segments->rules());
    settings_.weak_segment_profile = text;
    fill_weak_profiles();
    on_edited();
    // The settings changed before on_edited looked: told here.
    emit weak_segment_rules_changed();
}

void SettingsPanel::load_weak_preset(std::string_view key) {
    const SegmentPreset* preset = segment_preset(key);
    if (preset == nullptr) {
        return;
    }
    // Into the profile in use, Default too, as an edit of its rules: no
    // profile is added (New... makes one first, from the rules shown).
    store_segment_profile(settings_, segment_profile_in_use(settings_).name, preset->rules);
    fill_weak_profiles();
    on_edited();
    // The settings changed before on_edited looked: told here.
    emit weak_segment_rules_changed();
}

void SettingsPanel::rename_weak_profile() {
    std::vector<SegmentProfile> profiles = segment_profiles(settings_);
    const std::string current = segment_profile_in_use(settings_).name;
    if (current.empty()) {
        return;
    }
    const std::optional<QString> name =
        dialogs_.ask_text(app::tr("Rename weak segment profile"), app::tr("Profile name:"), text_of(current));
    if (!name) {
        return;
    }
    const std::string text = std::string(python_strip(name->toStdString()));
    std::vector<SegmentProfile> others;
    std::copy_if(profiles.begin(), profiles.end(), std::back_inserter(others),
                 [&current](const SegmentProfile& profile) { return profile.name != current; });
    if (!segment_profile_name_free(others, text)) {
        dialogs_.alert(app::tr("Profile not renamed"),
                       app::tr("A profile needs a name of its own, without commas, and not Default or another "
                               "profile's."));
        return;
    }
    for (SegmentProfile& profile : profiles) {
        if (profile.name == current) {
            profile.name = text;
        }
    }
    store_segment_profiles(settings_, profiles);
    settings_.weak_segment_profile = text;
    fill_weak_profiles();
    on_edited();
    // The settings changed before on_edited looked: told here.
    emit weak_segment_rules_changed();
}

void SettingsPanel::delete_weak_profile() {
    std::vector<SegmentProfile> profiles = segment_profiles(settings_);
    const std::string current = segment_profile_in_use(settings_).name;
    if (current.empty() ||
        !dialogs_.ask_no_default(app::tr("Delete profile"),
                                 app::tr("Delete the weak segment profile {name}? Default is used in its place.",
                                         {{"name", text_of(current)}}))) {
        return;
    }
    std::erase_if(profiles, [&current](const SegmentProfile& profile) { return profile.name == current; });
    store_segment_profiles(settings_, profiles);
    settings_.weak_segment_profile.clear();
    fill_weak_profiles();
    on_edited();
    // The settings changed before on_edited looked: told here.
    emit weak_segment_rules_changed();
}

QGroupBox* SettingsPanel::build_defaults_box() {
    auto* defaults_box = new QGroupBox(app::tr("Defaults for newly added videos"));
    auto* defaults_layout = new QVBoxLayout(defaults_box);
    auto* hint = new QLabel(app::tr("These are only a starting point -- each video's own settings are "
                                    "edited in the Videos tab."));
    theme::style(hint, "color: {muted}; font-style: {hint_style};");
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

    // The entries' data: Settings' film_grain_default values.
    auto* grain_row = new QHBoxLayout;
    grain_row->addWidget(new QLabel(app::tr("Film grain:")));
    film_grain_default = new QComboBox;
    film_grain_default->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    film_grain_default->addItem(app::tr("Ask at Calculate metrics"), QString());
    film_grain_default->addItem(app::tr("With film grain (what viewers see)"), QStringLiteral("with"));
    film_grain_default->addItem(app::tr("Without film grain (the coded pictures only)"), QStringLiteral("without"));
    film_grain_default->setToolTip(app::tr(
        "How a video whose stream asks the player to add film grain is scored: asked when its metrics are "
        "calculated, or with or without the grain. Each video keeps the choice it was given, by you or by this "
        "setting when it was added; change a video's own in the Options panel."));
    show_film_grain_default();
    connect(film_grain_default, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    grain_row->addWidget(film_grain_default);
    grain_row->addStretch(1);
    defaults_layout->addLayout(grain_row);
    return defaults_box;
}

QGroupBox* SettingsPanel::build_gpu_box() {
    auto* gpu_box = new QGroupBox(app::tr("GPU metrics"));
    auto* gpu_layout = new QVBoxLayout(gpu_box);
    // The GPU to score on, where there is a choice (Settings::gpu): the
    // entries' data is the text saved (gpu_choice_texts: the name, with its
    // place where two GPUs have it), empty for automatic.
    const std::vector<platform::GpuAdapter> adapters = platform::gpu_adapters();
    if (adapters.size() > 1 || !settings_.gpu.empty()) {
        auto* choice_row = new QHBoxLayout;
        choice_row->addWidget(new QLabel(app::tr("GPU:")));
        gpu_choice = new QComboBox;
        gpu_choice->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        gpu_choice->addItem(app::tr("Automatic (a discrete GPU first)"), QString());
        std::vector<std::string> names;
        for (const platform::GpuAdapter& adapter : adapters) {
            names.push_back(adapter.name);
        }
        for (const std::string& text : gpu_choice_texts(names)) {
            gpu_choice->addItem(QString::fromStdString(text), QString::fromStdString(text));
        }
        const QString saved = QString::fromStdString(settings_.gpu);
        const std::optional<std::size_t> found = saved.isEmpty() ? std::nullopt : chosen_among(names, settings_.gpu);
        int shown = saved.isEmpty() ? 0 : found ? static_cast<int>(*found) + 1 : -1;
        if (shown < 0) {
            gpu_choice->addItem(app::tr("{name} (not found)", {{"name", saved}}), saved);
            shown = gpu_choice->count() - 1;
        }
        gpu_choice->setCurrentIndex(shown);
        gpu_choice->setToolTip(app::tr(
            "The GPU that calculates VMAF, VMAF v1, SSIMULACRA2, Butteraugli and CVVDP on a PC with more than one, "
            "and whose maker's decoder is tried first when GPU decoding is automatic. A metric its GPU cannot "
            "calculate is calculated on the CPU rather than on another GPU. Automatic uses a discrete GPU first.\n\n"
            "Takes effect when the app is next started; the command line's --gpu chooses for one run."));
        connect(gpu_choice, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
        choice_row->addWidget(gpu_choice);
        choice_row->addStretch(1);
        gpu_layout->addLayout(choice_row);
    }
    if (vship_build_bundled(VshipBuild::stock)) {
        // The entries' data: Settings' default_vship_build values.
        auto* build_row = new QHBoxLayout;
        build_row->addWidget(new QLabel(app::tr("New videos use:")));
        vship_build_default = new QComboBox;
        add_vship_choices(*vship_build_default, bundled_vship_builds());
        show_vship_choice(*vship_build_default, default_vship_build(settings_.default_vship_build));
        // Settings has the room the Options panel lacks: the whole text, not add_vship_choices' short box.
        vship_build_default->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        vship_build_default->setToolTip(app::tr(
            "The Vship that calculates SSIMULACRA2, Butteraugli and CVVDP for newly added videos, and on the command "
            "line without --vship. Each video's own is set in the Options panel (Vship compute)."));
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
    if (vvdec_vulkan_bundled()) {
        gpu_layout->addLayout(build_vvc_decoder_row());
    }
    return gpu_box;
}

QHBoxLayout* SettingsPanel::build_vvc_decoder_row() {
    // The entries' data: Settings' vvc_decode_on_gpu. CPU stays as it is in every language.
    auto* row = new QHBoxLayout;
    row->addWidget(new QLabel(app::tr("VVC decoder:")));
    vvc_decoder = new QComboBox;
    vvc_decoder->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    vvc_decoder->addItem(app::tr("GPU (experimental)"), true);
    vvc_decoder->addItem(QStringLiteral("CPU"), false);
    vvc_decoder->setCurrentIndex(settings_.vvc_decode_on_gpu ? 0 : 1);
    vvc_decoder->setToolTip(app::tr(
        "How a VVC video is decoded for every metric: GPU (experimental) decodes it on the GPU, which frees the CPU "
        "but can be slower than a fast CPU; CPU decodes it with VVdeC. The CPU decodes it anyway for a video that is "
        "scaled or has AFGS1 film grain added, with CUDA VMAF, with GPU decoding off, for black-bar detection, in "
        "Video Compare, and where the GPU's decoder cannot start.\n\nThe scores are the same either way. Applies from "
        "the next run."));
    connect(vvc_decoder, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    row->addWidget(vvc_decoder);
    row->addStretch(1);
    return row;
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
    remember_size = new QCheckBox(app::tr("Reopen where and at the size the window was last closed"));
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

void SettingsPanel::show_film_grain_default() {
    const QSignalBlocker blocker(film_grain_default);
    const int index = film_grain_default->findData(QString::fromStdString(settings_.film_grain_default));
    film_grain_default->setCurrentIndex(index < 0 ? 0 : index);  // an unknown value asks
}

void SettingsPanel::clear_saved_results() {
    const std::filesystem::path directory = cache_folder();
    // The count the tab shows, made off the window's thread
    // (refresh_status): counting a large folder here would freeze the
    // window. While it is still being made, the question is asked anyway.
    const bool counted = counting_->isFinished() && counting_->future().resultCount() > 0 &&
                         counted_folder_ == directory;
    if (counted && counting_->result().comparisons == 0 && counting_->result().bytes == 0 &&
        file_writes_.idle()) {
        status->setText(app::tr("There are no saved results to clear."));
        return;
    }
    if (cleared_) {
        status->setText(app::tr("Saved results are already being cleared."));
        return;
    }
    if (!dialogs_.ask(app::tr("Clear saved results"),
                      app::tr("Delete saved results from {directory}?\n\nVideos already scored will have to be "
                              "recomputed. Those whose reference video is no longer where it was are kept: their "
                              "scores could not be calculated again. Delete them in the Saved Results tab.",
                              {{"directory", text_of(directory)}}))) {
        return;
    }
    // Saved scores are written and deleted through the same queue, one after
    // another. If a run just finished, its scores therefore land before they
    // are cleared, rather than coming back after the user cleared them all.
    // `directory` is the folder the question just named: if the setting
    // changes before the queue gets to this, it must not delete a folder the
    // user was never asked about.
    cleared_ = std::make_shared<ClearedResults>();
    file_writes_.submit(app::tr("clear saved results"), [directory, cleared = cleared_] {
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
        // Not those whose scores could not be calculated again.
        *cleared = vml::clear_saved_results(directory, true);
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
    const ClearedResults cleared = *cleared_;
    cleared_.reset();
    QString said = ntr("Removed {count} saved result.", "Removed {count} saved results.", cleared.removed);
    if (cleared.kept > 0) {
        said += " " + ntr("Kept {count} whose reference video is gone: delete it in the Saved Results tab.",
                          "Kept {count} whose reference videos are gone: delete them in the Saved Results tab.",
                          cleared.kept);
    }
    status->setText(said);
    refresh_status();
    emit saved_results_cleared();
}

void SettingsPanel::on_edited() {
    const Settings before = settings_;  // to see what this edit changed
    read_controls();
    show_export_colors();
    emit decoded_videos_changed(settings_.compare_decoded_videos);
    apply_global_changes(before);
    // Only the starting point for new rows; rows already in the table keep
    // what they were given.
    emit defaults_changed();
    const std::optional<std::string> error = settings_.save();
    if (settings_.weak_segment_conditions != before.weak_segment_conditions ||
        settings_.weak_segments_all != before.weak_segments_all ||
        settings_.weak_segment_min_frames != before.weak_segment_min_frames ||
        settings_.weak_segment_merge_frames != before.weak_segment_merge_frames ||
        settings_.weak_segment_profiles_json != before.weak_segment_profiles_json ||
        settings_.weak_segment_profile != before.weak_segment_profile) {
        emit weak_segment_rules_changed();
    }
    if (settings_.ffmpeg_dir != before.ffmpeg_dir) {
        emit ffmpeg_folder_changed();  // after the save: the tools are looked for in the folder saved
    }
    const bool classic_later = settings_.interface_style != before.interface_style &&
                               theme::interface_of(settings_.interface_style) == theme::Interface::classic &&
                               theme::current_interface() == theme::Interface::modern;
    status->setText(error                                     ? tr_message(text_of(*error))
                    : settings_.language != before.language ? app::tr("Settings saved. The new language shows when "
                                                                       "the app is next started.")
                    : classic_later ? app::tr("Settings saved. VML Classic shows when the app is next started.")
                                    : app::tr("Settings saved."));
    if (settings_.ffmpeg_dir != before.ffmpeg_dir || settings_.cache_dir != before.cache_dir) {
        refresh_status();
    }
}

void SettingsPanel::read_controls() {
    settings_.ffmpeg_dir = ffmpeg_edit->text().toStdString();
    settings_.cache_dir = cache_edit->text().toStdString();
    settings_.export_dir = export_edit->text().toStdString();
    settings_.export_items.clear();
    for (const std::string& key : export_item_keys()) {
        if (export_item_boxes.at(key)->isChecked()) {
            settings_.export_items.push_back(key);
        }
    }
    settings_.export_formats.clear();
    for (const std::string& key : export_format_keys()) {
        if (export_format_boxes.at(key)->isChecked()) {
            settings_.export_formats.push_back(key);
        }
    }
    settings_.export_zip = export_form->currentData().toBool();
    settings_.after_run_export = after_run_box->isChecked();
    settings_.after_run_place = after_run_place->currentData().toString().toStdString();
    // The editor shows the profile in use.
    store_segment_profile(settings_, segment_profile_in_use(settings_).name, weak_segments->rules());
    settings_.export_graph_theme = export_graph_theme->currentData().toString().toStdString();
    settings_.export_graph_background = export_background_button->property("chosen").toString().toStdString();
    settings_.export_graph_text = export_text_button->property("chosen").toString().toStdString();
    settings_.use_cache = use_cache->isChecked();
    settings_.recognise_test_remuxes = recognise_remuxes->isChecked();
    settings_.check_frame_counts = check_frame_counts->isChecked();
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
    settings_.theme = theme_buttons->checkedButton()->property("value").toString().toStdString();
    settings_.interface_style = interface_buttons->checkedId() == 1 ? "modern" : "classic";
    settings_.accent_colour = accent_value.toStdString();
    settings_.gpu_metrics_together = gpu_together->isChecked();
    if (gpu_choice != nullptr) {
        settings_.gpu = gpu_choice->currentData().toString().toStdString();
    }
    if (vvc_decoder != nullptr) {
        settings_.vvc_decode_on_gpu = vvc_decoder->currentData().toBool();
    }
    if (vship_build_default != nullptr) {
        settings_.default_vship_build = vship_build_default->currentData().toString().toStdString();
    }
    settings_.film_grain_default = film_grain_default->currentData().toString().toStdString();
    read_notification_controls();
}

void SettingsPanel::set_run_active(bool active) {
    // The running jobs hold a copy of the settings taken as the run started,
    // but these reach them anyway: the tools are looked for in the ffmpeg
    // folder, and saved scores and black bars are written to the saved
    // results' folder as each job ends. Every other setting takes effect for
    // new videos, the next run or the next start.
    const QString why = active ? app::tr("Locked while metrics are calculated: the run uses it.") : QString();
    for (QWidget* widget : run_locked_) {
        widget->setEnabled(!active);
        // Kept as a property: the widgets' own tooltips, if any, come back.
        if (active) {
            widget->setProperty("unlockedToolTip", widget->toolTip());
            widget->setToolTip(why);
        } else if (widget->property("unlockedToolTip").isValid()) {
            widget->setToolTip(widget->property("unlockedToolTip").toString());
        }
    }
    if (active) {
        const QString text = app::tr("Metrics are being calculated: the ffmpeg folder, the saved results' folder and "
                                     "Clear saved results are locked until the run ends, because the run uses them. "
                                     "Everything else can be changed now.");
        run_lock_note->setText(theme::current_interface() == theme::Interface::modern
                                   ? text
                                   : QStringLiteral("\U0001F512 ") + text);
    }
    run_lock_note->setVisible(active);
}

void SettingsPanel::apply_global_changes(const Settings& before) {
    if (settings_.theme != before.theme) {
        theme::apply_theme(text_of(settings_.theme));  // the window's changeEvent follows
    }
    // VML Modern at once, its accent too; VML Classic, chosen while Modern
    // is shown, at the next start (theme::apply_appearance; the status says so).
    if ((settings_.interface_style != before.interface_style || settings_.accent_colour != before.accent_colour) &&
        theme::interface_of(settings_.interface_style) == theme::Interface::modern) {
        theme::apply_appearance(theme::Interface::modern, theme::accent_of(settings_.accent_colour));
    }
}

void SettingsPanel::show_export_colors() {
    // The colours an export uses now, from the controls (not yet saved
    // while the panel is built).
    const std::string palette = export_graph_theme->currentData().toString().toStdString();
    const QString background = export_background_button->property("chosen").toString();
    const QString text = export_text_button->property("chosen").toString();
    const theme::ChartColours colours =
        theme::export_chart_colours(palette, background.toStdString(), text.toStdString());
    const auto show = [](QPushButton* button, const QString& label, const QString& chosen, const QColor& colour) {
        QPixmap swatch(14, 14);
        swatch.fill(colour);
        QPainter painter(&swatch);
        painter.setPen(QColor(128, 128, 128));
        painter.drawRect(0, 0, 13, 13);
        painter.end();
        button->setIcon(QIcon(swatch));
        button->setText(chosen.isEmpty() ? label : label + QStringLiteral("  ") + chosen.toUpper());
    };
    show(export_background_button, app::tr("Background"), background, colours.background);
    show(export_text_button, app::tr("Text"), text, colours.text);
}

void SettingsPanel::choose_export_color(QPushButton* button, const QString& title) {
    // Starts at the colour exports use now; Reset color goes back to the
    // palette's own.
    const bool background = button == export_background_button;
    const theme::ChartColours colours = theme::export_chart_colours(
        settings_.export_graph_theme, settings_.export_graph_background, settings_.export_graph_text);
    QColorDialog dialog(background ? colours.background : colours.text, this);
    dialog.setWindowTitle(title);
    dialog.setOption(QColorDialog::DontUseNativeDialog, true);
    bool reset = false;
    if (auto* buttons = dialog.findChild<QDialogButtonBox*>()) {
        QAbstractButton* reset_button = buttons->addButton(app::tr("Reset color"), QDialogButtonBox::ResetRole);
        connect(reset_button, &QAbstractButton::clicked, &dialog, [&dialog, &reset] {
            reset = true;
            dialog.accept();
        });
    }
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }
    button->setProperty("chosen", reset ? QString() : dialog.currentColor().name());
    on_edited();
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
    // A bare name: the dialog starts it in the Export folder (Dialogs::export_folder).
    const QString name =
        "VideoMetricsLab log " + QDateTime::currentDateTime().toString("yyyy-MM-dd HHmm") + ".zip";
    const QString path = dialogs_.save_file(app::tr("Export log"), name, app::tr("Zip archive (*.zip)"));
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
