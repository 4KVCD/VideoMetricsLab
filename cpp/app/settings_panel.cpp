#include "settings_panel.h"

#include <QAbstractButton>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QButtonGroup>
#include <QEasingCurve>
#include <QTimer>
#include <QVariantAnimation>
#include <QLocale>
#include <QDateEdit>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QFontMetrics>
#include <QRadioButton>
#include <QRawFont>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QMenu>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QScopedValueRollback>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSpinBox>
#include <QStyle>
#include <QStyleOptionComboBox>
#include <QTabWidget>
#include <QTextDocument>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

#include "app_dialog.h"
#include "file_writes.h"
#include "formatting.h"
#include "i18n.h"
#include "metric_list.h"
#include "metric_order.h"
#include "results_export.h"
#include "saved_deletion.h"
#include "saved_score_details.h"
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
#include "vml/saved_results.h"
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

QString text_of(const std::string& text) { return QString::fromStdString(text); }

}  // namespace

SettingsPanel::SettingsPanel(Settings& settings, FileWrites& file_writes, std::function<ToolsStatus()> tools,
                             const Dialogs& dialogs, QWidget* parent)
    : QWidget(parent),
      settings_(settings),
      file_writes_(file_writes),
      tools_(std::move(tools)),
      dialogs_(dialogs),
      counting_(new QFutureWatcher<CacheSummary>(this)),
      deletion_listing_(new QFutureWatcher<std::vector<SavedComparison>>(this)),
      started_language_(settings.language) {
    connect(counting_, &QFutureWatcher<CacheSummary>::finished, this, [this] { show_cache_summary(); });
    connect(deletion_listing_, &QFutureWatcher<std::vector<SavedComparison>>::finished, this,
            [this] { ask_and_delete(deletion_listing_->result()); });
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
    add_section(app::tr("Saved Results"), {build_saved_scores_box(), build_saved_results_box()});
    add_section(app::tr("Storage"), {build_storage_box()});
    sections->addTab(build_notifications_page(), app::tr("Notifications"));
    layout->addWidget(sections, 1);
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
    export_edit = new QLineEdit(text_of(settings_.export_dir));
    export_edit->setPlaceholderText(app::tr("blank = Documents"));
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
    storage_form->addRow(app::tr("Quick saves:"), build_quick_save_choices());
    storage_form->addRow(app::tr("After each run:"), build_after_run_export());


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

QGroupBox* SettingsPanel::build_saved_scores_box() {
    // How Saved Results shows a comparison's scores: where a metric has
    // several, a card for the one used or for each (Settings::
    // saved_score_versions); and the columns of its details panel
    // (Settings::saved_details_columns).
    auto* box = new QGroupBox(app::tr("Saved scores"));
    auto* layout = new QVBoxLayout(box);
    auto* versions = new QLabel(app::tr("When a metric has several saved scores:"));
    versions->setWordWrap(true);
    layout->addWidget(versions);
    // Use: which of them (Settings::saved_score_pick), and for the newest,
    // which device and engine first (pick_saved_score).
    const auto combo = [this](const std::string& current,
                              const std::vector<std::pair<QString, QString>>& choices) {
        auto* box = new QComboBox;
        for (const auto& [text, value] : choices) {
            box->addItem(text, value);
        }
        box->setCurrentIndex(std::max(0, box->findData(QString::fromStdString(current))));
        connect(box, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
        return box;
    };
    saved_pick = combo(settings_.saved_score_pick,
                       {{app::tr("the one the Videos tab would use"), QStringLiteral("videos")},
                        {app::tr("the newest"), QStringLiteral("newest")},
                        {app::tr("the one made by the newest version of the app"), QStringLiteral("newest_app")}});
    saved_pick->setObjectName(QStringLiteral("savedScorePick"));
    saved_pick->setToolTip(app::tr(
        "The score the card shows and Open in Metric Graphs and the exports take, unless you pick another in the "
        "details panel. The Videos tab's: the one a run would use again with the settings as they are (where it "
        "would use none, the newest)."));
    saved_device = combo(settings_.saved_score_device, {{app::tr("on any device"), QStringLiteral("any")},
                                                         {app::tr("on the GPU first"), QStringLiteral("gpu")},
                                                         {app::tr("on the CPU first"), QStringLiteral("cpu")}});
    saved_engine = combo(settings_.saved_score_engine, {{app::tr("by any engine"), QStringLiteral("any")},
                                                         {app::tr("by the app's own first"), QStringLiteral("app")},
                                                         {app::tr("by FFmpeg first"), QStringLiteral("ffmpeg")}});
    for (QComboBox* first : {saved_device, saved_engine}) {
        first->setToolTip(app::tr("Of the newest, or of those the newest version of the app made, these come "
                                  "first. The Videos tab's choice goes by the settings instead."));
    }
    const auto follow_pick = [this] {
        const bool own = saved_pick->currentData().toString() != QStringLiteral("videos");
        saved_device->setEnabled(own);
        saved_engine->setEnabled(own);
    };
    connect(saved_pick, &QComboBox::currentIndexChanged, this, follow_pick);
    follow_pick();
    auto* use_row = new QHBoxLayout;
    use_row->addWidget(new QLabel(app::tr("Use:")));
    use_row->addWidget(saved_pick);
    use_row->addWidget(saved_device);
    use_row->addWidget(saved_engine);
    use_row->addStretch(1);
    layout->addLayout(use_row);
    // Show: a card for the one used, or for each (Settings::saved_score_versions).
    saved_versions_used = new QRadioButton(app::tr("One card per metric, saying how many are saved"));
    saved_versions_used->setToolTip(
        app::tr("The card shows the score used; the details panel, opened by the button over the scores, lists "
                "them all, and a click on another uses it."));
    saved_versions_each = new QRadioButton(app::tr("A card for each saved score, to pick the one used"));
    saved_versions_each->setToolTip(
        app::tr("Each card shows when its score was saved; the round button on it picks the one used."));
    auto* group = new QButtonGroup(box);
    group->addButton(saved_versions_used);
    group->addButton(saved_versions_each);
    (settings_.saved_score_versions == "each" ? saved_versions_each : saved_versions_used)->setChecked(true);
    auto* show_row = new QHBoxLayout;
    show_row->addWidget(new QLabel(app::tr("Show:")));
    show_row->addWidget(saved_versions_used);
    show_row->addWidget(saved_versions_each);
    show_row->addStretch(1);
    layout->addLayout(show_row);
    connect(saved_versions_each, &QRadioButton::toggled, this, [this] { on_edited(); });

    auto* columns = new QLabel(app::tr("The details panel, opened by the button over the scores, shows each score's "
                                       "metric and score, and:"));
    columns->setWordWrap(true);
    layout->addWidget(columns);
    // Stacks of four, down then across: Median, StDev, Min, Max; the worst
    // 10% to 0.1%; Score bands and how it was made (the user, 2026-11-25:
    // the first two are cohorts).
    auto* grid = new QGridLayout;
    const std::vector<std::string>& keys = saved_details_column_keys();
    for (int index = 0; index < static_cast<int>(keys.size()); ++index) {
        const std::string& key = keys[static_cast<std::size_t>(index)];
        auto* tick = new QCheckBox(saved_details_heading(key));
        tick->setObjectName(QStringLiteral("savedDetails_") + QString::fromStdString(key));
        tick->setToolTip(saved_details_tip(key));
        tick->setChecked(std::find(settings_.saved_details_columns.begin(), settings_.saved_details_columns.end(),
                                   key) != settings_.saved_details_columns.end());
        connect(tick, &QCheckBox::toggled, this, [this] { on_edited(); });
        saved_details_boxes[key] = tick;
        grid->addWidget(tick, index % 4, index / 4);
    }
    grid->setColumnStretch(4, 1);
    layout->addLayout(grid);
    return box;
}

void SettingsPanel::show_saved_results_section() {
    for (int index = 0; index < sections->count(); ++index) {
        if (sections->tabText(index) == app::tr("Saved Results")) {
            sections->setCurrentIndex(index);
        }
    }
}

QGroupBox* SettingsPanel::build_saved_results_box() {
    // Where saved results are kept, how many there are, clearing them, and
    // when they are used (Settings > Saved Results; in Storage before).
    auto* saved_box = new QGroupBox(app::tr("Saved results"));
    auto* saved_form = new QFormLayout(saved_box);
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
    saved_form->addRow(app::tr("Folder:"), cache_row);
    cache_edit->setObjectName("savedResultsFolder");
    run_locked_.insert(run_locked_.end(), {cache_edit, browse_cache});

    cache_summary = new QLabel;
    cache_summary->setObjectName("savedResultsSummary");
    cache_summary->setWordWrap(true);
    saved_form->addRow("", cache_summary);

    // Deleting them, one way for all of it (docs/ARCHITECTURE.md, "Removing
    // and deleting"): every saved result, those not saved again for a while,
    // or those of a stretch of dates; what that would be said under it,
    // and the question before it the Saved Results tab's Delete... asks.
    delete_scope = new QComboBox;
    delete_scope->setObjectName("deleteSavedScope");
    delete_scope->addItem(app::tr("Every saved result"), QStringLiteral("all"));
    delete_scope->addItem(app::tr("Those saved more than"), QStringLiteral("older"));
    delete_scope->addItem(app::tr("Those saved from"), QStringLiteral("between"));
    delete_scope->setCurrentIndex(1);
    delete_days = new QSpinBox;
    delete_days->setObjectName("deleteSavedDays");
    delete_days->setRange(1, 3650);
    delete_days->setValue(90);
    auto* days_label = new QLabel;
    days_label->setObjectName("deleteSavedDaysLabel");
    const auto say_days = [this, days_label] {
        days_label->setText(ntr("day ago", "days ago", delete_days->value()));
    };
    say_days();
    delete_from = new QDateEdit(QDate::currentDate().addDays(-90));
    delete_to = new QDateEdit(QDate::currentDate());
    auto* to_label = new QLabel(app::tr("to"));
    to_label->setObjectName("deleteSavedToLabel");
    for (QDateEdit* date : {delete_from, delete_to}) {
        date->setCalendarPopup(true);
        date->setDisplayFormat(QLocale().dateFormat(QLocale::ShortFormat));
        connect(date, &QDateEdit::dateChanged, this, [this] {
            dates_chosen_ = true;
            show_delete_preview();
        });
    }
    delete_from->setObjectName("deleteSavedFrom");
    delete_to->setObjectName("deleteSavedTo");
    delete_button = new QPushButton(app::tr("Delete..."));
    delete_button->setObjectName("deleteSavedResults");
    theme::set_role(delete_button, "danger");  // files deleted, not to be undone
    connect(delete_button, &QPushButton::clicked, this, [this] { delete_saved_results(); });
    connect(delete_scope, &QComboBox::currentIndexChanged, this, [this] {
        show_delete_scope();
        show_delete_preview();
    });
    connect(delete_days, &QSpinBox::valueChanged, this, [this, say_days] {
        say_days();
        show_delete_preview();
    });
    auto* delete_row = new QHBoxLayout;
    delete_row->addWidget(delete_scope);
    delete_row->addWidget(delete_days);
    delete_row->addWidget(days_label);
    delete_row->addWidget(delete_from);
    delete_row->addWidget(to_label);
    delete_row->addWidget(delete_to);
    delete_row->addStretch(1);
    delete_row->addWidget(delete_button);
    saved_form->addRow(app::tr("Delete:"), delete_row);
    delete_preview = new QLabel;
    delete_preview->setObjectName("deleteSavedPreview");
    theme::style(delete_preview, "color: {muted};");
    saved_form->addRow("", delete_preview);
    delete_scope->setProperty("daysLabel", QVariant::fromValue<QObject*>(days_label));
    delete_scope->setProperty("toLabel", QVariant::fromValue<QObject*>(to_label));
    show_delete_scope();
    run_locked_.insert(run_locked_.end(), {delete_scope, delete_days, delete_from, delete_to, delete_button});

    use_cache = new QCheckBox(app::tr("Reuse a saved result when a video is added again"));
    use_cache->setChecked(settings_.use_cache);
    connect(use_cache, &QCheckBox::toggled, this, [this] { on_edited(); });
    saved_form->addRow("", use_cache);
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
    saved_form->addRow("", recognise_remuxes);
    return saved_box;
}

QWidget* SettingsPanel::build_export_choices() {
    // A box for each kind of export, so that another kind is one more box;
    // a box for each format the results and weak segments are written in;
    // then as a .zip file or loose files. The right-click menus' exports and
    // the one after each run alike.
    auto* exports = new QWidget;
    auto* row = new QHBoxLayout(exports);
    row->setContentsMargins(0, 0, 0, 0);
    const std::map<std::string, QString> described{
        {"results",
         app::tr("Scores and statistics, as Export results saves them, in the formats ticked: a video's alone "
                 "(<video> - results.json), or all the videos' in one file.")},
        {"frames",
         app::tr("Every frame's score of each metric, in the formats ticked: a file for each video "
                 "(<video> - frames.json). A long video's are big.")},
        {"score_summary",
         app::tr("Each metric's score with the statistics and how it was made that Settings > Saved Results "
                 "picks, a row a metric, in the formats ticked: all the videos' in one file (<video> - score "
                 "details.json).")},
        {"graphs",
         app::tr("Each video's graph of every metric, and its bitrate with the reference's, as PNG images "
                 "(<video> - <metric>.png); with several videos, all of them together too. They're saved once "
                 "the bitrate has been read, which the Bitrate Viewer does after each run.")},
        {"worst_frames",
         app::tr("Each video's worst frames of every metric, as many dips as the Worst Frames tab lists (10, 25 "
                 "or 50), each with every metric's score, in the formats ticked (<video> - worst "
                 "frames.json).")},
        {"weak_segments",
         app::tr("Weak segments by the rules the Export panel's Weak segments by picks (the profile in use, "
                 "Settings > Weak segments, until another is picked; after each run, always that profile), in "
                 "the formats ticked: a video's (<video>.vmaf.segments.json), or all the videos' in one report. "
                 "Nothing is saved while the rules have no condition.")},
    };
    for (const std::string& key : export_item_keys()) {
        const QString name = export_item_title(key);
        const QString& tip = described.at(key);
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
          app::tr("Results, per-frame scores, worst frames and weak segments as JSON files. With neither JSON "
                  "nor CSV ticked, only graphs are exported.")}},
        {"csv",
         {QStringLiteral("CSV"),
          app::tr("Results, per-frame scores, worst frames and weak segments as CSV files, for spreadsheets. "
                  "With neither JSON nor CSV ticked, only graphs are exported.")}},
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
                "the reference. Loose files: each file on its own, in the folder chosen. With several videos, each "
                "video's files are in a folder of its own, and the files of all of them together beside those."),
        Qt::WhiteSpaceNormal));
    export_form->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    export_form->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    connect(export_form, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    row->addWidget(new QLabel(app::tr("As:")));
    row->addWidget(export_form);
    row->addStretch(1);
    return exports;
}

void SettingsPanel::show_export_choices() {
    // Set without being taken as the user's edits here (on_edited would read
    // every control back, which these already match).
    const QSignalBlocker edit_held(export_edit);
    export_edit->setText(text_of(settings_.export_dir));
    for (const auto& [key, box] : export_item_boxes) {
        const QSignalBlocker held(box);
        box->setChecked(std::find(settings_.export_items.begin(), settings_.export_items.end(), key) !=
                        settings_.export_items.end());
    }
    for (const auto& [key, box] : export_format_boxes) {
        const QSignalBlocker held(box);
        box->setChecked(std::find(settings_.export_formats.begin(), settings_.export_formats.end(), key) !=
                        settings_.export_formats.end());
    }
    const QSignalBlocker form_held(export_form);
    export_form->setCurrentIndex(settings_.export_zip ? 0 : 1);
}

QWidget* SettingsPanel::build_quick_save_choices() {
    // The formats each card's save icon writes what it shows in (issue
    // #224): a box for each, as Exports include has; graphs are images.
    auto* quick = new QWidget;
    auto* row = new QHBoxLayout(quick);
    row->setContentsMargins(0, 0, 0, 0);
    for (const std::string& key : export_format_keys()) {
        auto* box = new QCheckBox(QString::fromStdString(key).toUpper());
        box->setToolTip(Qt::convertFromPlainText(
            app::tr("The formats a table's or list's save icon writes what it shows in. With neither ticked, "
                    "only graphs are saved at one click."),
            Qt::WhiteSpaceNormal));
        box->setChecked(std::find(settings_.quick_save_formats.begin(), settings_.quick_save_formats.end(), key) !=
                        settings_.quick_save_formats.end());
        connect(box, &QCheckBox::toggled, this, [this] { on_edited(); });
        quick_format_boxes[key] = box;
        row->addWidget(box);
    }
    row->addSpacing(12);
    auto* note = new QLabel(app::tr("Graphs: always PNG. Each table's, list's and graph's save icon saves what it "
                                    "shows, at one click, in the export folder."));
    note->setWordWrap(true);
    theme::style(note, "color: {muted};");
    row->addWidget(note, 1);
    return quick;
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
        app::tr("In the export folder: next to the video while no export folder is set. No file is replaced: "
                "where a name is taken, the new files are numbered."),
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
    const auto shown_keys = [] {
        std::vector<std::string> keys;
        for (const MetricDefinition* definition : shown_metrics()) {
            keys.emplace_back(definition->key);
        }
        return keys;
    };
    weak_segments = new SegmentRulesEditor(shown_keys(), false);
    connect(&metric_order_signal(), &MetricOrderSignal::changed, weak_segments,
            [this, shown_keys] { weak_segments->set_keys(shown_keys()); });
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

    // Default metrics: all of them, in the two sections Add/remove metrics
    // has, side by side, Specialized open (MetricList).
    auto* metrics_row = new QHBoxLayout;
    auto* metrics_label = new QLabel(app::tr("Default metrics:"));
    metrics_row->addWidget(metrics_label, 0, Qt::AlignTop);
    default_metrics = new MetricList(MetricList::Place::settings);
    std::set<std::string> ticked;
    for (const MetricSection section : {MetricSection::general_purpose, MetricSection::specialized}) {
        for (const std::string_view key : metrics_in(section)) {
            if (*settings_.default_compute(key)) {
                ticked.emplace(key);
            }
        }
    }
    default_metrics->set_ticked(ticked);
    connect(default_metrics, &MetricList::ticks_changed, this, [this] { on_edited(); });
    metrics_row->addWidget(default_metrics, 1);
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
    // Its list laid out with the first name's size for every name: put in
    // its box, the combo box's font changes and Qt lays the list out again,
    // which measured every name and so looked for the Chinese, Japanese,
    // Korean and Thai fonts too (see below): a quarter of a second the first
    // time and some 20 MB, as the Settings tab was first opened. They are
    // looked for when the list is first opened.
    if (auto* names = qobject_cast<QListView*>(language->view())) {
        names->setUniformItemSizes(true);
    }
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
    // Long names that differ only at their ends (an encode's CRF, its
    // encoder) shortened where the window shows them.
    video_names = new QComboBox;
    video_names->addItem(app::tr("What tells them apart"), "differs");
    video_names->addItem(app::tr("First and last characters"), "ends");
    video_names->addItem(app::tr("Whole names"), "full");
    video_names->setCurrentIndex(std::max(0, video_names->findData(text_of(settings_.video_names))));
    video_names->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    video_names->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    video_names->setToolTip(
        app::tr("How test videos' names are shown in Metric Graphs, Video Compare and Worst Frames. What tells them "
                "apart leaves "
                "out the start and end the videos shown share: \"…CRF 20\", \"…CRF 22\". Tooltips, exports and "
                "file names keep the whole name."));
    connect(video_names, &QComboBox::currentIndexChanged, this, [this] {
        video_name_ends->setVisible(video_names->currentData().toString() == "ends");
        on_edited();
    });
    video_name_start = new QSpinBox;
    video_name_start->setRange(0, 200);
    video_name_start->setValue(std::clamp(settings_.video_name_start, 0, 200));
    connect(video_name_start, &QSpinBox::valueChanged, this, [this] { on_edited(); });
    video_name_end = new QSpinBox;
    video_name_end->setRange(0, 200);
    video_name_end->setValue(std::clamp(settings_.video_name_end, 0, 200));
    connect(video_name_end, &QSpinBox::valueChanged, this, [this] { on_edited(); });
    video_name_ends = new QWidget;
    auto* ends_row = new QHBoxLayout(video_name_ends);
    ends_row->setContentsMargins(0, 0, 0, 0);
    ends_row->addWidget(new QLabel(app::tr("Characters from the start:")));
    ends_row->addWidget(video_name_start);
    ends_row->addWidget(new QLabel(app::tr("from the end:")));
    ends_row->addWidget(video_name_end);
    video_name_ends->setVisible(video_names->currentData().toString() == "ends");
    auto* names_row = new QHBoxLayout;
    names_row->addWidget(new QLabel(app::tr("Video names:")));
    names_row->addWidget(video_names);
    names_row->addWidget(video_name_ends);
    names_row->addStretch(1);
    window_layout->addLayout(names_row);
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
    confirm_close = new QCheckBox(app::tr("Ask before closing the window during a run"));
    confirm_close->setToolTip(app::tr("Closing the window stops a run in progress. Ticked, the window asks first."));
    confirm_close->setChecked(settings_.confirm_close_during_run);
    connect(confirm_close, &QCheckBox::toggled, this, [this] { on_edited(); });
    window_layout->addWidget(confirm_close);
    // The list of the window's keys (shortcuts.h), also F1.
    auto* keys_button = new QPushButton(app::tr("Keyboard shortcuts"));
    keys_button->setObjectName("keyboardShortcuts");
    connect(keys_button, &QPushButton::clicked, this, &SettingsPanel::shortcuts_requested);
    auto* keys_row = new QHBoxLayout;
    keys_row->addWidget(keys_button);
    keys_row->addStretch(1);
    window_layout->addLayout(keys_row);
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
    default_metrics->set_ticked(key, checked);
}

void SettingsPanel::show_film_grain_default() {
    const QSignalBlocker blocker(film_grain_default);
    const int index = film_grain_default->findData(QString::fromStdString(settings_.film_grain_default));
    film_grain_default->setCurrentIndex(index < 0 ? 0 : index);  // an unknown value asks
}

void SettingsPanel::show_confirm_close() {
    const QSignalBlocker blocker(confirm_close);
    confirm_close->setChecked(settings_.confirm_close_during_run);
}

bool SettingsPanel::deleting_everything() const { return delete_scope->currentData().toString() == "all"; }

std::pair<std::optional<std::chrono::system_clock::time_point>, std::optional<std::chrono::system_clock::time_point>>
SettingsPanel::deletion_range() const {
    const QString scope = delete_scope->currentData().toString();
    const auto at = [](const QDate& date) {  // the day's start, here
        return std::chrono::system_clock::time_point(
            std::chrono::milliseconds(date.startOfDay().toMSecsSinceEpoch()));
    };
    if (scope == "older") {
        return {std::nullopt, std::chrono::system_clock::now() - std::chrono::days(delete_days->value())};
    }
    if (scope == "between") {
        const QDate from = std::min(delete_from->date(), delete_to->date());
        const QDate to = std::max(delete_from->date(), delete_to->date());
        return {at(from), at(to.addDays(1))};  // the last day whole
    }
    return {std::nullopt, std::nullopt};
}

void SettingsPanel::show_delete_scope() {
    const QString scope = delete_scope->currentData().toString();
    delete_days->setVisible(scope == "older");
    if (auto* label = qobject_cast<QWidget*>(delete_scope->property("daysLabel").value<QObject*>())) {
        label->setVisible(scope == "older");
    }
    delete_from->setVisible(scope == "between");
    delete_to->setVisible(scope == "between");
    if (auto* label = qobject_cast<QWidget*>(delete_scope->property("toLabel").value<QObject*>())) {
        label->setVisible(scope == "between");
    }
}

void SettingsPanel::pulse_delete_preview() {
    // Bold in the text's colour, then, after a moment, faded back to the
    // muted line it is: each result of Delete... (what it deleted, or why
    // it deleted nothing) drawn to the eye once (the user, 2026-11-25),
    // without a window or a sound. The previews under it aren't.
    // The look's text, both interfaces: the panel's, as the line's own style
    // sheet makes its palette the muted colour.
    const QColor from = palette().color(QPalette::WindowText);
    const QColor to = theme::color("muted");
    // Colour and weight together, bold to regular: the weight dropped in one
    // step read as a snap, the colour's fade going unseen beside it (the
    // user, 2026-11-25). Held a moment first, then two seconds of fade.
    const auto at = [this, from, to](double t) {
        const auto mix = [t](int a, int b) { return static_cast<int>(std::lround(a + (b - a) * t)); };
        const QColor colour(mix(from.red(), to.red()), mix(from.green(), to.green()), mix(from.blue(), to.blue()));
        delete_preview->setStyleSheet(
            QStringLiteral("color: %1; font-weight: %2;").arg(colour.name()).arg(mix(700, 400)));
    };
    if (delete_pulse_ == nullptr) {
        delete_pulse_ = new QVariantAnimation(this);
        delete_pulse_->setDuration(2000);
        delete_pulse_->setStartValue(0.0);
        delete_pulse_->setEndValue(1.0);
        delete_pulse_->setEasingCurve(QEasingCurve::InOutQuad);
        connect(delete_pulse_, &QVariantAnimation::finished, this,
                [this] { theme::style(delete_preview, "color: {muted};"); });
    }
    delete_pulse_->stop();
    disconnect(delete_pulse_, &QVariantAnimation::valueChanged, this, nullptr);
    connect(delete_pulse_, &QVariantAnimation::valueChanged, this,
            [at](const QVariant& value) { at(value.toDouble()); });
    at(0.0);
    QTimer::singleShot(2000, delete_pulse_, [this] { delete_pulse_->start(); });
}

void SettingsPanel::show_delete_preview() {
    if (!deleted_note_.isEmpty()) {  // the count after a deletion: what it deleted, once
        delete_preview->setText(deleted_note_);
        deleted_note_.clear();
        return;
    }
    // From the count, which reads each comparison's time and size at no cost.
    const bool counted = counting_->isFinished() && counting_->future().resultCount() > 0 &&
                         counted_folder_ == cache_folder();
    if (!counted) {
        delete_preview->setText(app::tr("Counting the saved results..."));
        return;
    }
    const CacheSummary summary = counting_->result();
    const auto [from, to] = deletion_range();
    int count = 0;
    std::uintmax_t bytes = 0;
    for (const CacheSummary::Comparison& saved : summary.saved) {
        if ((!from || saved.saved >= *from) && (!to || saved.saved < *to)) {
            ++count;
            bytes += saved.bytes;
        }
    }
    if (deleting_everything()) {
        bytes = summary.bytes;  // the bitrate readings too
    }
    delete_preview->setText(count == 0 ? app::tr("No saved results in that range.")
                                       : ntr("Would delete {count} comparison's saved scores, {size}.",
                                             "Would delete {count} comparisons' saved scores, {size}.", count,
                                             {{"size", saved_size_text(bytes)}}));
}

void SettingsPanel::delete_saved_results() {
    if (cleared_) {
        delete_preview->setText(app::tr("Saved results are already being deleted."));
        pulse_delete_preview();
        return;
    }
    // Read first, off the window's thread: the question says how many of
    // them can't be scored again, which each comparison's context tells.
    deletion_folder_ = cache_folder();
    delete_button->setEnabled(false);
    delete_preview->setText(app::tr("Counting the saved results..."));
    deletion_listing_->setFuture(QtConcurrent::run(
        [folder = deletion_folder_] { return list_saved_comparisons(folder); }));
}

void SettingsPanel::wait_for_deletion_count() {
    deletion_listing_->waitForFinished();
    QCoreApplication::sendPostedEvents(deletion_listing_);
}

void SettingsPanel::wait_for_count() {
    counting_->waitForFinished();
    QCoreApplication::sendPostedEvents(counting_);
}

void SettingsPanel::ask_and_delete(const std::vector<SavedComparison>& listed) {
    delete_button->setEnabled(!run_lock_note->isVisibleTo(this));
    show_delete_preview();
    const std::filesystem::path directory = deletion_folder_;
    const bool everything = deleting_everything();
    const auto [from, to] = deletion_range();
    std::vector<SavedComparison> chosen;
    for (const SavedComparison& saved : listed) {
        if ((!from || saved.saved >= *from) && (!to || saved.saved < *to)) {
            chosen.push_back(saved);
        }
    }
    if (chosen.empty() && !(everything && counting_->isFinished() && counting_->future().resultCount() > 0 &&
                            counting_->result().bytes > 0)) {
        delete_preview->setText(app::tr("No saved results in that range: nothing deleted."));
        pulse_delete_preview();
        return;
    }
    SavedDeletion deletion = saved_deletion(chosen, directory);
    deletion.everything = everything;
    if (everything && counting_->isFinished() && counting_->future().resultCount() > 0 &&
        counted_folder_ == directory) {
        deletion.bytes = counting_->result().bytes;  // the bitrate readings too
    }
    // A run may have started while it was asked: then nothing is deleted.
    if (dialogs_.message(saved_deletion_message(deletion)).button != 0 || run_lock_note->isVisibleTo(this) ||
        cleared_) {
        return;
    }
    // Saved scores are written and deleted through the same queue, one after
    // another. If a run just finished, its scores therefore land before they
    // are deleted, rather than coming back after the user deleted them.
    // `directory` is the folder the question just named: if the setting
    // changes before the queue gets to this, it must not delete a folder the
    // user was never asked about.
    cleared_ = std::make_shared<ClearedResults>();
    if (everything) {
        file_writes_.submit(app::tr("delete saved results"), [directory, cleared = cleared_] {
            // The black bars kept with the saved results go too: deleted,
            // they start afresh.
            forget_crop_answers();
            std::error_code error;
            const std::filesystem::path black_bars = directory / black_bars_file;
            std::filesystem::remove(black_bars, error);
            if (error) {
                app_log::write(app_log::Level::warning, "vml.crop_detect",
                               "Could not delete " + platform::path_text(black_bars) + ": " + error.message());
            }
            *cleared = vml::clear_saved_results(directory, false);
        });
    } else {
        file_writes_.submit(app::tr("delete saved results"), [directory, from, to, cleared = cleared_] {
            *cleared = vml::delete_saved_between(directory, from, to);
        });
    }
    delete_preview->setText(app::tr("Deleting saved results..."));
}

// --------------------------------------------------------------- inside the tab

std::filesystem::path SettingsPanel::cache_folder() const {
    return settings_.cache_dir_path().value_or(default_cache_dir());
}

void SettingsPanel::show_cache_summary() {
    // How many, their size, and the dates they were last saved between.
    const CacheSummary summary = counting_->result();
    if (summary.saved.empty()) {
        cache_summary->setText(
            app::tr("No saved results in {directory}", {{"directory", text_of(counted_folder_)}}));
    } else {
        const QString first = saved_date_text(summary.saved.back().saved);
        const QString last = saved_date_text(summary.saved.front().saved);
        const std::vector<NamedValue> values{{"size", saved_size_text(summary.bytes)},
                                                  {"first", first},
                                                  {"last", last},
                                                  {"directory", text_of(counted_folder_)}};
        cache_summary->setText(first == last
                                   ? ntr("{count} saved result, {size}, saved {first}, in {directory}",
                                         "{count} saved results, {size}, saved {first}, in {directory}",
                                         summary.comparisons, values)
                                   : ntr("{count} saved result, {size}, saved {first} to {last}, in {directory}",
                                         "{count} saved results, {size}, saved {first} to {last}, in {directory}",
                                         summary.comparisons, values));
        if (!dates_chosen_) {  // the dates to delete between start as the saved ones'
            const QSignalBlocker from_blocker(delete_from);
            const QSignalBlocker to_blocker(delete_to);
            const auto day = [](std::chrono::system_clock::time_point when) {
                return QDateTime::fromMSecsSinceEpoch(
                           std::chrono::duration_cast<std::chrono::milliseconds>(when.time_since_epoch()).count())
                    .date();
            };
            delete_from->setDate(day(summary.saved.back().saved));
            delete_to->setDate(day(summary.saved.front().saved));
        }
    }
    show_delete_preview();
}

void SettingsPanel::on_file_writes_idle() {
    // Once the clearing has run: a write queued before it and finished just
    // as it was asked for says the queue is idle too, too early.
    if (!cleared_ || !file_writes_.idle()) {
        return;
    }
    const ClearedResults cleared = *cleared_;
    cleared_.reset();
    // Said beside Delete..., where it was asked for (docs/ARCHITECTURE.md,
    // "Asking and telling the user"), until the count shown again replaces
    // it with what the row would delete now.
    deleted_note_ = ntr("Deleted the saved scores of {count} comparison.",
                        "Deleted the saved scores of {count} comparisons.", cleared.removed);
    delete_preview->setText(deleted_note_);
    pulse_delete_preview();
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
    if (settings_.video_names != before.video_names || settings_.video_name_start != before.video_name_start ||
        settings_.video_name_end != before.video_name_end) {
        emit video_names_changed();
    }
    if (settings_.saved_score_versions != before.saved_score_versions ||
        settings_.saved_score_pick != before.saved_score_pick ||
        settings_.saved_score_device != before.saved_score_device ||
        settings_.saved_score_engine != before.saved_score_engine) {
        emit saved_scores_changed();
    }
    if (settings_.ffmpeg_dir != before.ffmpeg_dir) {
        emit ffmpeg_folder_changed();  // after the save: the tools are looked for in the folder saved
    }
    // A change that shows only once the app starts again: the window offers
    // to restart it now. Changed back (VML Modern picked again, the language
    // the app started in), none is needed: said with an empty `what`, which
    // takes the note away.
    if (!error) {
        const bool language_later = settings_.language != started_language_;
        const bool interface_later = theme::interface_of(settings_.interface_style) == theme::Interface::classic &&
                                     theme::current_interface() == theme::Interface::modern;
        const QString what = language_later  ? app::tr("The new language shows when the app is next started.")
                             : interface_later ? app::tr("VML Classic shows when the app is next started.")
                                               : QString();
        if (what != restart_said_) {
            restart_said_ = what;
            emit restart_needed(what);
        }
    }
    // What happened, in the tab bar's indicator: the line at the tab's foot
    // was easily missed (docs/ARCHITECTURE.md, "Asking and telling the user").
    // After the restart's note is taken away, where a change went back: a
    // done note doesn't replace a kept one.
    if (error) {
        say(tr_message(text_of(*error)), ExportIndicator::Note::failed);
    } else {
        say(app::tr("Settings saved."));
    }
    if (settings_.ffmpeg_dir != before.ffmpeg_dir || settings_.cache_dir != before.cache_dir) {
        refresh_status();
    }
}

void SettingsPanel::read_controls() {
    settings_.ffmpeg_dir = ffmpeg_edit->text().toStdString();
    settings_.cache_dir = cache_edit->text().toStdString();
    settings_.export_dir = export_edit->text().toStdString();
    settings_.saved_score_versions = saved_versions_each->isChecked() ? "each" : "used";
    settings_.saved_score_pick = saved_pick->currentData().toString().toStdString();
    settings_.saved_score_device = saved_device->currentData().toString().toStdString();
    settings_.saved_score_engine = saved_engine->currentData().toString().toStdString();
    settings_.saved_details_columns.clear();
    for (const std::string& key : saved_details_column_keys()) {
        if (saved_details_boxes.at(key)->isChecked()) {
            settings_.saved_details_columns.push_back(key);
        }
    }
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
    settings_.quick_save_formats.clear();
    for (const std::string& key : export_format_keys()) {
        if (quick_format_boxes.at(key)->isChecked()) {
            settings_.quick_save_formats.push_back(key);
        }
    }
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
    const std::set<std::string> ticked = default_metrics->ticked();
    for (const MetricSection section : {MetricSection::general_purpose, MetricSection::specialized}) {
        for (const std::string_view key : metrics_in(section)) {
            *settings_.default_compute(key) = ticked.contains(std::string(key));
        }
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
    settings_.confirm_close_during_run = confirm_close->isChecked();
    settings_.language = language->currentData().toString().toStdString();
    settings_.theme = theme_buttons->checkedButton()->property("value").toString().toStdString();
    settings_.light_look = light_look_buttons->checkedButton()->property("value").toString().toStdString();
    settings_.dark_look = dark_look_buttons->checkedButton()->property("value").toString().toStdString();
    settings_.interface_style = interface_buttons->checkedId() == 1 ? "modern" : "classic";
    settings_.accent_colour = accent_value.toStdString();
    settings_.video_names = video_names->currentData().toString().toStdString();
    settings_.video_name_start = video_name_start->value();
    settings_.video_name_end = video_name_end->value();
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
                                     "deleting saved results are locked until the run ends, because the run uses "
                                     "them. Everything else can be changed now.");
        run_lock_note->setText(theme::current_interface() == theme::Interface::modern
                                   ? text
                                   : QStringLiteral("\U0001F512 ") + text);
    }
    run_lock_note->setVisible(active);
}

void SettingsPanel::apply_global_changes(const Settings& before) {
    if (settings_.theme != before.theme || settings_.light_look != before.light_look ||
        settings_.dark_look != before.dark_look) {
        // the window's changeEvent follows
        theme::apply_theme(text_of(settings_.theme), text_of(settings_.light_look), text_of(settings_.dark_look));
        if (accent_value.isEmpty()) {
            set_accent(QString());  // the accent box shows the one in use now (the look's, or the system's)
        }
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
    const ColorChoice choice =
        choose_color(this, title, background ? colours.background : colours.text, app::tr("Reset color"));
    if (choice.outcome == ColorChoice::Outcome::cancelled) {
        return;
    }
    button->setProperty("chosen",
                        choice.outcome == ColorChoice::Outcome::reset ? QString() : choice.color.name());
    on_edited();
}

void SettingsPanel::pick_directory(const QString& title, QLineEdit* edit) {
    const QString directory = dialogs_.choose_folder(title, edit->text());
    if (!directory.isEmpty()) {
        edit->setText(directory);
        on_edited();
    }
}

void SettingsPanel::say(const QString& text, ExportIndicator::Note kind) const {
    if (notify) {
        notify(text, kind);
    }
}

void SettingsPanel::on_copy_log() {
    std::optional<std::pair<std::string, bool>> shared;
    try {
        shared = app_log::log_text_to_share();
    } catch (const std::runtime_error& error) {
        say(app::tr("Could not read the log: {error}", {{"error", QString::fromUtf8(error.what())}}),
            ExportIndicator::Note::failed);
        return;
    }
    if (!shared) {
        say(app::tr("There is no log to copy yet."), ExportIndicator::Note::info);
        return;
    }
    const QString text = QString::fromStdString(shared->first);
    QApplication::clipboard()->setText(text);
    const long long lines = text.count('\n') + 1;
    say(shared->second ? ntr("Copied the log to the clipboard ({count} line, the middle left out -- "
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
        report_to(report_export, {{}, could_not_write(path), QString::fromUtf8(error.what())});
        return;
    }
    if (names.empty()) {
        say(app::tr("There is no log to export yet."), ExportIndicator::Note::info);
        return;
    }
    logger.info("Log exported to " + path.toStdString());
    report_to(report_export, {{path}, {}, {}});  // as any export says it: "Saved ..." and Open folder
}

}  // namespace vml::app
