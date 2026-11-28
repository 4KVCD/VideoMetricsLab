#include "options_panel.h"

#include <QAbstractSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QMargins>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStringList>
#include <QTime>
#include <QTimeEdit>
#include <QVBoxLayout>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

#include "click_twice.h"
#include "formatting.h"
#include "i18n.h"
#include "theme.h"
#include "window_helpers.h"
#include "vml/film_grain_scoring.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"
#include "vml/vship.h"
#include "vship_names.h"

// The panel's text is app::tr's: inside a Q_OBJECT class, tr() is Qt's.
namespace vml::app {

FilmGrainControl film_grain_control(const std::vector<FilmGrainVideo>& videos) {
    FilmGrainControl control;
    QString why;  // why none can be chosen, as the first video that cannot says it
    bool first = true;
    // The choice twinned rows hold, shown greyed out where every selected
    // video is one of them and they share it.
    std::optional<bool> held;
    bool all_held = !videos.empty();
    for (const FilmGrainVideo& video : videos) {
        const bool holds = video.twinned && video.info != nullptr && film_grain_choosable(*video.info);
        const bool held_with = holds && effective_film_grain(*video.info).scoring == GrainScoring::with;
        all_held = all_held && holds && (!held || *held == held_with);
        held = held_with;
        if (video.resolution_test || video.twinned || video.info == nullptr || !film_grain_choosable(*video.info)) {
            if (why.isEmpty()) {
                why = video.resolution_test ? tr("A resolution test's film grain is the reference's, chosen when "
                                                 "metrics are calculated.")
                      : video.twinned ? tr("This video was added a second time to be scored with its film grain and "
                                           "without it, side by side: each of its rows keeps its choice.")
                      : video.info == nullptr || !video.info->film_grain ? tr("Its film grain was not read.")
                      : !film_grain_note(*video.info).isEmpty()          ? film_grain_note(*video.info)
                                                                         : tr("It has no film grain.");
            }
            continue;
        }
        const bool with = effective_film_grain(*video.info).scoring == GrainScoring::with;
        control.with = first || control.with == with ? std::optional(with) : std::nullopt;
        first = false;
        control.enabled = true;
    }
    if (!control.enabled) {
        control.with = all_held ? held : std::nullopt;
    }
    control.tooltip = control.enabled ? tr("Whether the scores are made with the film grain the video's stream asks "
                                           "the player to add, as viewers see the video, or without it, from the "
                                           "coded pictures.")
                                      : why;
    return control;
}

namespace {

// Manual's crop boxes in their columns' order, as their fields name them.
const std::array<std::pair<const char*, int CropEdges::*>, 4> crop_edges{
    {{"top", &CropEdges::top}, {"bottom", &CropEdges::bottom}, {"left", &CropEdges::left}, {"right", &CropEdges::right}}};

QString manual_crop_field(bool source, std::size_t column) {
    return QStringLiteral("manual_crop:%1:%2").arg(source ? "source" : "test", crop_edges[column].first);
}

}  // namespace

std::optional<ManualCropBox> manual_crop_box_of(const QString& field) {
    const QStringList parts = field.split(':');
    if (parts.size() != 3 || parts[0] != "manual_crop" || (parts[1] != "source" && parts[1] != "test")) {
        return std::nullopt;
    }
    for (const auto& [name, edge] : crop_edges) {
        if (parts[2] == name) {
            return ManualCropBox{parts[1] == "source" ? &VmafOptions::manual_source_crop : &VmafOptions::manual_test_crop,
                                 edge};
        }
    }
    return std::nullopt;
}

namespace {

// The scaling algorithms, as FFmpeg's scale filter names them.
const QStringList scale_algorithms{"bicubic", "lanczos", "bilinear", "spline"};

// The GPU decode dropdown's entries, in its order.
#ifdef __APPLE__
// A Mac decodes with its own media engine, whichever maker is asked.
const std::vector<GpuVendor> gpu_vendor_entries{GpuVendor::automatic};
#else
const std::vector<GpuVendor> gpu_vendor_entries{GpuVendor::automatic, GpuVendor::nvidia, GpuVendor::intel,
                                                GpuVendor::amd};
#endif

// The index of `choice` in a model dropdown; Auto's for one not there.
int index_of(const std::vector<ModelChoice>& choices, const std::string& choice) {
    const auto found = std::find_if(choices.begin(), choices.end(),
                                    [&choice](const ModelChoice& item) { return item.choice == choice; });
    return found == choices.end() ? 0 : static_cast<int>(found - choices.begin());
}

}  // namespace

const std::vector<ModelChoice>& model_choices() {
    static const std::vector<ModelChoice> choices{
        {N_("Auto (analysis resolution)"), std::string(auto_model_choice)},
        {N_("VMAF v0.6.1 (standard viewing)"), "version=vmaf_v0.6.1"},
        {N_("VMAF 4K v0.6.1 (4K / large-screen viewing)"), "version=vmaf_4k_v0.6.1"},
        {N_("Custom model file..."), std::string(custom_model_choice)},
    };
    return choices;
}

namespace {

// Whether VMAF can be calculated on a GPU here: one of the GPUs found (an
// NVIDIA, Intel or AMD one), or on macOS, where every Mac has one and it is
// not among them (Vulkan on Metal, vmaf_vulkan; a run whose GPU fails
// calculates on the CPU, as anywhere).
bool vmaf_gpu_possible(const std::vector<GpuVendor>& gpu_vendors) {
#ifdef __APPLE__
    static_cast<void>(gpu_vendors);
    return true;
#else
    return !gpu_vendors.empty();
#endif
}

// A time in whole seconds (HH:mm:ss), applied when it is entered or a step
// is taken, not at every digit typed: each change takes the rows' scores
// away.
QTimeEdit* new_seconds_edit() {
    auto* edit = new QTimeEdit;
    // In a right-to-left window (Arabic) QTimeEdit writes the sections
    // backwards ("00:00:00" reversed); times read left to right in every
    // language. The direction must be set before the display format.
    edit->setLayoutDirection(Qt::LeftToRight);
    edit->setDisplayFormat("HH:mm:ss.zzz");
    edit->setTime(QTime(0, 0));
    edit->setKeyboardTracking(false);
    return edit;
}

// The forms' labels as wide as the widest of them, so that their fields start
// at one edge, as in one form.
void same_label_width(std::initializer_list<QFormLayout*> forms) {
    std::vector<QWidget*> labels;
    int widest = 0;
    for (QFormLayout* form : forms) {
        for (int row = 0; row < form->rowCount(); ++row) {
            if (QLayoutItem* item = form->itemAt(row, QFormLayout::LabelRole); item != nullptr && item->widget()) {
                labels.push_back(item->widget());
                widest = std::max(widest, item->widget()->sizeHint().width());
            }
        }
    }
    for (QWidget* label : labels) {
        label->setMinimumWidth(widest);
    }
}

}  // namespace

OptionsPanel::OptionsPanel(const std::vector<GpuVendor>& gpu_vendors, QWidget* parent)
    : QGroupBox(app::tr("Options"), parent), vmaf_gpu_possible_(vmaf_gpu_possible(gpu_vendors)) {
    // Which videos it edits, then three boxes side by side (video
    // preparation, metric-specific settings, performance), then a note on
    // what affects scores.
    auto* layout = new QVBoxLayout(this);
    target_label = new QLabel;
    theme::style(target_label, "color: {muted}; font-style: {hint_style};");
    target_label->setWordWrap(true);
    layout->addWidget(target_label);

    // Side by side rather than stacked: the panel is the window's full width,
    // so stacking the boxes would leave empty space to the right and push the
    // video table up.
    auto* columns = new QHBoxLayout;
    std::vector<QFormLayout*> forms;
    const auto add_box = [&columns, &forms](const QString& title) {
        auto* box = new QGroupBox(title);
        forms.push_back(new QFormLayout(box));
        columns->addWidget(box, 1);
        return forms.back();
    };
    build_preparation(add_box(app::tr("Video preparation (calculation only)")));
    // The middle column: one box in Classic, "Metric-specific settings". In
    // VML Modern two cards, one above the other: VMAF (its three models and
    // the frame subsample) and CVVDP (its display), which have nothing in
    // common but the column (theme::classic_box, theme::container).
    auto* metric_box = new QGroupBox;
    theme::classic_box(metric_box, app::tr("Metric-specific settings"));
    metric_box->setObjectName("metricSettings");
    auto* metric_column = new QVBoxLayout(metric_box);
    // In Modern none, so that the cards' tops are level with the side boxes'
    // (Modern's box draws nothing of its own around them); again after a
    // switch to Modern while the app runs.
    theme::modern_margins(metric_box, QMargins());
    const auto add_card = [&metric_column, &forms](const QString& title) {
        QBoxLayout* card_layout = nullptr;
        metric_column->addWidget(theme::container("card", false, &card_layout));
        auto* heading = new QLabel(title);
        theme::set_role(heading, "title");
        theme::modern_only(heading);
        card_layout->addWidget(heading);
        auto* form = new QFormLayout;
        card_layout->addLayout(form);
        forms.push_back(form);
        return form;
    };
    QFormLayout* vmaf_form = add_card(QStringLiteral("VMAF"));
    QFormLayout* cvvdp_form = add_card(QStringLiteral("CVVDP"));
    metric_column->addStretch(1);
    columns->addWidget(metric_box, 1);
    build_metric_settings(vmaf_form);
    build_cvvdp(cvvdp_form);
    // One label column across the two, as the one form Classic had.
    same_label_width({vmaf_form, cvvdp_form});
    build_performance(add_box(app::tr("Performance")), gpu_vendors);
    layout->addLayout(columns);

    auto* metrics_hint = new QLabel(
        app::tr("Cropping, scaling and film grain here affect scores. Video Compare's tone mapping and playback "
                "resolution are display-only settings and do not change calculated metrics."));
    theme::style(metrics_hint, "color: {muted}; font-style: {hint_style};");
    metrics_hint->setWordWrap(true);
    layout->addWidget(metrics_hint);
    layout->addStretch(1);

    // Room for longer words than English's. A field that does not fit beside
    // its label goes under it, and a dropdown can be narrower than its
    // longest entry, which it shows in full when opened. Without this, the
    // German panel needs 1,590 px, more than the window's default 1,280, and
    // is cut off at the right. In English nothing moves.
    for (QFormLayout* form : forms) {
        form->setRowWrapPolicy(QFormLayout::WrapLongRows);
    }
    for (QComboBox* combo : findChildren<QComboBox*>()) {
        combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        combo->setMinimumContentsLength(12);
    }
}

void OptionsPanel::build_preparation(QFormLayout* form) {
    // Black bars, range, scaling algorithm and resolution mismatch.
    crop_combo = new QComboBox;
    // The entries' data: the CropMode's number.
    crop_combo->addItem(app::tr("Auto-detect (recommended)"), static_cast<int>(CropMode::automatic));
    crop_combo->addItem(app::tr("Source only (test video already cropped)"), static_cast<int>(CropMode::source_only));
    crop_combo->addItem(app::tr("None (use full frame)"), static_cast<int>(CropMode::none));
    crop_combo->addItem(app::tr("Manual (enter the crops)"), static_cast<int>(CropMode::manual));
    crop_combo->setItemData(1,
                            app::tr("The source's black bars are detected and cut; the test video is compared at its "
                                    "full frame, for an encode cropped when it was made."),
                            Qt::ToolTipRole);
    crop_combo->setItemData(3,
                            app::tr("Nothing is detected: the lines entered are cut from each video, as an encoder's "
                                    "crop settings cut them. A video with nothing entered is compared at its full "
                                    "frame."),
                            Qt::ToolTipRole);
    connect(crop_combo, &QComboBox::currentIndexChanged, this, [this] {
        show_manual_crop();
        emit field_edited("crop_mode");
    });
    form->addRow(app::tr("Black-bar handling:"), crop_combo);
    build_manual_crop(form);

    // The range compared: from its start to its end, to the millisecond (an
    // end of 0: to the videos' end), or from a frame to a frame.
    range_unit_combo = new QComboBox;
    range_unit_combo->addItem(app::tr("Time"));
    range_unit_combo->addItem(app::tr("Frames"));
    connect(range_unit_combo, &QComboBox::currentIndexChanged, this, [this] {
        show_range_unit();
        emit field_edited("range_in_frames");
    });
    range_start_edit = new_seconds_edit();
    range_end_edit = new_seconds_edit();
    connect(range_start_edit, &QTimeEdit::timeChanged, this, [this] {
        show_range_unit();
        emit field_edited("range_start");
    });
    connect(range_end_edit, &QTimeEdit::timeChanged, this, [this] {
        show_range_unit();
        emit field_edited("range_end");
    });
    range_first_spin = new QSpinBox;
    range_first_spin->setRange(0, std::numeric_limits<int>::max());
    range_first_spin->setKeyboardTracking(false);
    range_last_spin = new QSpinBox;
    // Below the first frame: to the videos' end.
    range_last_spin->setRange(-1, std::numeric_limits<int>::max());
    range_last_spin->setSpecialValueText(app::tr("End"));
    range_last_spin->setKeyboardTracking(false);
    connect(range_first_spin, &QSpinBox::valueChanged, this, [this] { emit field_edited("range_start"); });
    connect(range_last_spin, &QSpinBox::valueChanged, this, [this] { emit field_edited("range_end"); });
    auto* range_row = new QHBoxLayout;
    range_row->addWidget(range_unit_combo);
    range_row->addWidget(range_start_edit);
    range_row->addWidget(range_first_spin);
    range_row->addWidget(new QLabel(app::tr("to")));
    range_row->addWidget(range_end_edit);
    range_row->addWidget(range_last_spin);
    range_row->addStretch();
    auto* range_label = new QLabel(app::tr("Range:"));
    range_label->setToolTip(app::tr("The part of the videos compared: in time, to the millisecond, from the first "
                                    "time to the second, where an end of 00:00:00.000 compares to the end of the "
                                    "videos; or in frames, from the first frame to the last, both compared, counted "
                                    "from 0 as Video Compare counts them."));
    form->addRow(range_label, range_row);
    range_frames_note = new QLabel(app::tr("CVVDP scores whole seconds: its range is rounded to the nearest second."));
    range_frames_note->setWordWrap(true);
    theme::style(range_frames_note, "color: {muted}; font-style: {hint_style};");
    form->addRow(QString(), range_frames_note);
    // Seconds until a row in frames is shown. Hidden, not show_range_unit():
    // shown before the form has a parent, a field would be a window.
    for (QWidget* field : std::initializer_list<QWidget*>{range_first_spin, range_last_spin, range_frames_note}) {
        field->hide();
    }

    scale_algo_combo = new QComboBox;
    scale_algo_combo->addItems(scale_algorithms);
    connect(scale_algo_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("scale_algorithm"); });
    form->addRow(app::tr("Scaling algorithm:"), scale_algo_combo);

    scale_direction_combo = new QComboBox;
    // "scaled", not up or down: a test video can be larger than its source.
    // The row's Scaling note says which way, from the two sizes.
    scale_direction_combo->addItems({app::tr("Source scaled to test"), app::tr("Test scaled to source"),
                                     app::tr("Test both (adds a comparison row)")});
    scale_direction_combo->setToolTip(
        app::tr("When the two resolutions differ: either evaluate quality at the resolution actually\n"
                "delivered (source scaled to match test -- the default), or as if the test\n"
                "video were upscaled back to the reference's native resolution for playback.\n"
                "\"Test both\" doesn't change this row -- it adds a second row for the same test\n"
                "file using the other direction, so you can run and compare both."));
    connect(scale_direction_combo, &QComboBox::currentIndexChanged, this, &OptionsPanel::scale_direction_chosen);
    form->addRow(app::tr("Resolution mismatch:"), scale_direction_combo);

    // The entries' data: with the grain (true) or without; an entry without
    // data is shown in front of them for videos that differ ("Mixed"), and
    // empty where none can be chosen.
    film_grain_combo = new QComboBox;
    film_grain_combo->addItem(app::tr("With film grain (what viewers see)"), true);
    film_grain_combo->addItem(app::tr("Without film grain (the coded pictures only)"), false);
    connect(film_grain_combo, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (const QVariant with = film_grain_combo->itemData(index); with.isValid()) {
            emit film_grain_chosen(with.toBool());
        }
    });
    form->addRow(app::tr("Film grain:"), film_grain_combo);
}

void OptionsPanel::show_film_grain(const std::vector<FilmGrainVideo>& videos) {
    const FilmGrainControl control = film_grain_control(videos);
    const QSignalBlocker blocker(film_grain_combo);
    const bool has_blank = !film_grain_combo->itemData(0).isValid();
    if (!control.with && !has_blank) {
        film_grain_combo->insertItem(0, QString(), QVariant());
    } else if (control.with && has_blank) {
        film_grain_combo->removeItem(0);
    }
    if (!control.with) {
        // Empty where nothing can be chosen (the tooltip says why).
        film_grain_combo->setItemText(0, control.enabled ? app::tr("Mixed (the selected videos differ)") : QString());
    }
    film_grain_combo->setCurrentIndex(control.with ? film_grain_combo->findData(*control.with) : 0);
    film_grain_combo->setEnabled(control.enabled);
    film_grain_combo->setToolTip(control.tooltip);
}

void OptionsPanel::build_metric_settings(QFormLayout* form) {
    // The VMAF models and the frame subsample (CVVDP's display: build_cvvdp).
    model_combo = new QComboBox;
    for (const ModelChoice& choice : model_choices()) {
        model_combo->addItem(app::tr(choice.name));
    }
    connect(model_combo, &QComboBox::currentIndexChanged, this, &OptionsPanel::model_chosen);
    model_combo->setToolTip(
        app::tr("The VMAF v0.6.1 column's model. Auto picks the standard or the 4K model from the size frames are "
                "compared at, after black bars are removed and one video is scaled to the other."));
    form->addRow(app::tr("VMAF v0.6.1 model:"), model_combo);
    model_v1_combo = new QComboBox;
    for (const ModelChoice& choice : v1_model_choices()) {
        model_v1_combo->addItem(app::tr(choice.name));
    }
    connect(model_v1_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("model_v1"); });
    model_v1_combo->setToolTip(
        app::tr("The VMAF v1 column's model (Netflix's bundled VMAF v1 models). Auto picks the 1080p model at three "
                "screen heights, or the 4K model at one and a half for 4K comparisons, as for VMAF v0.6.1."));
    form->addRow(app::tr("VMAF v1 model:"), model_v1_combo);
    model_hdr_combo = new QComboBox;
    for (const ModelChoice& choice : hdr_model_choices()) {
        model_hdr_combo->addItem(app::tr(choice.name));
    }
    connect(model_hdr_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("model_hdr"); });
    model_hdr_combo->setToolTip(
        app::tr("The HDR VMAF column's model (Netflix's bundled HDR VMAF models). Auto picks the 1080p model at three "
                "screen heights, or the 4K model at one and a half for 4K comparisons, as for VMAF v0.6.1. PQ (HDR10) "
                "videos only."));
    form->addRow(app::tr("HDR VMAF model:"), model_hdr_combo);

    subsample_spin = new QSpinBox;
    subsample_spin->setRange(1, 60);
    subsample_spin->setValue(1);
    subsample_spin->setKeyboardTracking(false);  // as the range: not at every digit typed
    connect(subsample_spin, &QSpinBox::valueChanged, this, [this] { emit field_edited("n_subsample"); });
    form->addRow(app::tr("libvmaf frame subsample:"), subsample_spin);
    subsample_spin->setToolTip(
        app::tr("1 = every frame. Applies to VMAF, PSNR and SSIM. XPSNR is computed every frame; combined runs retain "
                "values at libvmaf's sampled frames."));
}

void OptionsPanel::build_cvvdp(QFormLayout* form) {
    // CVVDP's display: a preset, the display itself, the buttons that add,
    // edit and delete presets, and whether the video is scaled to fill the
    // display (see vml/cvvdp.h).
    cvvdp_preset_combo = new QComboBox;
    cvvdp_preset_combo->setToolTip(
        app::tr("The display CVVDP predicts visible differences on. The score depends on the display as much as on "
                "the videos, so compare videos scored for the same one.\n\nNew videos start with the preset chosen "
                "in Settings. Saving a preset of your own makes it that default."));
    connect(cvvdp_preset_combo, &QComboBox::currentIndexChanged, this, &OptionsPanel::cvvdp_preset_chosen);
    form->addRow(app::tr("CVVDP display:"), cvvdp_preset_combo);
    cvvdp_display_label = new QLabel;
    theme::style(cvvdp_display_label, "color: {muted};");
    cvvdp_display_label->setWordWrap(true);
    form->addRow(QString(), cvvdp_display_label);

    auto* buttons = new QHBoxLayout;
    cvvdp_add_button = new QPushButton(app::tr("Add preset"));
    cvvdp_add_button->setToolTip(
        app::tr("Make a new preset of your own, starting from the display shown here. It becomes the display newly "
                "added videos start with; the selected videos keep theirs until you choose it for them."));
    connect(cvvdp_add_button, &QPushButton::clicked, this, &OptionsPanel::cvvdp_add_requested);
    cvvdp_edit_button = new QPushButton(app::tr("Edit display"));
    cvvdp_edit_button->setToolTip(
        app::tr("Change the display's size, distance, brightness, room light and more; rename your preset or save "
                "the display as a new one."));
    connect(cvvdp_edit_button, &QPushButton::clicked, this, &OptionsPanel::cvvdp_edit_requested);
    cvvdp_delete_button = new QPushButton(app::tr("Delete preset"));
    cvvdp_delete_button->setToolTip(app::tr("Delete this preset of yours, clicked twice. Videos using it keep their "
                                            "settings. Built-in presets cannot be deleted."));
    connect(cvvdp_delete_button, &QPushButton::clicked, this, [this] { on_cvvdp_delete_clicked(); });
    for (QPushButton* button : {cvvdp_add_button, cvvdp_edit_button, cvvdp_delete_button}) {
        buttons->addWidget(button);
    }
    // Clicked twice, as Remove all: the line under the buttons says what
    // the second click deletes. Another button, or another preset chosen,
    // lets go.
    cvvdp_delete_guard_ = new ClickTwice(this);
    cvvdp_delete_line = new QLabel;
    cvvdp_delete_line->setObjectName("cvvdpDeleteLine");
    cvvdp_delete_guard_->show_line_in(cvvdp_delete_line);
    buttons->addStretch(1);
    for (QPushButton* other : {cvvdp_add_button, cvvdp_edit_button}) {
        connect(other, &QPushButton::clicked, cvvdp_delete_guard_, &ClickTwice::disarm);
    }
    connect(cvvdp_preset_combo, &QComboBox::currentIndexChanged, cvvdp_delete_guard_, &ClickTwice::disarm);
    form->addRow(QString(), buttons);
    form->addRow(cvvdp_delete_line);  // under the buttons, the row's whole width: the panel is narrow

    // Another display beside the videos' own, each scored in a row of its own.
    cvvdp_other_button = new QPushButton(app::tr("Also score CVVDP on"));
    cvvdp_other_button->setToolTip(
        app::tr("Adds a row for each selected video that scores CVVDP alone on another display, beside the display "
                "of its own row. The other metrics are not calculated again. Export results lists every display in "
                "the video's entry."));
    connect(cvvdp_other_button, &QPushButton::clicked, this, &OptionsPanel::cvvdp_other_display_requested);
    auto* other = new QHBoxLayout;
    other->addWidget(cvvdp_other_button);
    other->addStretch(1);
    form->addRow(QString(), other);

    cvvdp_resize_check = new QCheckBox(app::tr("Scale the video to fill the display"));
    cvvdp_resize_check->setToolTip(
        app::tr("Off (the official default): the video is shown pixel for pixel, so a 1080p video covers a quarter "
                "of a 4K display. On: it is scaled, keeping its shape, to fill the display.\n\nSet per video: "
                "choosing or saving a display preset does not change it."));
    // checkStateChanged, not toggled: a partly ticked box (a mixed selection)
    // already counts as checked, so ticking it fully would not toggle
    // anything.
    connect(cvvdp_resize_check, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState state) {
        if (state != Qt::PartiallyChecked) {
            emit cvvdp_resize_toggled(state == Qt::Checked);
        }
    });
    form->addRow(QString(), cvvdp_resize_check);
}

void OptionsPanel::build_performance(QFormLayout* form, const std::vector<GpuVendor>& gpu_vendors) {
    // GPU decoding, where each metric is calculated, the GPUs found, and
    // libvmaf's threads.
    gpu_checkbox = new QCheckBox(app::tr("Use GPU decoding"));
#ifdef __APPLE__
    gpu_checkbox->setToolTip(app::tr(
        "Decodes the videos with the Mac's media engine (VideoToolbox) where it can decode their codec: every video "
        "for SSIMULACRA2, Butteraugli and CVVDP; for VMAF, VMAF NEG, VMAF v1, PSNR, SSIM and XPSNR, the resolution "
        "test and black-bar detection only videos deeper than 8 bits (8-bit ones decode faster in the app). Each "
        "video falls back to the CPU on its own if VideoToolbox can't decode it."));
#else
    gpu_checkbox->setToolTip(
        app::tr("Hardware-decodes the reference and the test video. Each is decided separately, and either one falls "
                "back to the CPU on its own if this GPU can't decode its format."));
#endif
    gpu_checkbox->setChecked(true);
    gpu_vendor_combo = new QComboBox;
#ifdef __APPLE__
    gpu_vendor_combo->addItem(app::tr("Apple (VideoToolbox)"));
#else
    for (const GpuVendor vendor : gpu_vendor_entries) {
        gpu_vendor_combo->addItem(vendor == GpuVendor::automatic ? app::tr("Auto-detect") : qt_text(maker_name(vendor)));
    }
#endif
    connect(gpu_checkbox, &QCheckBox::toggled, gpu_vendor_combo, &QWidget::setEnabled);
    connect(gpu_checkbox, &QCheckBox::toggled, this, [this] { emit field_edited("gpu"); });
    connect(gpu_vendor_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("gpu"); });
    auto* gpu_row = new QHBoxLayout;
    gpu_row->addWidget(gpu_checkbox);
    gpu_row->addWidget(gpu_vendor_combo);
    form->addRow(app::tr("GPU decode:"), gpu_row);

    // VMAF v0.6.1 and NEG: libvmaf has GPU code (CUDA) for their features,
    // ported to Vulkan (vmaf_vulkan), which every GPU uses; CUDA only where
    // Vulkan cannot score VMAF. VMAF v1: all four of its features on the GPU,
    // with Vulkan, where the engine supports the model's options
    // (vmaf_v1_gpu). SSIMULACRA2 and Butteraugli have no choice of the CPU:
    // on the GPU where one can calculate them, and on the CPU without one;
    // only of the Vship build (below).
    vmaf_backend_combo = new QComboBox;
    vmaf_backend_combo->addItems({"GPU", "CPU"});
#ifdef __APPLE__
    vmaf_backend_combo->setToolTip(app::tr(
        "With GPU, VMAF v0.6.1, VMAF NEG and VMAF v1 are calculated on the GPU, with Vulkan (MoltenVK on Metal). "
        "VMAF v1: for a model whose options the GPU code does not support, its banding and colour features (CAMBI, "
        "SpEED) on the CPU. PSNR, SSIM and XPSNR are calculated on the CPU either way. The GPU's scores agree with "
        "the CPU's to within a thousandth of a point on every frame, so a saved score is kept whichever is chosen. "
        "If the GPU calculation fails, they are calculated on the CPU instead. "
        "HDR VMAF follows this choice too: its features on the GPU, its pictures converted on the CPU."));
#else
    vmaf_backend_combo->setToolTip(app::tr(
        "With GPU, VMAF v0.6.1, VMAF NEG and VMAF v1 are calculated on the GPU, with Vulkan (VMAF v0.6.1 and NEG "
        "with CUDA on an NVIDIA GPU whose Vulkan cannot calculate them). VMAF v1: for a model "
        "whose options the GPU code does not support, its banding and colour features (CAMBI, SpEED) on the CPU. "
        "PSNR, SSIM and XPSNR are calculated on the CPU either way. The GPU's scores agree with the CPU's to within a "
        "thousandth of a point on every frame, so a saved score is kept whichever is chosen. If the GPU calculation "
        "fails, they are calculated on the CPU instead. HDR VMAF follows this choice too: "
        "its features on the GPU, its pictures converted on the GPU where it can (else on the CPU)."));
#endif
    connect(vmaf_backend_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("vmaf_on_gpu"); });
    form->addRow(app::tr("VMAF compute:"), vmaf_backend_combo);
    build_metric_routes(form);

    // Without a GPU there is no choice: it shows CPU, which is where VMAF is
    // calculated (see show_vmaf_backend).
    if (!vmaf_gpu_possible_) {
        vmaf_backend_combo->setEnabled(false);
        vmaf_backend_combo->setToolTip(
            app::tr("No GPU was found: VMAF v0.6.1 and VMAF NEG are calculated on the CPU."));
    }

    // The Vship builds that are bundled.
    vship_build_combo = new QComboBox;
    add_vship_choices(*vship_build_combo, bundled_vship_builds());
    vship_build_combo->setToolTip(app::tr(
        "Which Vship calculates SSIMULACRA2, Butteraugli and CVVDP on the GPU. GPU, the app's own build, is the fastest "
        "and most accurate. Vship Vulkan is stock Vship; Vship HIP (AMD) and Vship CUDA (NVIDIA) are stock Vship too, "
        "for a GPU whose Vulkan driver has problems, and all three share their saved scores. GPU's scores differ "
        "slightly and are saved apart: switching shows those saved, or calculates them on the next run. If GPU fails, "
        "Vship Vulkan calculates them instead. Without a GPU the chosen Vship can use, or if it fails too, SSIMULACRA2 "
        "and Butteraugli are calculated on the CPU, and CVVDP is not available."));
    connect(vship_build_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("vship_build"); });
    form->addRow(app::tr("Vship compute:"), vship_build_combo);
    form->setRowVisible(vship_build_combo, vship_build_bundled(VshipBuild::stock));
    QStringList names;
    for (const GpuVendor vendor : gpu_vendors) {
        names.append(qt_text(vendor_label(vendor)));
    }
#ifdef __APPLE__
    // An Apple silicon Mac's GPU is its chip's (it lists no makers); an Intel
    // Mac's chip name is its CPU's.
    if (const std::string chip = platform::chip_name(); names.isEmpty() && chip.starts_with("Apple")) {
        names.append(qt_text(chip));
    }
#endif
    if (!names.isEmpty()) {
        form->addRow(QString(), new QLabel(app::tr("Detected GPU(s): {names}", {{"names", names.join(", ")}})));
    }

    threads_spin = new QSpinBox;
    threads_spin->setRange(0, 128);
    threads_spin->setValue(0);
    threads_spin->setSpecialValueText(app::tr("Auto"));
    threads_spin->setKeyboardTracking(false);
    connect(threads_spin, &QSpinBox::valueChanged, this, [this] { emit field_edited("n_threads"); });
    form->addRow(app::tr("libvmaf threads:"), threads_spin);
    threads_spin->setToolTip(
        app::tr("Controls VMAF, PSNR and SSIM extraction in libvmaf, not XPSNR or video decoding.\n\nAuto uses every "
                "core, or half of them for each video when two are calculated in parallel."));
}

void OptionsPanel::build_manual_crop(QFormLayout* form) {
    // The crops entered for Manual: the lines cut from each edge of the
    // source and of the test video, as HandBrake's crop fields take them.
    manual_crop_box = new QWidget;
    auto* grid = new QGridLayout(manual_crop_box);
    grid->setContentsMargins(0, 0, 0, 0);
    const QStringList edges{app::tr("Top"), app::tr("Bottom"), app::tr("Left"), app::tr("Right")};
    for (int column = 0; column < edges.size(); ++column) {
        grid->addWidget(new QLabel(edges[column]), 0, column + 1);
    }
    const auto add_row = [&](int row, const QString& label, std::array<QSpinBox*, 4>& spins, bool source) {
        grid->addWidget(new QLabel(label), row, 0);
        for (std::size_t column = 0; column < spins.size(); ++column) {
            QSpinBox* spin = new QSpinBox;
            // Even lines: 4:2:0 keeps its colour in pairs of lines (an odd
            // one typed is taken up to even, as HandBrake takes it).
            spin->setRange(0, 8192);
            spin->setSingleStep(2);
            spin->setKeyboardTracking(false);
            connect(spin, &QSpinBox::valueChanged, this,
                    [this, field = manual_crop_field(source, column)] { emit field_edited(field); });
            grid->addWidget(spin, row, static_cast<int>(column) + 1);
            spins[column] = spin;
        }
    };
    add_row(1, app::tr("Reference"), source_crop_spins, true);
    add_row(2, app::tr("Test video"), test_crop_spins, false);
    manual_crop_box->setToolTip(app::tr("The lines cut from each edge, as the encoder's crop settings cut them (an "
                                        "encode made with a crop of 22 lines at the top and bottom: Top 22, Bottom "
                                        "22 for the reference, nothing for the test video, already cropped). Even "
                                        "numbers."));
    form->addRow(QString(), manual_crop_box);
    // Hidden, not show_manual_crop(): shown before the form has a parent, the
    // box would be a window.
    manual_crop_box->hide();
}

void OptionsPanel::show_manual_crop() {
    manual_crop_box->setVisible(crop_combo->currentData().toInt() == static_cast<int>(CropMode::manual));
}

void OptionsPanel::show_range_unit() {
    // Time's fields or frames', and the note on CVVDP's whole seconds: for
    // frames, and for a time that is not whole seconds.
    const bool frames = range_unit_combo->currentIndex() == 1;
    range_start_edit->setVisible(!frames);
    range_end_edit->setVisible(!frames);
    range_first_spin->setVisible(frames);
    range_last_spin->setVisible(frames);
    const bool part_of_a_second = range_start_edit->time().msec() != 0 || range_end_edit->time().msec() != 0;
    range_frames_note->setVisible(frames || part_of_a_second);
}

void OptionsPanel::commit_typed_values() {
    for (QAbstractSpinBox* box : findChildren<QAbstractSpinBox*>()) {
        box->interpretText();
    }
}

void OptionsPanel::show_options(const VmafOptions& options, const std::vector<VmafOptions>& selected) {
    model_combo->setCurrentIndex(index_of(model_choices(), options.model_choice));
    custom_model_path = options.custom_model_path;
    model_v1_combo->setCurrentIndex(index_of(v1_model_choices(), options.model_choice_v1));
    model_hdr_combo->setCurrentIndex(index_of(hdr_model_choices(), options.model_choice_hdr));

    gpu_checkbox->setChecked(options.gpu_decode);
    const auto vendor = std::find(gpu_vendor_entries.begin(), gpu_vendor_entries.end(), options.gpu_vendor);
    gpu_vendor_combo->setCurrentIndex(
        vendor == gpu_vendor_entries.end() ? 0 : static_cast<int>(vendor - gpu_vendor_entries.begin()));
    gpu_vendor_combo->setEnabled(options.gpu_decode);

    show_vmaf_backend(options.vmaf_on_gpu);
    show_metric_routes(options);
    show_vship_choice(*vship_build_combo, options.vship_build);

    crop_combo->setCurrentIndex(std::max(0, crop_combo->findData(static_cast<int>(options.crop_mode))));
    const auto show_edges = [](const std::array<QSpinBox*, 4>& spins, const CropEdges& edges) {
        const std::array<int, 4> lines{edges.top, edges.bottom, edges.left, edges.right};
        for (std::size_t i = 0; i < spins.size(); ++i) {
            if (!spins[i]->hasFocus()) {
                spins[i]->setValue(lines[i]);
            }
        }
    };
    show_edges(source_crop_spins, options.manual_source_crop);
    show_edges(test_crop_spins, options.manual_test_crop);
    show_manual_crop();
    const auto can_show = [this](const QAbstractSpinBox* box) { return !keep_typing || !box->hasFocus(); };
    if (can_show(threads_spin)) {
        threads_spin->setValue(options.n_threads);
    }
    if (can_show(subsample_spin)) {
        subsample_spin->setValue(options.n_subsample);
    }
    scale_algo_combo->setCurrentIndex(
        std::max(0, static_cast<int>(scale_algorithms.indexOf(QString::fromStdString(options.scale_algorithm)))));
    scale_direction_combo->setCurrentIndex(options.scale_direction == ScaleDirection::distorted_to_source ? 1 : 0);

    range_unit_combo->setCurrentIndex(options.range_in_frames ? 1 : 0);
    const long long time_start = options.range_in_frames ? 0 : options.range_time_ms(options.range_start);
    const long long time_end = options.range_in_frames ? 0 : options.range_time_ms(options.range_end);
    if (can_show(range_start_edit)) {
        range_start_edit->setTime(QTime(0, 0).addMSecs(static_cast<int>(time_start)));
    }
    if (can_show(range_end_edit)) {
        range_end_edit->setTime(QTime(0, 0).addMSecs(static_cast<int>(time_end)));
    }
    if (can_show(range_first_spin)) {
        range_first_spin->setValue(options.range_in_frames ? static_cast<int>(options.range_start) : 0);
    }
    if (can_show(range_last_spin)) {
        range_last_spin->setValue(options.range_in_frames ? static_cast<int>(options.range_end) - 1 : -1);
    }
    show_range_unit();  // the note, for times shown that are not whole seconds

    const auto any = [&selected](auto&& predicate) { return std::any_of(selected.begin(), selected.end(), predicate); };
    model_combo->setEnabled(any([](const VmafOptions& item) { return item.compute_vmaf; }));
    model_v1_combo->setEnabled(any([](const VmafOptions& item) { return item.compute_vmaf_v1; }));
    model_hdr_combo->setEnabled(any([](const VmafOptions& item) { return item.compute_vmaf_hdr; }));
    const bool uses_libvmaf = any([](const VmafOptions& item) {
        return item.compute_vmaf || item.compute_vmaf_neg || item.compute_vmaf_v1 || item.compute_vmaf_hdr ||
               !item.extra_features.empty();
    });
    threads_spin->setEnabled(uses_libvmaf);
    // Also while it is above 1: CVVDP is unavailable then, and the way back
    // to it must not be a disabled control.
    subsample_spin->setEnabled(uses_libvmaf || any([](const VmafOptions& item) { return item.n_subsample > 1; }));
}

void OptionsPanel::show_vmaf_backend(bool on_gpu) {
    vmaf_backend_combo->setCurrentIndex(on_gpu && vmaf_gpu_possible_ ? 0 : 1);
}

void OptionsPanel::build_metric_routes(QFormLayout* form) {
    // CAMBI, CIEDE2000 and MS-SSIM have a GPU route of their own: the bundled Vulkan engine's, from the pictures
    // the CPU metrics are given, on the GPU VMAF runs on (cpu_plan).
    const std::array<QString, metric_gpu_routes.size()> labels{app::tr("CAMBI compute:"),
                                                               app::tr("CIEDE2000 compute:"),
                                                               app::tr("MS-SSIM compute:")};
    const std::array<QString, metric_gpu_routes.size()> agreement{
        app::tr("Its scores on the GPU are the CPU's exactly."),
        app::tr("Its scores on the GPU agree with the CPU's to within a few millionths."),
        app::tr("Its scores on the GPU agree with the CPU's to within a few millionths. MS-SSIM YUV follows "
                "this choice too.")};
    performance_form_ = form;
    for (std::size_t i = 0; i < metric_gpu_routes.size(); ++i) {
        auto* combo = new QComboBox;
        combo->addItems({"GPU", "CPU"});
        if (vmaf_gpu_possible_) {
            combo->setToolTip(
                app::tr("With GPU, it is calculated on the GPU by the app's Vulkan engine (on a Mac through "
                        "MoltenVK), on the GPU VMAF runs on, whether VMAF itself is calculated on the GPU or the CPU; "
                        "on the CPU where the GPU cannot calculate it or the GPU calculation fails. With CPU, "
                        "libvmaf calculates it on the CPU.") +
                " " + agreement[i] + " " + app::tr("A saved score is kept whichever is chosen."));
        } else {  // as VMAF's: CPU, the only place
            combo->setEnabled(false);
            combo->setToolTip(app::tr("No GPU was found: it is calculated on the CPU."));
        }
        const std::string_view field = metric_gpu_routes[i].field;
        connect(combo, &QComboBox::currentIndexChanged, this, [this, field] {
            emit field_edited(QString::fromLatin1(field.data(), static_cast<qsizetype>(field.size())));
        });
        form->addRow(labels[i], combo);
        metric_route_combos[i] = combo;
    }
}

void OptionsPanel::show_metric_routes(const VmafOptions& options) {
    for (std::size_t i = 0; i < metric_gpu_routes.size(); ++i) {
        metric_route_combos[i]->setCurrentIndex(options.*metric_gpu_routes[i].on_gpu && vmaf_gpu_possible_ ? 0 : 1);
    }
}

void OptionsPanel::show_metric_route_rows(const std::function<bool(std::string_view)>& listed) {
    for (std::size_t i = 0; i < metric_gpu_routes.size(); ++i) {
        const std::string_view key = metric_gpu_routes[i].key;
        performance_form_->setRowVisible(metric_route_combos[i], listed(key) ||
                                                                    (key == "ms_ssim" && listed("ms_ssim_yuv")));
    }
}

void OptionsPanel::on_cvvdp_delete_clicked() {
    const QString name = cvvdp_preset_combo->currentData().toString();
    if (name.isEmpty()) {
        return;
    }
    if (cvvdp_delete_guard_->armed() != nullptr && name != cvvdp_delete_armed_) {
        cvvdp_delete_guard_->disarm();
    }
    cvvdp_delete_armed_ = name;
    if (cvvdp_delete_guard_->click(cvvdp_delete_button,
                                   app::tr("Click again to delete \"{name}\".", {{"name", name}}))) {
        emit cvvdp_delete_requested();
    }
}

void OptionsPanel::show_cvvdp(const std::vector<CvvdpSettings>& selected, const std::string& preferred,
                              const std::string& user_presets_json, bool automatic) {
    const CvvdpSettings& settings = selected.front();
    std::set<std::string> displays;
    std::set<bool> resizes;
    for (const CvvdpSettings& item : selected) {
        displays.insert(item.display.identity().dump());
        resizes.insert(item.resize_to_display);
    }
    const bool mixed_display = displays.size() > 1;
    const bool mixed_resize = resizes.size() > 1;
    std::optional<CvvdpPreset> match;
    {
        const QSignalBlocker blocked(cvvdp_preset_combo);
        cvvdp_preset_combo->clear();
        // An empty name, which no preset has.
        cvvdp_preset_combo->addItem(automatic_cvvdp_label(), QString(""));
        cvvdp_preset_combo->setItemData(
            0, app::tr("The SDR or the HDR display, as the reference video is: the official tool's 'standard_4k' or "
                       "'standard_hdr_pq'."),
            Qt::ToolTipRole);
        for (const CvvdpPreset& preset : cvvdp_presets(user_presets_json)) {
            cvvdp_preset_combo->addItem(preset_label(preset), QString::fromStdString(preset.name));
            cvvdp_preset_combo->setItemData(cvvdp_preset_combo->count() - 1,
                                            preset.description.empty() ? app::tr("Your saved preset.")
                                                                       : app::tr(preset.description.c_str()),
                                            Qt::ToolTipRole);
        }
        if (!mixed_display) {
            match = matching_cvvdp_preset(settings, user_presets_json, preferred);
        }
        if (automatic && !mixed_display) {
            cvvdp_preset_combo->setCurrentIndex(0);
        } else if (mixed_display) {
            cvvdp_preset_combo->addItem(app::tr("Mixed (the selected videos use different displays)"), QVariant());
            cvvdp_preset_combo->setCurrentIndex(cvvdp_preset_combo->count() - 1);
        } else if (!match) {
            cvvdp_preset_combo->addItem(app::tr("Custom (not saved as a preset)"), QVariant());
            cvvdp_preset_combo->setCurrentIndex(cvvdp_preset_combo->count() - 1);
        } else {
            cvvdp_preset_combo->setCurrentIndex(cvvdp_preset_combo->findData(QString::fromStdString(match->name)));
        }
    }
    cvvdp_delete_button->setEnabled(!automatic && match && !match->builtin);
    if (!cvvdp_delete_button->isEnabled() || cvvdp_preset_combo->currentData().toString() != cvvdp_delete_armed_) {
        cvvdp_delete_guard_->disarm();  // the armed preset is no longer the one shown
    }
    QString description = mixed_display ? app::tr("The selected videos use different displays; choosing a preset "
                                                  "sets them all.")
                                        : display_description(settings.display);
    if (automatic && !mixed_display && match) {
        description = preset_label(*match) + ": " + description;
    }
    cvvdp_display_label->setText(description);
    const QSignalBlocker blocked(cvvdp_resize_check);
    cvvdp_resize_check->setTristate(mixed_resize);
    cvvdp_resize_check->setCheckState(mixed_resize                 ? Qt::PartiallyChecked
                                      : settings.resize_to_display ? Qt::Checked
                                                                   : Qt::Unchecked);
}

VmafOptions OptionsPanel::read_options() const {
    VmafOptions options;
    options.model.clear();  // worked out for each video as a run starts (resolve_model), once its size is known
    options.model_choice = model_choices()[static_cast<std::size_t>(model_combo->currentIndex())].choice;
    options.model_choice_v1 = v1_model_choices()[static_cast<std::size_t>(model_v1_combo->currentIndex())].choice;
    options.model_choice_hdr = hdr_model_choices()[static_cast<std::size_t>(model_hdr_combo->currentIndex())].choice;
    options.custom_model_path = custom_model_path;
    options.n_threads = threads_spin->value();
    options.n_subsample = subsample_spin->value();
    options.scale_algorithm = scale_algo_combo->currentText().toStdString();
    options.scale_direction = scale_direction_combo->currentIndex() == 1 ? ScaleDirection::distorted_to_source
                                                                         : ScaleDirection::source_to_distorted;
    if (range_unit_combo->currentIndex() == 1) {
        options.range_in_frames = true;
        options.range_in_milliseconds = false;
        options.range_start = range_first_spin->value();
        options.range_end = range_last_spin->value() + 1;  // the frame past the last; End (-1): 0, to the end
    } else {
        options.set_time_range(QTime(0, 0).msecsTo(range_start_edit->time()),
                               QTime(0, 0).msecsTo(range_end_edit->time()));
    }
    options.gpu_decode = gpu_checkbox->isChecked();
    options.gpu_vendor = gpu_checkbox->isChecked()
                             ? gpu_vendor_entries[static_cast<std::size_t>(gpu_vendor_combo->currentIndex())]
                             : GpuVendor::none;
    options.vmaf_on_gpu = vmaf_backend_combo->currentIndex() == 0;
    for (std::size_t i = 0; i < metric_gpu_routes.size(); ++i) {
        options.*metric_gpu_routes[i].on_gpu = metric_route_combos[i]->currentIndex() == 0;
    }
    options.vship_build = shown_vship_choice(*vship_build_combo);
    options.crop_mode = static_cast<CropMode>(crop_combo->currentData().toInt());
    const auto edges = [](const std::array<QSpinBox*, 4>& spins) {
        const auto even_up = [](const QSpinBox* spin) { return spin->value() + (spin->value() & 1); };
        return CropEdges{even_up(spins[0]), even_up(spins[1]), even_up(spins[2]), even_up(spins[3])};
    };
    options.manual_source_crop = edges(source_crop_spins);
    options.manual_test_crop = edges(test_crop_spins);
    return options;
}

}  // namespace vml::app
