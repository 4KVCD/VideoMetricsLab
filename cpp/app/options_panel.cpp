#include "options_panel.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
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
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>

#include "formatting.h"
#include "i18n.h"
#include "theme.h"
#include "window_helpers.h"
#include "vml/model_select.h"

// The panel's text is app::tr's: inside a Q_OBJECT class, tr() is Qt's.
namespace vml::app {

namespace {

// The scaling algorithms, as FFmpeg's scale filter names them.
const QStringList scale_algorithms{"bicubic", "lanczos", "bilinear", "spline"};

// The GPU decode dropdown's entries, in its order.
const std::vector<GpuVendor> gpu_vendor_entries{GpuVendor::automatic, GpuVendor::nvidia, GpuVendor::intel,
                                                GpuVendor::amd};

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

const std::vector<ModelChoice>& v1_model_choices() {
    static const std::vector<ModelChoice> choices{
        {N_("Auto (analysis resolution)"), std::string(auto_model_choice)},
        {N_("VMAF v1 (1080p / 3H)"), builtin_choice("vmaf_v1_3d0h")},
        {N_("VMAF v1 (1080p phone / 5H)"), builtin_choice("vmaf_v1_5d0h")},
        {N_("VMAF v1 (4K / 1.5H)"), builtin_choice("vmaf_v1_1d5h_2160")},
        {N_("VMAF v1 (4K / 3H, up to 110)"), builtin_choice("vmaf_v1_3d0h_2160")},
        {N_("VMAF v1 HFR (1080p / 3H)"), builtin_choice("vmaf_v1_hfr_3d0h")},
        {N_("VMAF v1 HFR (1080p phone / 5H)"), builtin_choice("vmaf_v1_hfr_5d0h")},
        {N_("VMAF v1 HFR (4K / 1.5H)"), builtin_choice("vmaf_v1_hfr_1d5h_2160")},
        {N_("VMAF v1 HFR (4K / 3H, up to 110)"), builtin_choice("vmaf_v1_hfr_3d0h_2160")},
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

}  // namespace

OptionsPanel::OptionsPanel(const std::vector<GpuVendor>& gpu_vendors, QWidget* parent)
    : QGroupBox(app::tr("Options"), parent), vmaf_gpu_possible_(vmaf_gpu_possible(gpu_vendors)) {
    // Which videos it edits, then three boxes side by side (video
    // preparation, metric-specific settings, performance), then a note on
    // what affects scores.
    auto* layout = new QVBoxLayout(this);
    target_label = new QLabel;
    theme::style(target_label, "color: {muted}; font-style: italic;");
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
    build_metric_settings(add_box(app::tr("Metric-specific settings")));
    build_performance(add_box(app::tr("Performance")), gpu_vendors);
    layout->addLayout(columns);

    auto* metrics_hint = new QLabel(
        app::tr("Cropping and scaling here affect scores. Video Compare's tone mapping and playback resolution are "
                "display-only settings and do not change calculated metrics."));
    theme::style(metrics_hint, "color: {muted}; font-style: italic;");
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
    // Black bars, duration, scaling algorithm and resolution mismatch.
    crop_combo = new QComboBox;
    crop_combo->addItems({app::tr("Auto-detect (recommended)"), app::tr("None (use full frame)")});
    connect(crop_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("crop_mode"); });
    form->addRow(app::tr("Black-bar handling:"), crop_combo);

    duration_edit = new QTimeEdit;
    // In a right-to-left window (Arabic) QTimeEdit writes the sections
    // backwards ("000.00:00:00"); times read left to right in every
    // language. The direction must be set before the display format.
    duration_edit->setLayoutDirection(Qt::LeftToRight);
    duration_edit->setDisplayFormat("HH:mm:ss.zzz");
    duration_edit->setTime(QTime(0, 0, 0, 0));
    connect(duration_edit, &QTimeEdit::timeChanged, this, [this] { emit field_edited("duration_limit"); });
    form->addRow(app::tr("Duration limit (0 = full):"), duration_edit);

    scale_algo_combo = new QComboBox;
    scale_algo_combo->addItems(scale_algorithms);
    connect(scale_algo_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("scale_algorithm"); });
    form->addRow(app::tr("Scaling algorithm:"), scale_algo_combo);

    scale_direction_combo = new QComboBox;
    scale_direction_combo->addItems({app::tr("Source downscaled to test"), app::tr("Test upscaled to source"),
                                     app::tr("Test both (adds a comparison row)")});
    scale_direction_combo->setToolTip(
        app::tr("When the two resolutions differ: either evaluate quality at the resolution actually\n"
                "delivered (source scaled to match test -- the default), or as if the test\n"
                "video were upscaled back to the reference's native resolution for playback.\n"
                "\"Test both\" doesn't change this row -- it adds a second row for the same test\n"
                "file using the other direction, so you can run and compare both."));
    connect(scale_direction_combo, &QComboBox::currentIndexChanged, this, &OptionsPanel::scale_direction_chosen);
    form->addRow(app::tr("Resolution mismatch:"), scale_direction_combo);
}

void OptionsPanel::build_metric_settings(QFormLayout* form) {
    // The VMAF models, the frame subsample, and CVVDP's display.
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

    subsample_spin = new QSpinBox;
    subsample_spin->setRange(1, 60);
    subsample_spin->setValue(1);
    connect(subsample_spin, &QSpinBox::valueChanged, this, [this] { emit field_edited("n_subsample"); });
    form->addRow(app::tr("libvmaf frame subsample:"), subsample_spin);
    subsample_spin->setToolTip(
        app::tr("1 = every frame. Applies to VMAF, PSNR and SSIM. XPSNR is computed every frame; combined runs retain "
                "values at libvmaf's sampled frames."));

    build_cvvdp(form);
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
    cvvdp_add_button = new QPushButton(app::tr("Add preset..."));
    cvvdp_add_button->setToolTip(
        app::tr("Make a new preset of your own, starting from the display shown here. It becomes the display newly "
                "added videos start with; the selected videos keep theirs until you choose it for them."));
    connect(cvvdp_add_button, &QPushButton::clicked, this, &OptionsPanel::cvvdp_add_requested);
    cvvdp_edit_button = new QPushButton(app::tr("Edit display..."));
    cvvdp_edit_button->setToolTip(
        app::tr("Change the display's size, distance, brightness, room light and more; rename your preset or save "
                "the display as a new one."));
    connect(cvvdp_edit_button, &QPushButton::clicked, this, &OptionsPanel::cvvdp_edit_requested);
    cvvdp_delete_button = new QPushButton(app::tr("Delete preset"));
    cvvdp_delete_button->setToolTip(app::tr("Delete this preset of yours. Built-in presets cannot be deleted."));
    connect(cvvdp_delete_button, &QPushButton::clicked, this, &OptionsPanel::cvvdp_delete_requested);
    for (QPushButton* button : {cvvdp_add_button, cvvdp_edit_button, cvvdp_delete_button}) {
        buttons->addWidget(button);
    }
    buttons->addStretch(1);
    form->addRow(QString(), buttons);

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
    gpu_checkbox->setToolTip(
        app::tr("Hardware-decodes the reference and the test video. Each is decided separately, and either one falls "
                "back to the CPU on its own if this GPU can't decode its format."));
    gpu_checkbox->setChecked(true);
    gpu_vendor_combo = new QComboBox;
    gpu_vendor_combo->addItems({app::tr("Auto-detect"), "NVIDIA", "Intel", "AMD"});
    connect(gpu_checkbox, &QCheckBox::toggled, gpu_vendor_combo, &QWidget::setEnabled);
    connect(gpu_checkbox, &QCheckBox::toggled, this, [this] { emit field_edited("gpu"); });
    connect(gpu_vendor_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("gpu"); });
    auto* gpu_row = new QHBoxLayout;
    gpu_row->addWidget(gpu_checkbox);
    gpu_row->addWidget(gpu_vendor_combo);
    form->addRow(app::tr("GPU decode:"), gpu_row);

    // VMAF v0.6.1 and NEG: libvmaf has GPU code (CUDA) for their features,
    // ported to Vulkan (vmaf_vulkan), which every GPU uses unless the GPU
    // backend is CUDA. VMAF v1: all four of its features on the GPU, with
    // Vulkan, where the engine supports the model's options (vmaf_v1_gpu).
    vmaf_backend_combo = new QComboBox;
    vmaf_backend_combo->addItems({"GPU", "CPU"});
    vmaf_backend_combo->setToolTip(app::tr(
        "With GPU, VMAF v0.6.1, VMAF NEG and VMAF v1 are calculated on the GPU. VMAF v0.6.1 and NEG: with Vulkan on "
        "every GPU, or with CUDA on an NVIDIA GPU when Settings > GPU metrics > GPU backend is CUDA. Both give the "
        "same scores. VMAF v1: all four of its features on the GPU with Vulkan; for a model whose options the GPU "
        "code does not support, its banding and colour features (CAMBI, SpEED) on the CPU. PSNR, SSIM and XPSNR are "
        "calculated on the CPU either way. The GPU's scores agree with the CPU's to within a thousandth of a point "
        "on every frame, so a saved score is kept whichever is chosen. If the GPU calculation fails, they are "
        "calculated on the CPU instead."));
    connect(vmaf_backend_combo, &QComboBox::currentIndexChanged, this, [this] { emit field_edited("vmaf_on_gpu"); });
    form->addRow(app::tr("VMAF compute:"), vmaf_backend_combo);

    // SSIMULACRA2's and Butteraugli's GPU/CPU choice, the same for both.
    const QString vship_tip =
        app::tr("GPU uses Vship when a supported device is available. If GPU computation cannot run, the metric "
                "automatically falls back to CPU. Choose CPU to always use the bundled libjxl reference "
                "implementation.");
    ssimulacra2_backend_combo = new QComboBox;
    ssimulacra2_backend_combo->addItems({"GPU", "CPU"});
    ssimulacra2_backend_combo->setToolTip(vship_tip);
    connect(ssimulacra2_backend_combo, &QComboBox::currentIndexChanged, this,
            [this] { emit metric_backend_changed("ssimulacra2"); });
    form->addRow(app::tr("SSIMULACRA2 compute:"), ssimulacra2_backend_combo);
    butteraugli_backend_combo = new QComboBox;
    butteraugli_backend_combo->addItems({"GPU", "CPU"});
    butteraugli_backend_combo->setToolTip(vship_tip);
    connect(butteraugli_backend_combo, &QComboBox::currentIndexChanged, this,
            [this] { emit metric_backend_changed("butteraugli"); });
    form->addRow(app::tr("Butteraugli compute:"), butteraugli_backend_combo);

    // Without a GPU there is no choice: it shows CPU, which is where VMAF is
    // calculated (see show_vmaf_backend).
    if (!vmaf_gpu_possible_) {
        vmaf_backend_combo->setEnabled(false);
        vmaf_backend_combo->setToolTip(
            app::tr("No GPU was found: VMAF v0.6.1 and VMAF NEG are calculated on the CPU."));
    }
    if (!gpu_vendors.empty()) {
        QStringList names;
        for (const GpuVendor vendor : gpu_vendors) {
            names.append(qt_text(vendor_label(vendor)));
        }
        form->addRow(QString(), new QLabel(app::tr("Detected GPU(s): {names}", {{"names", names.join(", ")}})));
    }

    threads_spin = new QSpinBox;
    threads_spin->setRange(0, 128);
    threads_spin->setValue(0);
    threads_spin->setSpecialValueText(app::tr("Auto"));
    connect(threads_spin, &QSpinBox::valueChanged, this, [this] { emit field_edited("n_threads"); });
    form->addRow(app::tr("libvmaf threads:"), threads_spin);
    threads_spin->setToolTip(
        app::tr("Controls VMAF, PSNR and SSIM extraction in libvmaf, not XPSNR or video decoding.\n\nAuto uses every "
                "core, or half of them for each video when two are calculated in parallel."));
}

void OptionsPanel::show_options(const VmafOptions& options, const std::map<std::string, std::string>& metric_backends,
                                const std::vector<VmafOptions>& selected) {
    model_combo->setCurrentIndex(index_of(model_choices(), options.model_choice));
    custom_model_path = options.custom_model_path;
    model_v1_combo->setCurrentIndex(index_of(v1_model_choices(), options.model_choice_v1));

    gpu_checkbox->setChecked(options.gpu_decode);
    const auto vendor = std::find(gpu_vendor_entries.begin(), gpu_vendor_entries.end(), options.gpu_vendor);
    gpu_vendor_combo->setCurrentIndex(
        vendor == gpu_vendor_entries.end() ? 0 : static_cast<int>(vendor - gpu_vendor_entries.begin()));
    gpu_vendor_combo->setEnabled(options.gpu_decode);

    show_backends(options.vmaf_on_gpu, metric_backends);

    crop_combo->setCurrentIndex(options.crop_mode == CropMode::automatic ? 0 : 1);
    threads_spin->setValue(options.n_threads);
    subsample_spin->setValue(options.n_subsample);
    scale_algo_combo->setCurrentIndex(
        std::max(0, static_cast<int>(scale_algorithms.indexOf(QString::fromStdString(options.scale_algorithm)))));
    scale_direction_combo->setCurrentIndex(options.scale_direction == ScaleDirection::distorted_to_source ? 1 : 0);

    // Rounded half to even, as Python's round() rounds.
    const auto milliseconds = static_cast<int>(std::nearbyint(options.duration_limit * 1000));
    duration_edit->setTime(QTime(0, 0, 0, 0).addMSecs(milliseconds));

    const auto any = [&selected](auto&& predicate) { return std::any_of(selected.begin(), selected.end(), predicate); };
    model_combo->setEnabled(any([](const VmafOptions& item) { return item.compute_vmaf; }));
    model_v1_combo->setEnabled(any([](const VmafOptions& item) { return item.compute_vmaf_v1; }));
    const bool uses_libvmaf = any([](const VmafOptions& item) {
        return item.compute_vmaf || item.compute_vmaf_neg || item.compute_vmaf_v1 || !item.extra_features.empty();
    });
    threads_spin->setEnabled(uses_libvmaf);
    // Also while it is above 1: CVVDP is unavailable then, and the way back
    // to it must not be a disabled control.
    subsample_spin->setEnabled(uses_libvmaf || any([](const VmafOptions& item) { return item.n_subsample > 1; }));
}

void OptionsPanel::show_backends(bool vmaf_on_gpu, const std::map<std::string, std::string>& metric_backends) {
    show_vmaf_backend(vmaf_on_gpu);
    const auto on_gpu = [&metric_backends](const std::string& key) {
        const auto found = metric_backends.find(key);
        return found == metric_backends.end() || found->second == "gpu";
    };
    ssimulacra2_backend_combo->setCurrentIndex(on_gpu("ssimulacra2") ? 0 : 1);
    butteraugli_backend_combo->setCurrentIndex(on_gpu("butteraugli") ? 0 : 1);
}

void OptionsPanel::show_vmaf_backend(bool on_gpu) {
    vmaf_backend_combo->setCurrentIndex(on_gpu && vmaf_gpu_possible_ ? 0 : 1);
}

void OptionsPanel::show_cvvdp(const std::vector<CvvdpSettings>& selected, const std::string& preferred,
                              const std::string& user_presets_json) {
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
        if (mixed_display) {
            cvvdp_preset_combo->addItem(app::tr("Mixed (the selected videos use different displays)"), QVariant());
            cvvdp_preset_combo->setCurrentIndex(cvvdp_preset_combo->count() - 1);
        } else if (!match) {
            cvvdp_preset_combo->addItem(app::tr("Custom (not saved as a preset)"), QVariant());
            cvvdp_preset_combo->setCurrentIndex(cvvdp_preset_combo->count() - 1);
        } else {
            cvvdp_preset_combo->setCurrentIndex(cvvdp_preset_combo->findData(QString::fromStdString(match->name)));
        }
    }
    cvvdp_delete_button->setEnabled(match && !match->builtin);
    cvvdp_display_label->setText(
        mixed_display ? app::tr("The selected videos use different displays; choosing a preset sets them all.")
                      : display_description(settings.display));
    const QSignalBlocker blocked(cvvdp_resize_check);
    cvvdp_resize_check->setTristate(mixed_resize);
    cvvdp_resize_check->setCheckState(mixed_resize                 ? Qt::PartiallyChecked
                                      : settings.resize_to_display ? Qt::Checked
                                                                   : Qt::Unchecked);
}

VmafOptions OptionsPanel::read_options(const std::vector<std::string>& extra_features, bool compute_vmaf,
                                       bool compute_vmaf_neg, bool compute_vmaf_v1, bool compute_xpsnr) const {
    VmafOptions options;
    options.model.clear();  // worked out for each video as a run starts (resolve_model), once its size is known
    options.model_choice = model_choices()[static_cast<std::size_t>(model_combo->currentIndex())].choice;
    options.model_choice_v1 = v1_model_choices()[static_cast<std::size_t>(model_v1_combo->currentIndex())].choice;
    options.compute_vmaf_v1 = compute_vmaf_v1;
    options.custom_model_path = custom_model_path;
    options.extra_features = extra_features;
    options.n_threads = threads_spin->value();
    options.n_subsample = subsample_spin->value();
    options.scale_algorithm = scale_algo_combo->currentText().toStdString();
    options.scale_direction = scale_direction_combo->currentIndex() == 1 ? ScaleDirection::distorted_to_source
                                                                         : ScaleDirection::source_to_distorted;
    options.compute_xpsnr = compute_xpsnr;
    options.compute_vmaf = compute_vmaf;
    options.compute_vmaf_neg = compute_vmaf_neg;
    options.duration_limit = QTime(0, 0, 0, 0).msecsTo(duration_edit->time()) / 1000.0;
    options.gpu_decode = gpu_checkbox->isChecked();
    options.gpu_vendor = gpu_checkbox->isChecked()
                             ? gpu_vendor_entries[static_cast<std::size_t>(gpu_vendor_combo->currentIndex())]
                             : GpuVendor::none;
    options.vmaf_on_gpu = vmaf_backend_combo->currentIndex() == 0;
    options.crop_mode = crop_combo->currentIndex() == 0 ? CropMode::automatic : CropMode::none;
    return options;
}

std::string OptionsPanel::metric_backend(const std::string& metric_key) const {
    const QComboBox* combo = metric_key == "ssimulacra2"   ? ssimulacra2_backend_combo
                             : metric_key == "butteraugli" ? butteraugli_backend_combo
                                                           : nullptr;
    if (combo == nullptr) {
        throw std::invalid_argument("Unknown standalone metric: " + metric_key);
    }
    return combo->currentIndex() == 0 ? "gpu" : "cpu";
}

}  // namespace vml::app
