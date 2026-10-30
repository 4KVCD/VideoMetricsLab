// The Options panel of the Videos tab (the Python app's
// vmaf_app/ui/options_panel.py): the settings of the videos selected in the
// table.
//
// The panel shows a video's options (show_options, show_cvvdp) and reads
// them back (read_options). It decides nothing about the rows itself: each
// control the user edits is reported to the window with a signal, and the
// window applies the change to the selected rows. Which metrics a video
// calculates is not here: it is ticked in the table's own columns.
#pragma once

#include <QGroupBox>
#include <QString>

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vml/cvvdp.h"
#include "vml/options.h"
#include "vml/video.h"

class QCheckBox;
class QComboBox;
class QFormLayout;
class QLabel;
class QPushButton;
class QSpinBox;
class QTimeEdit;

namespace vml::app {

// A VMAF model in a dropdown: its name (shown translated) and the choice a
// video saves (VmafOptions' model_choice or model_choice_v1).
struct ModelChoice {
    const char* name;
    std::string choice;
};

// The VMAF v0.6.1 column's models, in the dropdown's order.
const std::vector<ModelChoice>& model_choices();
// The VMAF v1 column's models, in the dropdown's order.
const std::vector<ModelChoice>& v1_model_choices();

// A selected row's video as the Film grain dropdown sees it.
struct FilmGrainVideo {
    const VideoInfo* info = nullptr;  // nothing: not read yet
    bool resolution_test = false;     // its video is the reference, chosen with it at Calculate metrics
};

// What the Film grain dropdown shows for the selected videos: whether it can
// be used (a video with film grain it can be scored with or without), the
// choice they share (nothing: they differ, or none has one to make), and why.
struct FilmGrainControl {
    bool enabled = false;
    std::optional<bool> with;
    QString tooltip;
};
FilmGrainControl film_grain_control(const std::vector<FilmGrainVideo>& videos);

class OptionsPanel : public QGroupBox {
    Q_OBJECT

public:
    // `gpu_vendors`: the GPUs found on this PC. Without one, VMAF can only
    // be calculated on the CPU.
    explicit OptionsPanel(const std::vector<GpuVendor>& gpu_vendors, QWidget* parent = nullptr);

    // Shows `options` (the first selected video's) in the controls, and
    // `selected` every selected video's options, which decide what can be
    // edited. The window ignores its own handlers while this runs.
    void show_options(const VmafOptions& options, const std::vector<VmafOptions>& selected);
    // The VMAF GPU/CPU choice as a run would take it: CPU when there is no
    // GPU, whatever the video is set to.
    void show_vmaf_backend(bool on_gpu);
    // Shows the selected videos' CVVDP settings: the preset they match (or
    // "Custom"), or "Automatic" when their displays all follow the reference
    // (`automatic`), the display in one line, and the resize box. `preferred`
    // is the preset the first one was given, to tell apart presets with the
    // same display; `user_presets_json` the user's own (Settings').
    //
    // Videos that differ show as "Mixed" in the dropdown and a partly ticked
    // box. Don't show only the first video's values: choosing the preset it
    // already has, or clicking the box, would then do nothing for the other
    // videos, or set them the wrong way round.
    void show_cvvdp(const std::vector<CvvdpSettings>& selected, const std::string& preferred,
                    const std::string& user_presets_json, bool automatic);
    // Shows the selected videos' film grain choice (film_grain_control).
    void show_film_grain(const std::vector<FilmGrainVideo>& videos);

    // The options the controls show. The metrics are not in the panel, so
    // the window gives them, from the table's column headings.
    VmafOptions read_options(const std::vector<std::string>& extra_features, bool compute_vmaf, bool compute_vmaf_neg,
                             bool compute_vmaf_v1, bool compute_xpsnr) const;

    // The model file chosen for "Custom model file...", until it is applied
    // to the rows.
    std::string custom_model_path;

    // The controls. Public, as the Python app's are: the window reads them,
    // and the tests use them as a user would.
    QLabel* target_label = nullptr;
    QComboBox* crop_combo = nullptr;
    QTimeEdit* duration_edit = nullptr;
    QComboBox* scale_algo_combo = nullptr;
    QComboBox* scale_direction_combo = nullptr;
    QComboBox* film_grain_combo = nullptr;
    QComboBox* model_combo = nullptr;
    QComboBox* model_v1_combo = nullptr;
    QSpinBox* subsample_spin = nullptr;
    QComboBox* cvvdp_preset_combo = nullptr;
    QLabel* cvvdp_display_label = nullptr;
    QPushButton* cvvdp_add_button = nullptr;
    QPushButton* cvvdp_edit_button = nullptr;
    QPushButton* cvvdp_delete_button = nullptr;
    QCheckBox* cvvdp_resize_check = nullptr;
    QCheckBox* gpu_checkbox = nullptr;
    QComboBox* gpu_vendor_combo = nullptr;
    QComboBox* vmaf_backend_combo = nullptr;
    QComboBox* vship_build_combo = nullptr;
    QSpinBox* threads_spin = nullptr;

signals:
    // A control was edited: the name of the VmafOptions field it sets ("gpu"
    // for the decode box and the vendor, "model_v1", "crop_mode"...).
    void field_edited(const QString& field);
    // The VMAF v0.6.1 model dropdown changed to this index (model_choices).
    void model_chosen(int index);
    // The CVVDP preset dropdown changed to this index.
    void cvvdp_preset_chosen(int index);
    // "Scale the video to fill the display" was ticked (true) or unticked.
    void cvvdp_resize_toggled(bool checked);
    void cvvdp_add_requested();
    void cvvdp_edit_requested();
    void cvvdp_delete_requested();
    // The resolution mismatch dropdown changed to this index (2: "Test both").
    void scale_direction_chosen(int index);
    // The Film grain dropdown chose with the grain (true) or without.
    void film_grain_chosen(bool with);

private:
    void build_preparation(QFormLayout* form);
    void build_metric_settings(QFormLayout* form);
    void build_cvvdp(QFormLayout* form);
    void build_performance(QFormLayout* form, const std::vector<GpuVendor>& gpu_vendors);

    // Whether there is a GPU here for VMAF: without one, its choice is CPU.
    const bool vmaf_gpu_possible_;
};

}  // namespace vml::app
