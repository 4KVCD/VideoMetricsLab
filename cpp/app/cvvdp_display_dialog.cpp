#include "cvvdp_display_dialog.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>

#include <stdexcept>
#include <utility>

#include "i18n.h"
#include "theme.h"
#include "vml/python_text.h"

// The dialog's text is app::tr's: inside a Q_OBJECT class, tr() is Qt's.
namespace vml::app {

namespace {

// The largest display resolution offered: 8K. With "Scale the video to fill
// the display", Vship works at the display's resolution. On a 4K video, an
// 8K display took CVVDP alone (CUDA build) to 7.3 GB more VRAM, and a
// 16384x16384 one to 15.9 GB more. No such display exists, and that is more
// memory than most GPUs have.
constexpr int max_display_pixels = 8192;

// A number box with these limits, steps and tooltip.
QDoubleSpinBox* spin(double low, double high, double value, int decimals, const QString& suffix, double step,
                     const QString& tip) {
    auto* box = new QDoubleSpinBox;
    box->setRange(low, high);
    box->setDecimals(decimals);
    box->setSingleStep(step);
    box->setSuffix(suffix);
    box->setValue(value);
    box->setToolTip(tip);
    return box;
}

}  // namespace

CvvdpDisplayDialog::CvvdpDisplayDialog(const CvvdpDisplay& display, QWidget* parent,
                                       std::optional<std::string> own_preset, bool new_preset,
                                       std::set<std::string> taken_names)
    : QDialog(parent), original_(display), own_preset_(std::move(own_preset)), taken_names_(std::move(taken_names)) {
    setWindowTitle(new_preset ? app::tr("New CVVDP preset") : app::tr("CVVDP display"));
    auto* form = new QFormLayout(this);
    auto* intro = new QLabel(
        app::tr("CVVDP predicts how visible the differences are to someone watching this display, from this "
                "distance, in this light. Each value below changes the score, so compare videos scored for the "
                "same display."));
    intro->setWordWrap(true);
    form->addRow(intro);

    const bool own = own_preset_.has_value() && !own_preset_->empty();
    name_edit = new QLineEdit(QString::fromStdString(own_preset_.value_or("")));
    name_edit->setPlaceholderText(new_preset || !own_preset_ ? app::tr("Name for the new preset") : QString());
    name_edit->setToolTip(own ? app::tr("Change it and click Save preset to rename this preset.")
                              : app::tr("Type a name and click Save as new preset to keep this display as a preset."));
    form->addRow(app::tr("Preset name:"), name_edit);

    width_spin = new QSpinBox;
    height_spin = new QSpinBox;
    for (const auto& [box, value] : {std::pair{width_spin, display.width}, std::pair{height_spin, display.height}}) {
        box->setRange(16, max_display_pixels);
        box->setValue(static_cast<int>(value));
        box->setToolTip(app::tr("The display's own resolution in pixels (not the video's)."));
    }
    auto* resolution = new QHBoxLayout;
    resolution->addWidget(width_spin);
    resolution->addWidget(new QLabel("x"));
    resolution->addWidget(height_spin);
    resolution->addStretch(1);
    form->addRow(app::tr("Resolution:"), resolution);
    diagonal_spin = spin(1, 1000, display.diagonal_inches, 1, " in", 1, app::tr("The screen's diagonal size."));
    form->addRow(app::tr("Screen size:"), diagonal_spin);
    distance_spin = spin(0.05, 50, display.viewing_distance_m, 4, " m", 0.05,
                         app::tr("How far the viewer's eyes are from the screen. Closer makes small artifacts "
                                 "easier to see."));
    distance_note = new QLabel;
    theme::style(distance_note, "color: {muted};");
    auto* distance = new QHBoxLayout;
    distance->addWidget(distance_spin);
    distance->addWidget(distance_note);
    distance->addStretch(1);
    form->addRow(app::tr("Viewing distance:"), distance);
    peak_spin = spin(1, 10000, display.peak_luminance, 0, app::tr(" cd/m\xC2\xB2"), 50,
                     app::tr("The display's peak brightness (nits): about 200 for an office monitor, 600-1500 for an "
                             "HDR monitor, 1000-4000 for an HDR TV."));
    form->addRow(app::tr("Peak brightness:"), peak_spin);
    contrast_spin = spin(1, 10'000'000, display.contrast, 0, " : 1", 100,
                         app::tr("Peak to black: about 1000:1 for a typical LCD, 1,000,000:1 for OLED or the official "
                                 "HDR displays."));
    form->addRow(app::tr("Contrast:"), contrast_spin);
    ambient_spin = spin(0, 100'000, display.ambient_lux, 1, " lux", 10,
                        app::tr("Light falling on the screen: about 250 lux in an office, 50 in a lit living room, "
                                "5-10 watching a film with the lights low, 0 in the dark."));
    form->addRow(app::tr("Room light:"), ambient_spin);
    reflectivity_spin = spin(0, 99.9, display.reflectivity * 100, 2, " %", 0.1,
                             app::tr("How much of the room light the screen reflects back at the viewer; 0.5% is the "
                                     "official models' value."));
    form->addRow(app::tr("Screen reflectivity:"), reflectivity_spin);
    exposure_spin = spin(0.01, 100, display.exposure, 2, QString(), 0.1,
                         app::tr("Brightness multiplier for the pictures; 1 shows them as encoded."));
    form->addRow(app::tr("Exposure:"), exposure_spin);
    hdr_check = new QCheckBox(app::tr("HDR display"));
    hdr_check->setChecked(display.hdr);
    hdr_check->setToolTip(app::tr("An HDR display, as in the official HDR display models. Off: an SDR one."));
    form->addRow(QString(), hdr_check);

    auto* buttons = new QDialogButtonBox;
    if (own_preset_ && !new_preset) {
        QPushButton* save = buttons->addButton(app::tr("Save preset"), QDialogButtonBox::AcceptRole);
        save->setToolTip(app::tr("Save these values (and the name above) as your preset \"{own_preset}\".",
                                 {{"own_preset", QString::fromStdString(*own_preset_)}}));
        connect(save, &QPushButton::clicked, this, [this] { finish("save"); });
    }
    QPushButton* save_new = buttons->addButton(app::tr("Save as new preset"), QDialogButtonBox::AcceptRole);
    save_new->setToolTip(app::tr("Keep this display as a preset of your own under the name above. It becomes the "
                                 "display newly added videos start with."));
    connect(save_new, &QPushButton::clicked, this, [this] { finish("save_new"); });
    if (!new_preset) {
        QPushButton* apply = buttons->addButton(app::tr("Apply without saving"), QDialogButtonBox::AcceptRole);
        apply->setToolTip(app::tr("Use this display for the selected videos without saving a preset."));
        connect(apply, &QPushButton::clicked, this, [this] { finish("apply"); });
    }
    QPushButton* cancel = buttons->addButton(QDialogButtonBox::Cancel);
    connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
    form->addRow(buttons);

    // What each box holds before the user changes anything (see display).
    for (QDoubleSpinBox* box :
         {diagonal_spin, distance_spin, peak_spin, contrast_spin, ambient_spin, reflectivity_spin, exposure_spin}) {
        given_[box] = box->value();
    }
    for (QSpinBox* box : {width_spin, height_spin}) {
        connect(box, &QSpinBox::valueChanged, this, [this] { update_distance_note(); });
    }
    for (QDoubleSpinBox* box : {diagonal_spin, distance_spin}) {
        connect(box, &QDoubleSpinBox::valueChanged, this, [this] { update_distance_note(); });
    }
    update_distance_note();
}

CvvdpDisplay CvvdpDisplayDialog::display() const {
    // A box still holding what it was given was not changed. Not compared
    // with the original rounded instead: that would have to round a half up,
    // as the box does (the Python app rounded it to even, and 350.5 nits
    // came back as 351).
    const auto value = [this](const QDoubleSpinBox* box, double before, double scale = 1.0) {
        return box->value() == given_.at(box) ? before : box->value() / scale;
    };
    CvvdpDisplay display;
    display.width = width_spin->value();
    display.height = height_spin->value();
    display.diagonal_inches = value(diagonal_spin, original_.diagonal_inches);
    display.viewing_distance_m = value(distance_spin, original_.viewing_distance_m);
    display.peak_luminance = value(peak_spin, original_.peak_luminance);
    display.contrast = value(contrast_spin, original_.contrast);
    display.ambient_lux = value(ambient_spin, original_.ambient_lux);
    display.reflectivity = value(reflectivity_spin, original_.reflectivity, 100);
    display.exposure = value(exposure_spin, original_.exposure);
    display.hdr = hdr_check->isChecked();
    return display;
}

void CvvdpDisplayDialog::update_distance_note() {
    // The viewing distance in screen heights.
    distance_note->setText(app::tr("= {distance_in_heights:.2f} x screen height",
                                   {{"distance_in_heights", display().distance_in_heights()}}));
}

std::string CvvdpDisplayDialog::preset_name() const {
    return std::string(python_strip(name_edit->text().toStdString()));
}

std::optional<QString> CvvdpDisplayDialog::name_problem(const QString& wanted) const {
    // Why the typed name cannot be saved with the action `wanted`, if it
    // cannot.
    const std::string name = preset_name();
    if (wanted == "apply") {
        return std::nullopt;
    }
    if (name.empty()) {
        return app::tr("Type a name for the preset.");
    }
    const bool renaming_own = wanted == "save" && own_preset_ && name == *own_preset_;
    if (taken_names_.contains(name) && !renaming_own) {
        return app::tr("There is already a preset called \"{name}\". Choose another name.",
                       {{"name", QString::fromStdString(name)}});
    }
    return std::nullopt;
}

void CvvdpDisplayDialog::finish(const QString& wanted) {
    std::optional<QString> problem = name_problem(wanted);
    if (!problem) {
        try {
            display().validate();
        } catch (const std::invalid_argument& error) {
            problem = tr_message(QString::fromStdString(error.what()));
        }
    }
    if (problem) {
        QMessageBox::warning(this, windowTitle(), *problem);
        return;
    }
    action = wanted;
    accept();
}

}  // namespace vml::app
