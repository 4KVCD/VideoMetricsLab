// The CVVDP display editor (the Python app's CvvdpDisplayDialog): edits the
// display CVVDP models, and saves it as a preset. Every value changes the
// score.
#pragma once

#include <QDialog>
#include <QString>

#include <map>
#include <optional>
#include <set>
#include <string>

#include "vml/cvvdp.h"

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;

namespace vml::app {

class CvvdpDisplayDialog : public QDialog {
    Q_OBJECT

public:
    // `own_preset` is the name of the user's preset being edited, if the
    // display is one: its name can then be changed ("Save preset" renames and
    // updates it). `new_preset` is the "Add preset..." form, which can only
    // save a new preset. `taken_names` are every preset name in use, which a
    // new name may not repeat.
    CvvdpDisplayDialog(const CvvdpDisplay& display, QWidget* parent, std::optional<std::string> own_preset,
                       bool new_preset, std::set<std::string> taken_names);

    // The edited display. A value the user did not change keeps its exact
    // original value: the boxes show fewer decimals than a display can have,
    // and reading 0.7472 m back as 0.747 made "Apply without saving" with
    // nothing changed drop the CVVDP score and turn the preset into "Custom".
    CvvdpDisplay display() const;
    // The preset name typed, without the spaces around it.
    std::string preset_name() const;
    // Closes the dialog with the action `wanted` ("save", "save_new" or
    // "apply"), or says why the name or the display cannot be used.
    void finish(const QString& wanted);

    // After exec(), which button closed it: "save" (update or rename the
    // user's preset), "save_new", or "apply" (use without saving).
    QString action;

    // The fields, as the tests fill them in.
    QLineEdit* name_edit = nullptr;
    QSpinBox* width_spin = nullptr;
    QSpinBox* height_spin = nullptr;
    QDoubleSpinBox* diagonal_spin = nullptr;
    QDoubleSpinBox* distance_spin = nullptr;
    QDoubleSpinBox* peak_spin = nullptr;
    QDoubleSpinBox* contrast_spin = nullptr;
    QDoubleSpinBox* ambient_spin = nullptr;
    QDoubleSpinBox* reflectivity_spin = nullptr;
    QDoubleSpinBox* exposure_spin = nullptr;
    QCheckBox* hdr_check = nullptr;
    QLabel* distance_note = nullptr;

private:
    void update_distance_note();
    std::optional<QString> name_problem(const QString& wanted) const;

    CvvdpDisplay original_;
    std::optional<std::string> own_preset_;
    std::set<std::string> taken_names_;
    // What each number box held before the user changed anything.
    std::map<const QDoubleSpinBox*, double> given_;
};

}  // namespace vml::app
