// The form that adds a resolution test: a listed size, or a width or a height
// typed, the other following from the reference's shape, and the scaling
// algorithm of both steps.
#pragma once

#include <QDialog>

#include <string>
#include <vector>

#include "vml/geometry.h"
#include "vml/options.h"

class QComboBox;
class QSpinBox;

namespace vml::app {

class ResampleTargetDialog : public QDialog {
public:
    // `reference` is the reference's size, which has widths to test
    // (resample_smallest_width), and `scaler` the algorithm shown first.
    ResampleTargetDialog(Size reference, const std::string& scaler, QWidget* parent);

    // The test chosen, with what is typed in a box not left yet.
    ResampleTarget target() const;

private:
    void show_size(Size size);
    void list_chosen(int index);
    void height_entered(int height);

    Size reference_;
    std::vector<ResampleTarget> listed_;  // the list's entries; "Custom" follows them
    Size shown_;
    QComboBox* sizes_;
    QSpinBox* width_;
    QSpinBox* height_;
    QComboBox* scalers_;
};

}  // namespace vml::app
