#include "resample_target_dialog.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QString>

#include <algorithm>
#include <string_view>

#include "i18n.h"

namespace vml::app {

namespace {

// A box of even numbers, which says a value only when it is entered: a height
// half typed must not change the width.
QSpinBox* size_box(int low, int high, const char* name) {
    auto* box = new QSpinBox;
    box->setRange(low, high);
    box->setSingleStep(2);
    box->setKeyboardTracking(false);
    box->setObjectName(name);
    return box;
}

}  // namespace

ResampleTargetDialog::ResampleTargetDialog(Size reference, const std::string& scaler, QWidget* parent)
    : QDialog(parent), reference_(reference) {
    setWindowTitle(app::tr("Add resolution test"));
    const int smallest = resample_smallest_width(reference);
    const int largest = resample_largest_width(reference);
    sizes_ = new QComboBox;
    for (const ResampleTarget& target : resample_target_choices()) {
        if (target.width >= smallest && target.width <= largest) {
            listed_.push_back(target);
            sizes_->addItem(QString::fromStdString(target.label));
        }
    }
    sizes_->addItem(app::tr("Custom"));
    width_ = size_box(smallest, largest, "resample_width");
    height_ = size_box(resample_down_height(smallest, reference), resample_down_height(largest, reference),
                       "resample_height");
    scalers_ = new QComboBox;
    for (const std::string_view name : resample_scalers) {
        scalers_->addItem(QString::fromUtf8(name.data(), static_cast<qsizetype>(name.size())));
    }
    scalers_->setCurrentText(QString::fromStdString(scaler));

    auto* form = new QFormLayout(this);
    form->addRow(app::tr("Downscale to (then scale back up):"), sizes_);
    form->addRow(app::tr("Width:"), width_);
    form->addRow(app::tr("Height:"), height_);
    auto* black_bars_note = new QLabel(app::tr("With black bars cut, the height follows the picture without them."));
    black_bars_note->setWordWrap(true);
    form->addRow(black_bars_note);
    form->addRow(app::tr("Scaling algorithm:"), scalers_);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    form->addRow(buttons);

    // The second listed size, as the list has always opened, or half the width
    // where the reference is too small for any.
    show_size(resample_size_for_width(
        listed_.empty() ? reference.first / 2 : listed_[std::min<std::size_t>(1, listed_.size() - 1)].width,
        reference));
    connect(sizes_, &QComboBox::currentIndexChanged, this, &ResampleTargetDialog::list_chosen);
    connect(width_, &QSpinBox::valueChanged, this,
            [this](int width) { show_size(resample_size_for_width(width, reference_)); });
    connect(height_, &QSpinBox::valueChanged, this, &ResampleTargetDialog::height_entered);
}

ResampleTarget ResampleTargetDialog::target() const {
    width_->interpretText();  // a click on OK does not leave the box on a Mac
    height_->interpretText();
    ResampleTarget target = resample_target_of_width(width_->value());
    target.scaler = scalers_->currentText().toStdString();
    return target;
}

void ResampleTargetDialog::show_size(Size size) {
    shown_ = size;
    const QSignalBlocker width_quiet(width_);
    const QSignalBlocker height_quiet(height_);
    const QSignalBlocker list_quiet(sizes_);
    width_->setValue(size.first);
    height_->setValue(size.second);
    const auto listed = std::find_if(listed_.begin(), listed_.end(),
                                     [&size](const ResampleTarget& target) { return target.width == size.first; });
    sizes_->setCurrentIndex(static_cast<int>(listed - listed_.begin()));  // "Custom" follows the listed sizes
}

void ResampleTargetDialog::list_chosen(int index) {
    if (static_cast<std::size_t>(index) < listed_.size()) {
        show_size(resample_size_for_width(listed_[static_cast<std::size_t>(index)].width, reference_));
    }
}

void ResampleTargetDialog::height_entered(int height) {
    Size size = resample_size_for_height(height, reference_);
    // On a tall picture one step of the width moves the height by more than the
    // box's own step: an arrow that found the same size would never get past it.
    if (size.second == shown_.second) {
        size = resample_size_for_width(size.first + (height > shown_.second ? 2 : -2), reference_);
    }
    show_size(size);
}

}  // namespace vml::app
