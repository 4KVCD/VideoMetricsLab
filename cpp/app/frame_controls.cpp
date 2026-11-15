#include "frame_controls.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>

#include <algorithm>
#include <array>

#include "i18n.h"

namespace vml::app {

FrameControls::FrameControls(QWidget* parent) : QWidget(parent) {
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(new QLabel(app::tr("Go to frame:")));
    frame_spin = new QSpinBox;
    frame_spin->setRange(0, 0);
    frame_spin->setKeyboardTracking(false);  // on Enter, not at each digit typed
    frame_spin->setToolTip(app::tr("Reports every visible series at this exact frame, so two encodes can be compared "
                                   "at one moment."));
    connect(frame_spin, &QSpinBox::valueChanged, this, [this](int frame) { emit frame_requested(frame); });
    layout->addWidget(frame_spin);
    auto* go_button = new QPushButton(app::tr("Go"));
    connect(go_button, &QPushButton::clicked, this, [this] { emit frame_requested(frame_spin->value()); });
    layout->addWidget(go_button);
    layout->addSpacing(16);
    // The range in boxes of its own: "—", each box's lowest value, as the
    // tables say no value, is no end.
    const QString range_tip = app::tr("A frame range: the graph shades the frames from the first box's frame to the "
                                      "second's, and the table gives their statistics. Shift+drag across the graph "
                                      "to select one.");
    const auto new_range_box = [this, &range_tip] {
        auto* box = new QSpinBox;
        box->setRange(-1, 0);
        box->setValue(-1);
        box->setSpecialValueText(QStringLiteral("—"));
        box->setKeyboardTracking(false);
        box->setToolTip(range_tip);
        connect(box, &QSpinBox::valueChanged, this, [this] { on_range_box_changed(); });
        return box;
    };
    layout->addWidget(new QLabel(app::tr("Highlight range from frame:")));
    range_start_spin = new_range_box();
    layout->addWidget(range_start_spin);
    layout->addWidget(new QLabel(app::tr("to frame:")));
    range_end_spin = new_range_box();
    layout->addWidget(range_end_spin);
    clear_range_button = new QPushButton(app::tr("Clear range"));
    clear_range_button->setEnabled(false);
    clear_range_button->setToolTip(app::tr("Shows the statistics of the whole video again."));
    connect(clear_range_button, &QPushButton::clicked, this, [this] { clear(); });
    layout->addWidget(clear_range_button);
}

std::optional<std::pair<int, int>> FrameControls::selected_frames() const {
    if (!range_) {
        return std::nullopt;
    }
    return std::pair(std::min(range_->first, range_->second), std::max(range_->first, range_->second));
}

void FrameControls::set_bounds(int lo, int hi) {
    lo_ = lo;
    hi_ = std::max(lo, hi);
    const bool blocked = frame_spin->blockSignals(true);
    frame_spin->setRange(lo_, hi_);
    frame_spin->blockSignals(blocked);
    refresh_range_boxes();
}

void FrameControls::set_dragged_range(int first, int last, bool finished) {
    range_ = std::pair(first, last);
    refresh_range_boxes();
    emit range_changed(finished);
}

void FrameControls::clear() {
    forget_range();
    emit range_changed(true);
}

void FrameControls::forget_range() {
    range_.reset();
    reset_range_boxes();
    clear_range_button->setEnabled(false);
}

void FrameControls::on_range_box_changed() {
    // Read from both boxes each time, so that it rests on nothing kept: a
    // box at its lowest value ("—") has no end.
    const auto end_in = [](const QSpinBox* box) {
        return box->value() == box->minimum() ? std::nullopt : std::optional<int>(box->value());
    };
    const std::optional<int> first = end_in(range_start_spin);
    const std::optional<int> last = end_in(range_end_spin);
    if (first && last) {
        range_ = std::pair(*first, *last);
    } else {
        range_.reset();  // one end typed so far: the whole video until the other is
    }
    refresh_range_boxes();
    emit range_changed(true);
}

void FrameControls::reset_range_boxes() {
    for (QSpinBox* box : {range_start_spin, range_end_spin}) {
        const bool blocked = box->blockSignals(true);
        box->setValue(box->minimum());
        box->blockSignals(blocked);
    }
}

void FrameControls::refresh_range_boxes() {
    // The range boxes bounded by the graph's frames, widened to their own
    // ends, so that an end is shown as chosen even where the lines drawn stop
    // short of it. Each box's ends are read before setRange, which clamps the
    // value, and each box's "—" is its own lowest value, which moves with the
    // graph's first frame.
    std::array<QSpinBox*, 2> boxes{range_start_spin, range_end_spin};
    std::array<std::optional<int>, 2> ends;
    if (range_) {
        ends = {range_->first, range_->second};
    } else {
        for (std::size_t i = 0; i < boxes.size(); ++i) {
            if (boxes[i]->value() != boxes[i]->minimum()) {
                ends[i] = boxes[i]->value();  // one end typed, the other not yet
            }
        }
    }
    int lo = lo_;
    int hi = hi_;
    for (const std::optional<int>& end : ends) {
        if (end) {
            lo = std::min(lo, *end);
            hi = std::max(hi, *end);
        }
    }
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const bool blocked = boxes[i]->blockSignals(true);
        boxes[i]->setRange(lo - 1, hi);
        boxes[i]->setValue(ends[i].value_or(lo - 1));
        boxes[i]->blockSignals(blocked);
    }
    clear_range_button->setEnabled(ends[0] || ends[1]);
}

}  // namespace vml::app
