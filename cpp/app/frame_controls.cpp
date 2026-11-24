#include "frame_controls.h"

#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSpinBox>

#include <algorithm>
#include <array>
#include <cmath>

#include "i18n.h"

namespace vml::app {

FrameControls::FrameControls(QWidget* parent) : QWidget(parent) {
    // Frames in the first row, times under them (set_timeline), each box
    // under its kind in the other row.
    auto* grid = new QGridLayout(this);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setVerticalSpacing(2);
    auto* go_label = new QLabel(app::tr("Go to frame:"));
    grid->addWidget(go_label, 0, 0, Qt::AlignRight);
    frame_spin = new QSpinBox;
    frame_spin->setRange(0, 0);
    frame_spin->setKeyboardTracking(false);  // on Enter, not at each digit typed
    frame_spin->setToolTip(app::tr("Reports every visible series at this exact frame, so two encodes can be compared "
                                   "at one moment."));
    connect(frame_spin, &QSpinBox::valueChanged, this, [this](int frame) {
        show_frame_time(frame);
        emit frame_requested(frame);
    });
    grid->addWidget(frame_spin, 0, 1);
    auto* go_button = new QPushButton(app::tr("Go"));
    connect(go_button, &QPushButton::clicked, this, [this] {
        show_frame_time(frame_spin->value());
        emit frame_requested(frame_spin->value());
    });
    grid->addWidget(go_button, 0, 2);
    grid->setColumnMinimumWidth(3, 16);
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
        frame_range_tip_ = range_tip;
        return box;
    };
    grid->addWidget(new QLabel(app::tr("Highlight range from frame:")), 0, 4, Qt::AlignRight);
    range_start_spin = new_range_box();
    grid->addWidget(range_start_spin, 0, 5);
    grid->addWidget(new QLabel(app::tr("to frame:")), 0, 6, Qt::AlignRight);
    range_end_spin = new_range_box();
    grid->addWidget(range_end_spin, 0, 7);
    clear_range_button = new QPushButton(app::tr("Clear range"));
    clear_range_button->setEnabled(false);
    clear_range_button->setToolTip(app::tr("Shows the statistics of the whole video again."));
    connect(clear_range_button, &QPushButton::clicked, this, [this] { clear(); });
    grid->addWidget(clear_range_button, 0, 8);

    // The same in times, to the millisecond: typed, then Enter (or Go).
    const QString time_hint = QStringLiteral("h:mm:ss.sss");
    const auto new_time_box = [&time_hint] {
        auto* box = new QLineEdit;
        box->setPlaceholderText(time_hint);
        box->setLayoutDirection(Qt::LeftToRight);  // times read left to right in every language
        box->setMaximumWidth(box->fontMetrics().horizontalAdvance(QStringLiteral("00:00:00.0000")) + 16);
        return box;
    };
    auto* time_label = new QLabel(app::tr("Go to time:"));
    grid->addWidget(time_label, 1, 0, Qt::AlignRight);
    time_edit = new_time_box();
    time_edit->setToolTip(app::tr("Reports every visible series at the frame shown at this time (h:mm:ss.sss, or "
                                  "seconds), so two encodes can be compared at one moment."));
    connect(time_edit, &QLineEdit::returnPressed, this, [this] { on_time_requested(); });
    grid->addWidget(time_edit, 1, 1);
    time_go_button = new QPushButton(app::tr("Go"));
    connect(time_go_button, &QPushButton::clicked, this, [this] { on_time_requested(); });
    grid->addWidget(time_go_button, 1, 2);
    const QString range_time_tip = app::tr("A range in time (h:mm:ss.sss, or seconds): the graph shades the frames "
                                           "from the first time to the second, both included, and the table gives "
                                           "their statistics. The frame boxes show its frames.");
    auto* from_label = new QLabel(app::tr("Highlight range from time:"));
    grid->addWidget(from_label, 1, 4, Qt::AlignRight);
    range_start_time = new_time_box();
    grid->addWidget(range_start_time, 1, 5);
    auto* to_label = new QLabel(app::tr("to time:"));
    grid->addWidget(to_label, 1, 6, Qt::AlignRight);
    range_end_time = new_time_box();
    grid->addWidget(range_end_time, 1, 7);
    for (QLineEdit* box : {range_start_time, range_end_time}) {
        box->setToolTip(range_time_tip);
        connect(box, &QLineEdit::editingFinished, this, [this, box] {
            if (box->isModified()) {
                box->setModified(false);
                on_range_time_edited();
            }
        });
    }
    time_row_ = {time_label, time_edit, time_go_button, from_label, range_start_time, to_label, range_end_time};
    for (QWidget* widget : time_row_) {
        widget->hide();
    }
    grid->setColumnStretch(9, 1);
}

void FrameControls::set_timeline(Timeline timeline) {
    timeline_ = std::move(timeline);
    for (QWidget* widget : time_row_) {
        widget->show();
    }
    refresh_range_times();
}

std::optional<double> FrameControls::parse_time(const QString& text) {
    // [[h:]m:]s[.fff]: minutes and seconds after a colon under 60, at most
    // three decimals (a millisecond).
    static const QRegularExpression form(
        QStringLiteral(R"(^\s*(?:(?:(\d{1,4}):)?(\d{1,2}):)?(\d{1,6})(?:[.,](\d{1,3}))?\s*$)"));
    const QRegularExpressionMatch parts = form.match(text);
    if (!parts.hasMatch()) {
        return std::nullopt;
    }
    const bool hours = !parts.captured(1).isEmpty();
    const bool minutes = !parts.captured(2).isEmpty();
    const long long h = hours ? parts.captured(1).toLongLong() : 0;
    const long long m = minutes ? parts.captured(2).toLongLong() : 0;
    const long long s = parts.captured(3).toLongLong();
    if ((minutes && s >= 60) || (hours && m >= 60)) {
        return std::nullopt;
    }
    const QString fraction = parts.captured(4);
    const long long ms = fraction.isEmpty() ? 0 : (fraction + QStringLiteral("00")).left(3).toLongLong();
    return static_cast<double>(((h * 60 + m) * 60 + s) * 1000 + ms) / 1000.0;
}

QString FrameControls::time_text(double seconds) {
    const long long ms = std::llround(std::max(0.0, seconds) * 1000.0);
    return QStringLiteral("%1:%2:%3.%4")
        .arg(ms / 3600000)
        .arg(ms / 60000 % 60, 2, 10, QLatin1Char('0'))
        .arg(ms / 1000 % 60, 2, 10, QLatin1Char('0'))
        .arg(ms % 1000, 3, 10, QLatin1Char('0'));
}

void FrameControls::on_time_requested() {
    const std::optional<double> seconds = parse_time(time_edit->text());
    if (!timeline_ || !seconds) {
        return;
    }
    std::optional<int> frame = timeline_->frame_at(*seconds, false);
    if (!frame) {
        frame = timeline_->frame_at(*seconds, true);  // before the first frame: the first
    }
    if (!frame) {
        return;
    }
    const bool blocked = frame_spin->blockSignals(true);
    frame_spin->setValue(*frame);
    frame_spin->blockSignals(blocked);
    emit frame_requested(*frame);
}

void FrameControls::on_range_time_edited() {
    const std::optional<double> from = parse_time(range_start_time->text());
    const std::optional<double> to = parse_time(range_end_time->text());
    if (in_seconds_) {
        // The times as typed: each line's own frames between them.
        seconds_ = from && to ? std::optional(std::pair(std::min(*from, *to), std::max(*from, *to))) : std::nullopt;
        clear_range_button->setEnabled(!range_start_time->text().isEmpty() || !range_end_time->text().isEmpty());
        emit range_changed(true);
        return;
    }
    if (!timeline_) {
        return;
    }
    std::optional<int> first;
    std::optional<int> last;
    if (from && to) {
        first = timeline_->frame_at(std::min(*from, *to), true);
        last = timeline_->frame_at(std::max(*from, *to), false);
    }
    if (first && last && *first <= *last) {
        range_ = std::pair(*first, *last);
    } else {
        range_.reset();  // one time typed so far, or none of the frames between them
        reset_range_boxes();
    }
    refresh_range_boxes();  // the frame boxes show its frames; the times stay as typed
    emit range_changed(true);
}

void FrameControls::refresh_range_times() {
    if (in_seconds_) {
        if (seconds_) {
            for (const auto& [box, at] : {std::pair(range_start_time, seconds_->first),
                                          std::pair(range_end_time, seconds_->second)}) {
                box->setText(time_text(at));
                box->setModified(false);
            }
        }
        return;
    }
    if (!timeline_) {
        return;
    }
    const auto show = [this](QLineEdit* box, std::optional<int> frame) {
        const std::optional<double> seconds = frame ? timeline_->time_of(*frame) : std::nullopt;
        box->setText(seconds ? time_text(*seconds) : QString());
        box->setModified(false);
    };
    // No range: the times left as they are, one perhaps typed so far
    // (forget_range empties them).
    if (const std::optional<std::pair<int, int>> frames = selected_frames()) {
        show(range_start_time, frames->first);
        show(range_end_time, frames->second);
    }
}

void FrameControls::show_frame_time(int frame) {
    if (timeline_) {
        const std::optional<double> seconds = timeline_->time_of(frame);
        time_edit->setText(seconds ? time_text(*seconds) : QString());
    }
}

void FrameControls::set_range_in_seconds(bool on, const QString& why) {
    if (on == in_seconds_) {
        if (on) {
            range_start_spin->setToolTip(why);
            range_end_spin->setToolTip(why);
        }
        return;
    }
    forget_range();  // frames and seconds don't carry over
    seconds_.reset();
    in_seconds_ = on;
    for (QSpinBox* box : {range_start_spin, range_end_spin}) {
        box->setEnabled(!on);
        box->setToolTip(on ? why : frame_range_tip_);
    }
}

std::optional<std::pair<double, double>> FrameControls::selected_seconds() const {
    return in_seconds_ ? seconds_ : std::nullopt;
}

void FrameControls::set_dragged_seconds(double first, double last, bool finished) {
    seconds_ = std::pair(std::min(first, last), std::max(first, last));
    refresh_range_times();
    clear_range_button->setEnabled(true);
    emit range_changed(finished);
}

std::optional<std::pair<int, int>> FrameControls::selected_frames() const {
    if (!range_ || in_seconds_) {
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
    refresh_range_times();  // the page shown may place the frames elsewhere in time
}

void FrameControls::set_dragged_range(int first, int last, bool finished) {
    range_ = std::pair(first, last);
    refresh_range_boxes();
    refresh_range_times();
    emit range_changed(finished);
}

void FrameControls::clear() {
    forget_range();
    emit range_changed(true);
}

void FrameControls::forget_range() {
    range_.reset();
    seconds_.reset();
    reset_range_boxes();
    clear_range_button->setEnabled(false);
    for (QLineEdit* box : {range_start_time, range_end_time}) {
        box->clear();
        box->setModified(false);
    }
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
    refresh_range_boxes(BoxBounds::keep_wider);  // a held arrow keeps stepping (issue #28)
    refresh_range_times();
    emit range_changed(true);
}

void FrameControls::reset_range_boxes() {
    for (QSpinBox* box : {range_start_spin, range_end_spin}) {
        const bool blocked = box->blockSignals(true);
        box->setValue(box->minimum());
        box->blockSignals(blocked);
    }
}

void FrameControls::refresh_range_boxes(BoxBounds bounds) {
    // The range boxes bounded by the graph's frames, widened to their own
    // ends, so that an end is shown as chosen even where the lines drawn stop
    // short of it. Each box's ends are read before setRange, which clamps the
    // value, and each box's "—" is its own lowest value, which moves with the
    // graph's first frame.
    //
    // setRange also stops the repeat of an arrow held down on the box, so it
    // runs only where the bounds change; after a box's own step
    // (BoxBounds::keep_wider) not even where they would only narrow, as they
    // do while an end beyond the graph's frames is stepped back towards
    // them. Both boxes always have the same bounds.
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
    const bool same = boxes[0]->minimum() == lo - 1 && boxes[0]->maximum() == hi;
    const bool wider = boxes[0]->minimum() <= lo - 1 && boxes[0]->maximum() >= hi;
    const bool set_range = !same && !(bounds == BoxBounds::keep_wider && wider);
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const bool blocked = boxes[i]->blockSignals(true);
        if (set_range) {
            boxes[i]->setRange(lo - 1, hi);
        }
        boxes[i]->setValue(ends[i].value_or(boxes[i]->minimum()));
        boxes[i]->blockSignals(blocked);
    }
    clear_range_button->setEnabled(ends[0] || ends[1] || (range_start_time != nullptr &&
                                                          (!range_start_time->text().isEmpty() ||
                                                           !range_end_time->text().isEmpty())));
}

}  // namespace vml::app
