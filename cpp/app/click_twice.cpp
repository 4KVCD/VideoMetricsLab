#include "click_twice.h"

#include <QLabel>
#include <QPushButton>
#include <QSizePolicy>
#include <QTimer>

#include "theme.h"

namespace vml::app {

ClickTwice::ClickTwice(QObject* parent, int hold_ms) : QObject(parent) {
    timer_ = new QTimer(this);
    timer_->setSingleShot(true);
    timer_->setInterval(hold_ms);
    connect(timer_, &QTimer::timeout, this, &ClickTwice::disarm);
}

bool ClickTwice::click(QPushButton* button, const QString& line) {
    if (armed_ == button) {
        disarm();
        return true;
    }
    arm(button, line);
    return false;
}

void ClickTwice::arm(QPushButton* button, const QString& line) {
    disarm();
    armed_ = button;
    armed_role_ = button->property("vmlRole").toString();
    line_ = line;
    theme::set_role(button, "danger");
    timer_->start();
    emit changed();
}

void ClickTwice::disarm() {
    if (armed_ == nullptr) {
        return;
    }
    theme::set_role(armed_, armed_role_.toUtf8().constData());
    armed_ = nullptr;
    line_.clear();
    timer_->stop();
    emit changed();
}

void ClickTwice::show_line_in(QLabel* label) {
    label->setTextFormat(Qt::PlainText);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);  // never widens the row
    label->setMinimumHeight(label->fontMetrics().height());               // its line kept, empty or not
    label->setText(line_);
    style_as_armed(label, true);
    connect(this, &ClickTwice::changed, label, [this, label] { label->setText(line_); });
}

void ClickTwice::style_as_armed(QLabel* label, bool armed) {
    label->setStyleSheet(armed ? QStringLiteral("QLabel { color: %1; font-weight: 600; }")
                                     .arg(theme::color("failed").name())
                               : QString());
}

}  // namespace vml::app
