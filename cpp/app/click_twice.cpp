#include "click_twice.h"

#include <QPushButton>
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

}  // namespace vml::app
