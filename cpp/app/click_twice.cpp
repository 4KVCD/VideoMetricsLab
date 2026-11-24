#include "click_twice.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QPushButton>
#include <QTimer>

#include <memory>

#include "theme.h"

namespace vml::app {

ClickTwice::ClickTwice(QObject* parent, Clock clock, int hold_ms) : QObject(parent), clock_(std::move(clock)) {
    if (!clock_) {
        auto started = std::make_shared<QElapsedTimer>();
        started->start();
        clock_ = [started] { return started->elapsed(); };
    }
    timer_ = new QTimer(this);
    timer_->setSingleShot(true);
    timer_->setInterval(hold_ms);
    connect(timer_, &QTimer::timeout, this, &ClickTwice::disarm);
}

bool ClickTwice::click(QPushButton* button, const QString& line) {
    if (armed_ == button) {
        if (clock_() - armed_at_ < QApplication::doubleClickInterval()) {
            return false;  // a double-click's second click
        }
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
    armed_at_ = clock_();
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
