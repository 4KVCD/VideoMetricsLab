#include "screen_frame_clock.h"

#include <QCoreApplication>
#include <QEvent>
#include <QPointer>
#include <QWindow>

#ifdef Q_OS_WIN
#include <dwmapi.h>
#endif

namespace vml::app {

ScreenFrameClock& ScreenFrameClock::instance() {
    static QPointer<ScreenFrameClock> clock;
    if (clock == nullptr) {
        clock = new ScreenFrameClock(QCoreApplication::instance());
    }
    return *clock;
}

ScreenFrameClock::ScreenFrameClock(QObject* parent) : QObject(parent) {}

#ifdef Q_OS_WIN

// QWindow::requestUpdate is no use here: for a window of widgets on Windows it
// is a 5 ms timer. DwmFlush returns at the desktop compositor's next pass, and
// blocks, so it gets a thread of its own.

ScreenFrameClock::~ScreenFrameClock() {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_one();
    if (waiter_.joinable()) {
        waiter_.join();
    }
}

void ScreenFrameClock::request_frame(QWindow* /*window*/) {
    if (!waiter_.joinable()) {
        waiter_ = std::thread([this] { wait_for_frames(); });
    }
    {
        const std::lock_guard lock(mutex_);
        frame_wanted_ = true;
    }
    wake_.notify_one();
}

void ScreenFrameClock::wait_for_frames() {
    std::unique_lock lock(mutex_);
    while (true) {
        wake_.wait(lock, [this] { return frame_wanted_ || stopping_; });
        if (stopping_) {
            return;
        }
        frame_wanted_ = false;
        lock.unlock();
        DwmFlush();
        QMetaObject::invokeMethod(this, [this] { emit frame_started(); }, Qt::QueuedConnection);
        lock.lock();
    }
}

#else

// On macOS, Qt's update requests come from the screen's display link.
ScreenFrameClock::~ScreenFrameClock() = default;

void ScreenFrameClock::request_frame(QWindow* window) {
    if (window != nullptr) {
        window->installEventFilter(this);  // installed once, however often asked
        window->requestUpdate();
    }
}

bool ScreenFrameClock::eventFilter(QObject* /*watched*/, QEvent* event) {
    if (event->type() == QEvent::UpdateRequest) {
        emit frame_started();
    }
    return false;
}

#endif

}  // namespace vml::app
