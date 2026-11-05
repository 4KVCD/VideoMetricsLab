// Says when the screen starts a new frame, for what should move once a frame
// and in step with it, whatever the screen's refresh rate. A timer cannot: its
// whole milliseconds drift against the screen's frames.
#pragma once

#include <QObject>

#ifdef Q_OS_WIN
#include <condition_variable>
#include <mutex>
#include <thread>
#endif

class QWindow;

namespace vml::app {

class ScreenFrameClock : public QObject {
    Q_OBJECT

public:
    // The program's one clock. For the window's thread only.
    static ScreenFrameClock& instance();
    ~ScreenFrameClock() override;

    // Asks for frame_started at the next frame of the screen `window` is on.
    // Nothing comes without a request, so that an idle window does no work.
    // `window` may be null, where it has no native window yet: nothing comes.
    void request_frame(QWindow* window);

signals:
    void frame_started();

private:
    explicit ScreenFrameClock(QObject* parent);

#ifdef Q_OS_WIN
    void wait_for_frames();

    std::thread waiter_;  // started at the first request
    std::mutex mutex_;
    std::condition_variable wake_;
    bool frame_wanted_ = false;
    bool stopping_ = false;
#else
    bool eventFilter(QObject* watched, QEvent* event) override;
#endif
};

}  // namespace vml::app
