// Runs file writes (saved scores) off the window's thread (the Python app's
// vmaf_app/ui/file_worker.py): a feature-length video's scores take seconds
// to write, and written in the handler that shows a finished video they
// would freeze the window just as its result arrives. They run on one
// thread of their own, so they stay in the order they were asked for.
#pragma once

#include <QObject>
#include <QString>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace vml::app {

class FileWrites : public QObject {
    Q_OBJECT

public:
    explicit FileWrites(QObject* parent = nullptr);
    // Waits for the writes left: one cut short would lose a result.
    ~FileWrites() override;

    // Queues a write, to run after those already queued. It runs on another
    // thread, so it must not touch a widget: give it plain data.
    void submit(QString description, std::function<void()> write);

    // Waits until every write queued has run, for at most `timeout`; whether
    // they all have. For closing the window, which must not cut one short.
    bool wait_until_idle(std::chrono::milliseconds timeout);

signals:
    // A write failed: what it was writing, and why.
    void write_failed(const QString& description, const QString& error);

private:
    void write_all();

    std::mutex mutex_;
    std::condition_variable idle_;
    std::deque<std::pair<QString, std::function<void()>>> writes_;
    int pending_ = 0;       // queued, not yet run
    bool writing_ = false;  // a thread is taking writes off the queue
    std::thread thread_;
};

}  // namespace vml::app
