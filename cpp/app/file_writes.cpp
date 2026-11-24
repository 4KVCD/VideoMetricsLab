#include "file_writes.h"

#include <exception>

namespace vml::app {

FileWrites::FileWrites(QObject* parent) : QObject(parent) {}

FileWrites::~FileWrites() {
    if (thread_.joinable()) {
        thread_.join();  // it ends once the queue is empty
    }
}

void FileWrites::submit(QString description, std::function<void()> write) {
    submit(std::move(description), std::move(write), {});
}

void FileWrites::submit(QString description, std::function<void()> write,
                        std::function<void(const QString& error)> done) {
    std::lock_guard lock(mutex_);
    writes_.push_back({std::move(description), std::move(write), std::move(done)});
    ++pending_;
    if (writing_) {
        return;  // the thread takes it after those before it
    }
    writing_ = true;
    if (thread_.joinable()) {
        thread_.join();  // the last thread, which has run out of writes
    }
    thread_ = std::thread([this] { write_all(); });
}

bool FileWrites::wait_until_idle(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return idle_.wait_for(lock, timeout, [this] { return pending_ == 0; });
}

void FileWrites::write_all() {
    while (true) {
        Write next;
        {
            std::lock_guard lock(mutex_);
            if (writes_.empty()) {
                writing_ = false;
                return;
            }
            next = std::move(writes_.front());
            writes_.pop_front();
        }
        auto& [description, write, done] = next;
        QString failure;
        try {
            write();
        } catch (const std::exception& error) {
            failure = QString::fromStdString(error.what());
            if (!done) {
                emit write_failed(description, failure);
            }
        }
        if (done) {
            QMetaObject::invokeMethod(this, [done = std::move(done), failure] { done(failure); }, Qt::QueuedConnection);
        }
        bool now_idle = false;
        {
            std::lock_guard lock(mutex_);
            now_idle = --pending_ == 0;
            if (now_idle) {
                idle_.notify_all();
            }
        }
        if (now_idle) {
            emit became_idle();
        }
    }
}

bool FileWrites::idle() {
    std::lock_guard lock(mutex_);
    return pending_ == 0;
}

}  // namespace vml::app
