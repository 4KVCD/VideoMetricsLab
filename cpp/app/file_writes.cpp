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
    std::lock_guard lock(mutex_);
    writes_.emplace_back(std::move(description), std::move(write));
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
        std::pair<QString, std::function<void()>> next;
        {
            std::lock_guard lock(mutex_);
            if (writes_.empty()) {
                writing_ = false;
                return;
            }
            next = std::move(writes_.front());
            writes_.pop_front();
        }
        auto& [description, write] = next;
        try {
            write();
        } catch (const std::exception& error) {
            emit write_failed(description, QString::fromStdString(error.what()));
        }
        std::lock_guard lock(mutex_);
        if (--pending_ == 0) {
            idle_.notify_all();
        }
    }
}

}  // namespace vml::app
