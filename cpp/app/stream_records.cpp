#include "stream_records.h"

#include <exception>

#include "vml/metric_cache.h"

namespace vml::app {

StreamRecords::~StreamRecords() {
    stop();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void StreamRecords::set_reporting(std::function<void(const std::filesystem::path&)> started,
                                  std::function<void()> idle) {
    const std::lock_guard lock(mutex_);
    started_ = std::move(started);
    idle_ = std::move(idle);
}

void StreamRecords::add(std::filesystem::path base, std::filesystem::path test) {
    const std::lock_guard lock(mutex_);
    if (stopped_) {
        return;
    }
    queue_.emplace_back(std::move(base), std::move(test));
    start_locked();
}

void StreamRecords::start_locked() {
    if (recording_ || paused_ || stopped_ || queue_.empty()) {
        return;  // the thread under way takes it, or a resume does
    }
    recording_ = true;
    if (thread_.joinable()) {
        thread_.join();  // one that has emptied the queue and is ending
    }
    thread_ = std::thread([this] { record_all(); });
}

void StreamRecords::pause() {
    const std::lock_guard lock(mutex_);
    paused_ = true;
    current_.cancel();
}

void StreamRecords::resume() {
    const std::lock_guard lock(mutex_);
    paused_ = false;
    start_locked();
}

void StreamRecords::stop() {
    const std::lock_guard lock(mutex_);
    stopped_ = true;
    queue_.clear();
    current_.cancel();
}

void StreamRecords::record_all() {
    while (true) {
        std::pair<std::filesystem::path, std::filesystem::path> next;
        CancelToken cancel;
        std::function<void(const std::filesystem::path&)> started;
        std::function<void()> idle;
        bool done = false;
        {
            const std::lock_guard lock(mutex_);
            if (queue_.empty() || paused_ || stopped_) {
                recording_ = false;
                done = true;
                idle = idle_;
            } else {
                next = queue_.front();
                current_ = CancelToken();
                cancel = current_;
                started = started_;
            }
        }
        if (done) {
            if (idle) {
                idle();
            }
            return;
        }
        if (started) {
            started(next.second);
        }
        try {
            record_test_stream(next.first, next.second, cancel);
        } catch (const std::exception&) {
            // A background boundary: a record only, the scores are saved.
        }
        const std::lock_guard lock(mutex_);
        // Read, or nothing to read: done. Cancelled by a pause: read again
        // once resumed; by stop: the queue is gone already.
        if (!cancel.cancelled() && !queue_.empty() && queue_.front() == next) {
            queue_.pop_front();
        }
    }
}

}  // namespace vml::app
