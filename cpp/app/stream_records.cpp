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

void StreamRecords::add(std::filesystem::path base, std::filesystem::path test) {
    const std::lock_guard lock(mutex_);
    if (stopped_) {
        return;
    }
    queue_.emplace_back(std::move(base), std::move(test));
    if (recording_) {
        return;  // the thread under way takes it
    }
    recording_ = true;
    if (thread_.joinable()) {
        thread_.join();  // one that has emptied the queue and is ending
    }
    thread_ = std::thread([this] { record_all(); });
}

void StreamRecords::stop() {
    const std::lock_guard lock(mutex_);
    stopped_ = true;
    queue_.clear();
    cancel_.cancel();
}

void StreamRecords::record_all() {
    while (true) {
        std::pair<std::filesystem::path, std::filesystem::path> next;
        {
            const std::lock_guard lock(mutex_);
            if (queue_.empty() || stopped_) {
                recording_ = false;
                return;
            }
            next = std::move(queue_.front());
            queue_.pop_front();
        }
        try {
            record_test_stream(next.first, next.second, cancel_);
        } catch (const std::exception&) {
            // A background boundary: a record only, the scores are saved.
        }
    }
}

}  // namespace vml::app
