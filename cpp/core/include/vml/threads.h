// Passing work and failures between threads.
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <string>

namespace vml {

// A queue between threads: put() from one, get() in another, which waits
// until there is an item.
template <typename T>
class Channel {
public:
    void put(T item) {
        {
            const std::lock_guard lock(mutex_);
            items_.push_back(std::move(item));
        }
        ready_.notify_one();
    }

    T get() {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return !items_.empty(); });
        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    // The next item, or nothing after `timeout`.
    std::optional<T> get_for(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        if (!ready_.wait_for(lock, timeout, [this] { return !items_.empty(); })) {
            return std::nullopt;
        }
        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<T> items_;
};

// The message of the exception `error` holds, as a thread that caught it
// passes it on.
inline std::string message_of(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& caught) {
        return caught.what();
    } catch (...) {
        return "an unknown error";
    }
}

}  // namespace vml
