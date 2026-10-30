// Passing work and failures between threads.
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

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

// A thread of its own that runs one job at a time beside its caller's: a
// step of a frame's work made at the same time as another (a decoder's
// download beside the other video's), without starting a thread per frame.
// Every job run() starts must be waited for (wait) before what it uses goes.
class SideThread {
public:
    SideThread() : thread_([this] { serve(); }) {}
    ~SideThread() {
        jobs_.put(nullptr);
        thread_.join();
    }
    SideThread(const SideThread&) = delete;
    SideThread& operator=(const SideThread&) = delete;

    void run(std::function<void()> job) { jobs_.put(std::move(job)); }

    // Waits for the job run() started, and throws what it threw.
    void wait() {
        if (const std::exception_ptr error = done_.get()) {
            std::rethrow_exception(error);
        }
    }

    // wait(), for a caller unwinding from a failure of its own: the job's is
    // dropped.
    void wait_quietly() noexcept {
        try {
            wait();
        } catch (...) {
            // The caller's failure is the one to report.
        }
    }

    // Runs `there` on this thread and `here` on the caller's at the same
    // time, and returns once both are done. Throws `here`'s failure, else
    // `there`'s.
    void both(const std::function<void()>& there, const std::function<void()>& here) {
        run(there);
        try {
            here();
        } catch (...) {
            wait_quietly();
            throw;
        }
        wait();
    }

private:
    void serve() {
        while (std::function<void()> job = jobs_.get()) {
            std::exception_ptr error;
            try {
                job();
            } catch (...) {
                error = std::current_exception();
            }
            done_.put(error);
        }
    }

    Channel<std::function<void()>> jobs_;
    Channel<std::exception_ptr> done_;
    std::thread thread_;  // last: it starts once the channels are made
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
