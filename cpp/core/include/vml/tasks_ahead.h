#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <thread>
#include <utility>
#include <vector>

namespace vml {

// Runs task(0) to task(count - 1) ahead of need on up to `threads` threads,
// started in index order; get(i) waits for task i and gives its result, or
// throws what it threw. For work that mostly waits on other programs (ffprobe,
// FFmpeg), so that their runs overlap while the results are still taken in
// order. The destructor starts no more tasks and waits for those running.
template <class Result>
class TasksAhead {
public:
    TasksAhead(std::size_t count, std::size_t threads, std::function<Result(std::size_t)> task)
        : task_(std::move(task)), promises_(count) {
        futures_.reserve(count);
        for (std::promise<Result>& promise : promises_) {
            futures_.push_back(promise.get_future());
        }
        const std::size_t workers = std::min(count, std::max<std::size_t>(threads, 1));
        for (std::size_t i = 0; i < workers; ++i) {
            threads_.emplace_back([this] { work(); });
        }
    }
    ~TasksAhead() {
        stop_ = true;
        for (std::thread& thread : threads_) {
            thread.join();
        }
    }
    TasksAhead(const TasksAhead&) = delete;
    TasksAhead& operator=(const TasksAhead&) = delete;

    // Task `index`'s result, once (a second get of the same index throws std::future_error)
    Result get(std::size_t index) { return futures_.at(index).get(); }

private:
    void work() {
        for (std::size_t index = next_++; index < promises_.size() && !stop_; index = next_++) {
            try {
                promises_[index].set_value(task_(index));
            } catch (...) {
                promises_[index].set_exception(std::current_exception());
            }
        }
    }

    std::function<Result(std::size_t)> task_;
    std::vector<std::promise<Result>> promises_;
    std::vector<std::future<Result>> futures_;
    std::atomic<std::size_t> next_{0};
    std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;  // last: started once the rest is set up
};

}  // namespace vml
