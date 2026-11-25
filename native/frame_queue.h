// What every frame decoder library (gpu_frames.h) shares between its feeding
// thread and the app's: the pictures decoded and waiting for nvf_pop, and how
// decoding stands. Each library's Decoder derives from FrameQueue and keeps
// its own state under the same `mutex` (its free slots, its pictures).

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>

#include "gpu_frames.h"

// A decoded picture waiting for nvf_pop: its slot and time stamp.
struct Ready {
    int slot;
    long long pts;
};

struct FrameQueue {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Ready> ready;  // in display order
    bool ended = false, aborted = false, failed = false;
    std::string error;  // the first failure's, for nvf_error

    // Decoding failed: `text` kept unless an earlier failure's is, and every
    // waiter woken.
    void fail(const std::string &text) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!failed) error = text;
        failed = true;
        changed.notify_all();
    }

    bool stopped() {
        std::lock_guard<std::mutex> guard(mutex);
        return failed || aborted;
    }

    // nvf_pop: the next picture into `slot` and `pts` (NVF_FRAME), or why
    // there is none; a negative `timeout_ms` waits for as long as it takes.
    int pop(int timeout_ms, int *slot, long long *pts) {
        std::unique_lock<std::mutex> guard(mutex);
        auto settled = [this] { return !ready.empty() || ended || aborted || failed; };
        if (timeout_ms < 0) {
            changed.wait(guard, settled);
        } else if (!changed.wait_for(guard, std::chrono::milliseconds(timeout_ms), settled)) {
            return NVF_TIMEOUT;
        }
        if (aborted) return NVF_ABORTED;
        if (failed) return NVF_ERROR;
        if (ready.empty()) return NVF_END;
        *slot = ready.front().slot;
        *pts = ready.front().pts;
        ready.pop_front();
        return NVF_FRAME;
    }

    // nvf_abort: every wait ends, and decoding stops.
    void abort() {
        std::lock_guard<std::mutex> guard(mutex);
        aborted = true;
        changed.notify_all();
    }

    // nvf_error: the failure's text into the caller's buffer, and its length.
    int error_text(char *out, int size) {
        std::lock_guard<std::mutex> guard(mutex);
        copy_text(out, size, error);
        return static_cast<int>(error.size());
    }
};
