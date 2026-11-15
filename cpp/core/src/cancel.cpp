#include "vml/cancel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <utility>
#include <vector>

#include "vml/platform/process.h"

namespace vml {

// The programs are acted on with `mutex` held: their owners detach them
// under it before they go, so none is gone while it is acted on, and a Pause
// cannot come between a program's attaching and its being held.
struct CancelToken::State {
    explicit State(RunClock run_clock) : clock(std::move(run_clock)) {}

    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    bool paused = false;
    std::vector<platform::ChildProcess*> processes;
    const RunClock clock;
    double paused_since = 0.0;   // the clock when the pause on began
    double paused_before = 0.0;  // the pauses that ended, in all
};

double steady_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

CancelToken::CancelToken() : CancelToken(steady_seconds) {}

CancelToken::CancelToken(RunClock clock) : state_(std::make_shared<State>(std::move(clock))) {}

void CancelToken::cancel() {
    const std::lock_guard lock(state_->mutex);
    state_->cancelled = true;
    for (platform::ChildProcess* process : state_->processes) {
        process->terminate();
    }
}

bool CancelToken::cancelled() const { return state_->cancelled.load(); }

void CancelToken::throw_if_cancelled() const {
    if (cancelled()) {
        throw Cancelled();
    }
}

std::unique_lock<std::timed_mutex> lock_unless_cancelled(std::timed_mutex& mutex, const CancelToken& cancel) {
    std::unique_lock lock(mutex, std::defer_lock);
    if (lock.try_lock()) {
        return lock;
    }
    // Cancel is looked at between waits short enough to answer it at once.
    do {
        cancel.throw_if_cancelled();
    } while (!lock.try_lock_for(std::chrono::milliseconds(100)));
    return lock;
}

void CancelToken::pause() {
    const std::lock_guard lock(state_->mutex);
    if (state_->paused) {
        return;
    }
    state_->paused = true;
    state_->paused_since = state_->clock();
    for (platform::ChildProcess* process : state_->processes) {
        process->suspend();
    }
}

void CancelToken::resume() {
    const std::lock_guard lock(state_->mutex);
    if (!state_->paused) {
        return;
    }
    state_->paused = false;
    state_->paused_before += state_->clock() - state_->paused_since;
    for (platform::ChildProcess* process : state_->processes) {
        process->resume();
    }
}

bool CancelToken::paused() const {
    const std::lock_guard lock(state_->mutex);
    return state_->paused;
}

double CancelToken::now() const { return state_->clock(); }

double CancelToken::paused_seconds() const {
    const std::lock_guard lock(state_->mutex);
    return state_->paused_before + (state_->paused ? state_->clock() - state_->paused_since : 0.0);
}

CancelToken::Attached CancelToken::attach(platform::ChildProcess& process) const {
    const std::lock_guard lock(state_->mutex);
    state_->processes.push_back(&process);
    if (state_->cancelled) {
        process.terminate();
    } else if (state_->paused) {
        process.suspend();
    }
    return Attached(state_, &process);
}

CancelToken::Attached::Attached(std::shared_ptr<State> state, platform::ChildProcess* process)
    : state_(std::move(state)), process_(process) {}

CancelToken::Attached::Attached(Attached&& other) noexcept
    : state_(std::move(other.state_)), process_(std::exchange(other.process_, nullptr)) {}

CancelToken::Attached::~Attached() {
    if (process_ == nullptr) {
        return;
    }
    const std::lock_guard lock(state_->mutex);
    std::erase(state_->processes, process_);
    if (state_->paused) {
        process_->resume();
    }
}

}  // namespace vml
