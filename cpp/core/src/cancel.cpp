#include "vml/cancel.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

#include "vml/platform/process.h"

namespace vml {

// The programs are acted on with `mutex` held: their owners detach them
// under it before they go, so none is gone while it is acted on, and a Pause
// cannot come between a program's attaching and its being held.
struct CancelToken::State {
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    bool paused = false;
    std::vector<platform::ChildProcess*> processes;
};

CancelToken::CancelToken() : state_(std::make_shared<State>()) {}

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

void CancelToken::pause() {
    const std::lock_guard lock(state_->mutex);
    if (state_->paused) {
        return;
    }
    state_->paused = true;
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
    for (platform::ChildProcess* process : state_->processes) {
        process->resume();
    }
}

bool CancelToken::paused() const {
    const std::lock_guard lock(state_->mutex);
    return state_->paused;
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
