// Cancelling a run (the window's Cancel, the command line's Ctrl+C).
#pragma once

#include <atomic>
#include <memory>
#include <stdexcept>

namespace vml {

// A run was cancelled by the user: nothing went wrong.
class Cancelled : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("Cancelled by user") {}
};

// A flag the run checks and the user sets; copies share it.
class CancelToken {
public:
    void cancel() { flag_->store(true); }
    bool cancelled() const { return flag_->load(); }
    // Throws Cancelled once the run has been cancelled.
    void throw_if_cancelled() const {
        if (cancelled()) {
            throw Cancelled();
        }
    }

private:
    std::shared_ptr<std::atomic<bool>> flag_ = std::make_shared<std::atomic<bool>>(false);
};

}  // namespace vml
