// Cancelling a run (the window's Cancel, the command line's Ctrl+C), and
// pausing it (the window's Pause).
#pragma once

#include <memory>
#include <stdexcept>

namespace vml {

namespace platform {
class ChildProcess;
}

// A run was cancelled by the user: nothing went wrong.
class Cancelled : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("Cancelled by user") {}
};

// A run's Cancel and Pause, as the Python app's cancel event and process
// handles hold them; copies share them. The run's code checks cancelled(),
// and attaches the programs it starts, which Cancel ends at once and Pause
// holds until resume(): a program held writes nothing, so the code reading
// it could not look at the flag meanwhile. A program attached after a
// Cancel is ended at once, and one attached while paused is held at once.
// Work in a process of its own (isolated.h) is held with that process.
class CancelToken {
    struct State;

public:
    CancelToken();

    // Stops the run: sets the flag, and ends every program attached.
    void cancel();
    bool cancelled() const;
    // Throws Cancelled once the run has been cancelled.
    void throw_if_cancelled() const;

    // Holds every program attached, and each attached later, until resume().
    void pause();
    void resume();
    bool paused() const;

    // A program attached to the run, while this lives.
    class Attached {
    public:
        Attached(Attached&& other) noexcept;
        Attached& operator=(Attached&&) = delete;
        Attached(const Attached&) = delete;
        Attached& operator=(const Attached&) = delete;
        // Detaches the program, letting go of the run's hold on it.
        ~Attached();

    private:
        friend class CancelToken;
        Attached(std::shared_ptr<State> state, platform::ChildProcess* process);

        std::shared_ptr<State> state_;
        platform::ChildProcess* process_;
    };

    // Attaches `process` to the run until the returned Attached is gone. The
    // process must stay where it is (not be moved) until then: declare the
    // Attached after it, so that it goes first.
    [[nodiscard]] Attached attach(platform::ChildProcess& process) const;

private:
    std::shared_ptr<State> state_;
};

}  // namespace vml
