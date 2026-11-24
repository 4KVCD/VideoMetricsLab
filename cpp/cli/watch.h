// --timeout and --video-timeout: when the run, and each video's run, is
// due to be stopped. The times are seconds on any clock that only goes
// forward (the command's watch thread gives the steady clock's; the tests
// their own), so that this keeps no time of its own.
#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "vml/cancel.h"

namespace vml::cli {

// Where the command is, for what Ctrl+C, --timeout and the end of the
// standard input (--cancel-on-stdin-eof) do: outside the run (the videos
// read, the results written), the command ends at once; during the run
// (the videos scored, and the frames of those answered by their saved
// scores counted), the run is cancelled and what finished is given; once
// the results are given (a notification sent, test streams recorded),
// nothing but Ctrl+C ends it.
enum class RunStage { outside, running, results_given };

// What --timeout does when it falls due in `stage`: ends the command at
// once (outside the run), cancels the run, or nothing: once the results
// are given, which say the run was not timed out, or once the run was
// cancelled already (Ctrl+C, the input's end), whose exit code (130) a
// timeout during its cancel does not change.
enum class TimeoutAction { end_command, cancel_run, nothing };
TimeoutAction on_run_timeout(RunStage stage, bool cancelled);

// `count` cancel tokens, each of its own: a copy of a token shares its state,
// so one copied into each place would cancel them all at once.
std::vector<CancelToken> separate_tokens(std::size_t count);

class Deadlines {
public:
    // `run`: the whole command's limit, counted from `start`; `video`: each
    // video's, counted from when it starts (started()). None: no limit.
    Deadlines(double start, std::optional<double> run, std::optional<double> video);

    // Job `job` (a video, or one of its further CVVDP displays) started at
    // `now`, or ended: an ended one is due no more.
    void started(std::size_t job, double now);
    void ended(std::size_t job);

    // What is due at `now` that was not said before: each once.
    struct Due {
        bool run = false;
        std::vector<std::size_t> jobs;  // in order
    };
    Due due(double now);

private:
    double run_ends_;  // when the run is due; infinity without a limit
    std::optional<double> video_;
    bool run_said_ = false;
    std::map<std::size_t, double> running_;  // each running job's end
};

}  // namespace vml::cli
