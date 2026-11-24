// The progress button's state (#226), in both interfaces: Ready before a
// run, the run's progress during it, and how it ended after it.
//
// RunBadge keeps which of these the button shows. Done and Cancelled stay
// until the next run starts or until the ticked videos or their metrics
// differ from those the run ended with: then the button is Ready again,
// the app being ready to run what is ticked now. badge_look gives the
// button's text, the state its accessible name says, its bar and its tone.
#pragma once

#include <QString>

#include <string>
#include <utility>
#include <vector>

namespace vml::app {

// The ticked videos and their metrics, as a run would take them: each
// video's identity and its metrics' keys, in the table's order.
using RunTicks = std::vector<std::pair<std::string, std::vector<std::string>>>;

// A run's videos in counts, and how far it has got (0 to 1).
struct RunCounts {
    int total = 0;     // the videos run (those skipped as already scored left out)
    int finished = 0;  // over, failed ones included
    int failed = 0;    // failed, wholly or in part
    double fraction = 0.0;
};

class RunBadge {
public:
    enum class Phase { ready, running, done, cancelled };

    // A run starts: its progress until it ends.
    void start();
    // The run has ended, cancelled or not, with `counts`, `tooltip` (what
    // the end said) and the `ticks` it ended with: Done or Cancelled.
    void end(bool cancelled, RunCounts counts, QString tooltip, RunTicks ticks);
    // The ticks now. Done or Cancelled, with ticks other than the run ended
    // with, is Ready again. True if the phase changed.
    bool ticks_now(const RunTicks& ticks);

    Phase phase() const { return phase_; }
    // The ended run's: for Done and Cancelled.
    const RunCounts& ended_counts() const { return counts_; }
    const QString& ended_tooltip() const { return tooltip_; }

private:
    Phase phase_ = Phase::ready;
    RunCounts counts_;
    QString tooltip_;
    RunTicks ticks_;
};

// The button's look: its text ("1 of 2 · 50 %"), its state for the
// accessible name ("1 of 2, 50 %"), its bar (0 to 1), and its tone.
struct BadgeLook {
    enum class Tone { normal, error, muted };
    QString text;
    QString state;
    double fraction = 0.0;
    Tone tone = Tone::normal;
};

// The look for `phase`: Ready; the run's progress, `paused` or not;
// Done · 2 of 2 (the bar full), or Done · 1 failed in the error colour;
// Cancelled · 1 of 2 (muted, the bar where it stopped). `counts` are the
// run's, live while it runs.
BadgeLook badge_look(RunBadge::Phase phase, bool paused, const RunCounts& counts);

// Ready's tooltip: how many test videos are ticked, and their metrics.
QString ready_tooltip(const RunTicks& ticks);

}  // namespace vml::app
