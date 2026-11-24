#include "run_badge.h"

#include "i18n.h"

#include <QStringList>

#include <algorithm>

namespace vml::app {

void RunBadge::start() {
    phase_ = Phase::running;
    counts_ = {};
    tooltip_.clear();
    ticks_.clear();
}

void RunBadge::end(bool cancelled, RunCounts counts, QString tooltip, RunTicks ticks) {
    phase_ = cancelled ? Phase::cancelled : Phase::done;
    counts_ = counts;
    tooltip_ = std::move(tooltip);
    ticks_ = std::move(ticks);
}

bool RunBadge::ticks_now(const RunTicks& ticks) {
    if ((phase_ != Phase::done && phase_ != Phase::cancelled) || ticks == ticks_) {
        return false;
    }
    phase_ = Phase::ready;
    return true;
}

BadgeLook badge_look(RunBadge::Phase phase, bool paused, const RunCounts& counts) {
    const QString of = app::tr("{done} of {total}", {{"done", counts.finished}, {"total", counts.total}});
    QStringList parts;
    BadgeLook look;
    switch (phase) {
    case RunBadge::Phase::ready:
        parts = {app::tr("Ready")};
        break;
    case RunBadge::Phase::running:
        look.fraction = std::clamp(counts.fraction, 0.0, 1.0);
        parts = {of, app::tr("{percent} %", {{"percent", static_cast<int>(look.fraction * 100)}})};
        if (paused) {
            parts.prepend(app::tr("Paused"));
        }
        break;
    case RunBadge::Phase::done:
        look.fraction = 1.0;  // every video is over, failed or not
        if (counts.failed > 0) {
            parts = {app::tr("Done"), app::tr("{count} failed", {{"count", counts.failed}})};
            look.tone = BadgeLook::Tone::error;
        } else {
            parts = {app::tr("Done"), of};
        }
        break;
    case RunBadge::Phase::cancelled:
        look.fraction = std::clamp(counts.fraction, 0.0, 1.0);  // where it stopped
        parts = {app::tr("Cancelled"), of};
        look.tone = BadgeLook::Tone::muted;
        break;
    }
    look.text = parts.join(QStringLiteral(" · "));
    look.state = parts.join(app::tr(", "));
    return look;
}

QString ready_tooltip(const RunTicks& ticks) {
    if (ticks.empty()) {
        return app::tr("Ready: no test videos ticked.");
    }
    const auto [fewest, most] = std::minmax_element(
        ticks.begin(), ticks.end(), [](const auto& a, const auto& b) { return a.second.size() < b.second.size(); });
    const auto low = static_cast<long long>(fewest->second.size());
    const auto high = static_cast<long long>(most->second.size());
    const auto videos = static_cast<long long>(ticks.size());
    const QString metrics = low != high      ? app::tr("{low} to {high} metrics", {{"low", low}, {"high", high}})
                            : videos == 1    ? ntr("{count} metric", "{count} metrics", low)
                                             : ntr("{count} metric each", "{count} metrics each", low);
    return app::tr("Ready: {videos}, {metrics}.",
                   {{"videos", ntr("{count} test video ticked", "{count} test videos ticked", videos)},
                    {"metrics", metrics}});
}

}  // namespace vml::app
