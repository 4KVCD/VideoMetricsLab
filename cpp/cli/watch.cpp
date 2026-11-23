#include "watch.h"

#include <limits>

namespace vml::cli {

std::vector<CancelToken> separate_tokens(std::size_t count) {
    std::vector<CancelToken> tokens;
    tokens.reserve(count);
    for (std::size_t made = 0; made < count; ++made) {
        tokens.emplace_back();
    }
    return tokens;
}

Deadlines::Deadlines(double start, std::optional<double> run, std::optional<double> video)
    : run_ends_(run ? start + *run : std::numeric_limits<double>::infinity()), video_(video) {}

void Deadlines::started(std::size_t job, double now) {
    if (video_) {
        running_[job] = now + *video_;
    }
}

void Deadlines::ended(std::size_t job) { running_.erase(job); }

Deadlines::Due Deadlines::due(double now) {
    Due due;
    if (!run_said_ && now >= run_ends_) {
        run_said_ = true;
        due.run = true;
    }
    for (auto item = running_.begin(); item != running_.end();) {
        if (now >= item->second) {
            due.jobs.push_back(item->first);
            item = running_.erase(item);
        } else {
            ++item;
        }
    }
    return due;
}

}  // namespace vml::cli
