// Metric Graphs' Worst frames list as Video Compare steps through it with
// Previous / Next worst (issue #24).
#pragma once

#include <QString>

#include <string>
#include <vector>

namespace vml::app {

// Whose frames they are (a graph line's identity, Video Compare's pair), the
// metric, and each frame with its score as the list shows it, worst first.
// No stops: not opened from the list.
struct WorstFrameList {
    struct Stop {
        long long frame = 0;
        QString score;
    };
    std::string identity;
    QString metric;
    std::vector<Stop> stops;
};

}  // namespace vml::app
