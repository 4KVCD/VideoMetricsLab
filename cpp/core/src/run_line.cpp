#include "vml/run_line.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string_view>

#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/time_format.h"

namespace vml {

namespace {

constexpr std::string_view white_space = " \t\r\n\f\v";
constexpr std::string_view ellipsis = "…";

// A group of a video's metrics that are calculated together, in one place,
// with how far along they are: one entry on the video's line.
struct MetricUnit {
    std::string place;  // "CPU" or "GPU": where the metrics are calculated now
    std::vector<std::string> keys;
    // "running", "starting", "waiting", "queued" (a later pass of a task),
    // "done" or "failed"
    std::string state;
    const TaskSnapshot* task = nullptr;
    long long done = 0;  // progress is done / whole; the line rounds it down
    long long whole = 0;
    // whole / scale is the pass's frame count, so (whole - done) / scale /
    // fps is the seconds left.
    long long scale = 1;
    double fps = 0.0;
};

bool contains(const std::vector<std::string>& keys, const std::string& key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

std::string labels_of(const std::vector<std::string>& keys) {
    std::string labels;
    for (const std::string& key : keys) {
        labels += (labels.empty() ? "" : ", ") + std::string(metric_definition(key).label);
    }
    return labels;
}

// Splits a video's metrics into the units they are calculated in, in the
// order they run. A unit is one pass of one or more metrics, in one place.
// Each metric appears in one unit only: the last one it is in, for example
// its retry. When the CPU takes over the metrics of a task in the GPU's
// queue (after a GPU failure), they are shown as CPU metrics, with the
// CPU's figures.
std::vector<MetricUnit> metric_units(const std::vector<TaskSnapshot>& tasks) {
    std::vector<MetricUnit> units;
    for (const TaskSnapshot& task : tasks) {
        // Adds the units for a task's finished metrics: done or failed.
        const auto finished = [&units, &task](const std::vector<std::string>& keys, const std::string& place) {
            std::vector<std::string> kept;
            std::vector<std::string> failed;
            for (const std::string& key : keys) {
                (contains(task.done_keys, key) ? kept : failed).push_back(key);
            }
            if (!kept.empty()) {
                units.push_back({place, kept, "done", &task});
            }
            if (!failed.empty()) {
                units.push_back({place, failed, "failed", &task});
            }
        };
        // Adds the unit for metrics still being calculated.
        const auto under_way = [&units, &task](const std::vector<std::string>& keys, const std::string& place) {
            units.push_back({place, keys, task.state, &task, task.current, task.total, 1, task.fps});
        };
        const bool ended = task.state == "done" || task.state == "failed";
        if (task.lane == "cpu" || !task.cpu_keys.empty()) {
            // One pass of the task's metrics on the CPU, after any passes on
            // the GPU.
            if (!task.cpu_keys.empty()) {
                for (const std::vector<std::string>& group : task.passes) {
                    std::vector<std::string> on_gpu;
                    for (const std::string& key : group) {
                        if (!contains(task.cpu_keys, key)) {
                            on_gpu.push_back(key);
                        }
                    }
                    finished(on_gpu, "GPU");
                }
            }
            const std::vector<std::string>& keys = task.cpu_keys.empty() ? task.metric_keys : task.cpu_keys;
            if (ended) {
                finished(keys, "CPU");
            } else {
                under_way(keys, "CPU");
            }
            continue;
        }
        std::vector<std::vector<std::string>> passes;
        for (const std::vector<std::string>& group : task.passes) {
            if (!group.empty()) {
                passes.push_back(group);
            }
        }
        if (passes.empty()) {
            passes.push_back(task.metric_keys);
        }
        if (ended) {
            for (const std::vector<std::string>& group : passes) {
                finished(group, "GPU");
            }
            continue;
        }
        if (task.state != "running" || !task.phase) {
            under_way(passes.front(), "GPU");
            for (std::size_t position = 1; position < passes.size(); ++position) {
                units.push_back({"GPU", passes[position], "queued", &task});
            }
            continue;
        }
        const auto [number, count, start] = *task.phase;
        for (std::size_t position = 1; position <= passes.size(); ++position) {
            const auto at = static_cast<long long>(position);
            if (at < number) {
                finished(passes[position - 1], "GPU");
            } else if (at == number) {
                // All of the task's passes share one progress count (Vship's
                // task). This pass counts from `start`, out of `left` passes
                // of the same length.
                const long long left = std::max(1LL, count - number + 1);
                units.push_back({"GPU", passes[position - 1], "running", &task, (task.current - start) * left,
                                 task.total - start, left, task.fps});
            } else {
                units.push_back({"GPU", passes[position - 1], "queued", &task});
            }
        }
    }
    std::vector<MetricUnit> kept;
    std::set<std::string> seen;
    for (auto unit = units.rbegin(); unit != units.rend(); ++unit) {
        std::vector<std::string> keys;
        for (const std::string& key : unit->keys) {
            if (seen.insert(key).second) {
                keys.push_back(key);
            }
        }
        if (!keys.empty()) {
            kept.push_back(*unit);
            kept.back().keys = std::move(keys);
        }
    }
    std::reverse(kept.begin(), kept.end());
    return kept;
}

// The unit's progress as a percentage, rounded down, so that 99.96% does
// not read "100.0%" while work remains.
std::string percent(const MetricUnit& unit) {
    const long long tenths = unit.whole > 0 ? std::min(1000LL, 1000 * std::max(0LL, unit.done) / unit.whole) : 0;
    return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + "%";
}

// One unit's entry on the line: `head` (for example "GPU metrics 1–2 of 5:
// VMAF v0.6.1, VMAF NEG"), then its progress or what it is waiting for.
// `holder` names this video's metrics that hold the GPU's turn.
std::string unit_text(const std::string& head, const MetricUnit& unit, bool paused, const std::string& holder) {
    if (unit.state == "waiting") {
        if (unit.task->waiting_for == "CPU") {
            return head + " queued (waiting for a free CPU slot)";
        }
        if (!holder.empty()) {
            return head + " queued (waiting for this video's " + holder + " to finish)";
        }
        return head + " queued (another video is using the GPU)";
    }
    if (unit.state == "starting" || unit.state == "queued") {
        const std::string step = unit.state == "starting" ? step_text(unit.task->step) : "";
        return step.empty() ? head + " starting" : head + " (" + step + ")";
    }
    if (unit.state == "done") {
        return head + " done";
    }
    if (unit.state == "failed") {
        return head + " failed";
    }
    std::string details;
    if (paused) {
        // The last speed and time left would read as if it were still running.
        details = "paused";
    } else if (unit.fps > 0 && unit.whole > 0) {
        const double seconds =
            static_cast<double>(std::max(0LL, unit.whole - unit.done)) / static_cast<double>(unit.scale) / unit.fps;
        details = format_fixed(unit.fps, 1) + " fps, " + format_hms(seconds) + " remaining";
    }
    return head + " " + percent(unit) + (details.empty() ? "" : " (" + details + ")");
}

}  // namespace

std::string step_text(const RunStatus& status) {
    if (status.kind == StatusKind::starting || status.kind == StatusKind::gpu_pass) {
        return "";
    }
    std::string text = status.brief.empty() ? status.text : status.brief;
    const auto trim = [&text] {
        text.erase(0, std::min(text.size(), text.find_first_not_of(white_space)));
        text.erase(text.find_last_not_of(white_space) + 1);
    };
    trim();
    while (true) {  // the dots and ellipses that end it
        if (text.ends_with('.')) {
            text.pop_back();
        } else if (text.ends_with(ellipsis)) {
            text.erase(text.size() - ellipsis.size());
        } else {
            break;
        }
    }
    trim();
    const std::string_view distorted = "distorted";
    const std::string_view test_video = "test video";
    for (std::size_t at = text.find(distorted); at != std::string::npos; at = text.find(distorted, at)) {
        text.replace(at, distorted.size(), test_video);
        at += test_video.size();
    }
    return text;
}

std::string numbers_text(std::vector<int> numbers) {
    std::sort(numbers.begin(), numbers.end());
    if (numbers.size() > 1 && numbers.back() - numbers.front() == static_cast<int>(numbers.size()) - 1) {
        return std::to_string(numbers.front()) + "–" + std::to_string(numbers.back());
    }
    std::string text;
    for (const int number : numbers) {
        text += (text.empty() ? "" : ", ") + std::to_string(number);
    }
    return text;
}

std::vector<std::string> metric_line(const std::vector<TaskSnapshot>& tasks, bool paused) {
    const std::vector<MetricUnit> units = metric_units(tasks);
    std::map<std::string, std::vector<std::string>> order;  // each place's metrics, numbered in this order
    for (const MetricUnit& unit : units) {
        std::vector<std::string>& keys = order[unit.place];
        for (const std::string& key : unit.keys) {
            if (!contains(keys, key)) {
                keys.push_back(key);
            }
        }
    }
    // The start of a unit's text, such as "GPU metrics 1 of 2: VMAF NEG".
    const auto head = [&order](const MetricUnit& unit) {
        const std::vector<std::string>& keys = order[unit.place];
        std::vector<int> numbers;
        for (const std::string& key : unit.keys) {
            numbers.push_back(static_cast<int>(std::find(keys.begin(), keys.end(), key) - keys.begin()) + 1);
        }
        std::sort(numbers.begin(), numbers.end());
        std::vector<std::string> labelled;
        for (const int number : numbers) {
            labelled.push_back(keys[static_cast<std::size_t>(number) - 1]);
        }
        return unit.place + " metrics " + numbers_text(numbers) + " of " + std::to_string(keys.size()) + ": " +
               labels_of(labelled);
    };
    // The metrics of this video whose task holds the GPU's turn while `unit`
    // waits for it. That task may be running on the CPU after a GPU
    // failure; it keeps the GPU's turn until it finishes.
    const auto holder = [&units](const MetricUnit& unit) {
        for (const MetricUnit& other : units) {
            if (other.task != unit.task && (other.state == "running" || other.state == "starting") &&
                other.task->lane == "gpu") {
                return labels_of(other.keys);
            }
        }
        return std::string();
    };
    std::vector<std::string> parts;
    for (const std::string place : {"CPU", "GPU"}) {
        if (order[place].empty()) {
            continue;
        }
        const auto in_place = [&units, &place](const std::function<bool(const MetricUnit&)>& wanted) {
            std::vector<const MetricUnit*> found;
            for (const MetricUnit& unit : units) {
                if (unit.place == place && wanted(unit)) {
                    found.push_back(&unit);
                }
            }
            return found;
        };
        // The units under way; else the first that waits; else the first
        // queued; else those that failed.
        std::vector<const MetricUnit*> shown =
            in_place([](const MetricUnit& unit) { return unit.state == "running" || unit.state == "starting"; });
        if (shown.empty()) {
            shown = in_place([](const MetricUnit& unit) { return unit.state == "waiting"; });
            shown.resize(std::min<std::size_t>(shown.size(), 1));
        }
        if (shown.empty()) {
            shown = in_place([](const MetricUnit& unit) { return unit.state == "queued"; });
            shown.resize(std::min<std::size_t>(shown.size(), 1));
        }
        if (shown.empty()) {
            shown = in_place([](const MetricUnit& unit) { return unit.state == "failed"; });
        }
        if (shown.empty()) {
            parts.push_back(place + " metrics done");
        }
        for (const MetricUnit* unit : shown) {
            parts.push_back(unit_text(head(*unit), *unit, paused, holder(*unit)));
        }
    }
    return parts;
}

}  // namespace vml
