#include "vml/run_line.h"

#include <algorithm>
#include <map>
#include <set>
#include <string_view>

#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/time_format.h"
#include "vml/translatable.h"

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

// How each place is named on the line.
std::string_view place_name(std::string_view place) { return place == "CPU" ? N_("CPU metrics") : N_("GPU metrics"); }

// Where a task's metrics are calculated now: "CPU" for a task in the CPU's
// queue, and for a task in the GPU's queue whose metrics the CPU has taken
// over; "GPU" otherwise.
std::string place_of(const TaskSnapshot& task) {
    return task.lane == "cpu" || !task.cpu_keys.empty() ? "CPU" : "GPU";
}

// `english` with its {fields} filled from `values`, as Python's str.format
// fills them: "{name}", and "{name:.Nf}" for a number with N decimals.
std::string filled(std::string_view english, const WordingValues& values) {
    std::string out;
    std::size_t at = 0;
    while (at < english.size()) {
        const char c = english[at];
        if ((c == '{' || c == '}') && at + 1 < english.size() && english[at + 1] == c) {
            out += c;
            at += 2;
            continue;
        }
        const std::size_t close = c == '{' ? english.find('}', at) : std::string_view::npos;
        if (close == std::string_view::npos) {
            out += c;
            ++at;
            continue;
        }
        const std::string_view field = english.substr(at + 1, close - at - 1);
        const std::size_t colon = field.find(':');
        const std::string_view name = field.substr(0, colon);
        const std::string_view spec = colon == std::string_view::npos ? std::string_view() : field.substr(colon + 1);
        const auto found = std::find_if(values.begin(), values.end(),
                                        [name](const auto& value) { return value.first == name; });
        if (found == values.end()) {
            out += english.substr(at, close - at + 1);
        } else if (const auto* text = std::get_if<std::string>(&found->second)) {
            out += *text;
        } else if (const auto* number = std::get_if<long long>(&found->second)) {
            out += std::to_string(*number);
        } else if (spec.size() >= 3 && spec.front() == '.' && spec.back() == 'f') {
            const int decimals = std::stoi(std::string(spec.substr(1, spec.size() - 2)));
            out += format_fixed(std::get<double>(found->second), decimals);
        } else {
            out += python_repr(std::get<double>(found->second));
        }
        at = close + 1;
    }
    return out;
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

// The seconds a running unit has left.
double seconds_left(const MetricUnit& unit) {
    return static_cast<double>(std::max(0LL, unit.whole - unit.done)) / static_cast<double>(unit.scale) / unit.fps;
}

// One unit's entry on the line: `head` (for example "GPU metrics 1–2 of 5:
// VMAF v0.6.1, VMAF NEG"), then its progress or what it is waiting for.
// `holder` names this video's metrics that hold the GPU's turn.
std::string unit_text(const std::string& head, const MetricUnit& unit, bool paused, const std::string& holder,
                      const Wording& wording) {
    const auto& say = wording.text;
    if (unit.state == "waiting") {
        if (unit.task->waiting_for == "CPU") {
            return say("{kind} queued (waiting for a free CPU slot)", {{"kind", head}});
        }
        if (!holder.empty()) {
            return say("{kind} queued (waiting for this video's {other} to finish)",
                       {{"kind", head}, {"other", holder}});
        }
        return say("{kind} queued (another video is using the GPU)", {{"kind", head}});
    }
    if (unit.state == "starting" || unit.state == "queued") {
        const std::string step = unit.state == "starting" ? step_text(unit.task->step, wording) : "";
        return step.empty() ? say("{kind} starting", {{"kind", head}})
                            : say("{kind} ({step})", {{"kind", head}, {"step", step}});
    }
    if (unit.state == "done") {
        return say("{kind} done", {{"kind", head}});
    }
    if (unit.state == "failed") {  // the reason is in the video's tooltip once the run is over
        return say("{kind} failed", {{"kind", head}});
    }
    std::vector<std::string> details;
    if (paused) {
        // The last speed and time left would read as if it were still running.
        details.push_back(say("paused", {}));
    } else if (unit.fps > 0 && unit.whole > 0) {
        details.push_back(say("{fps:.1f} fps", {{"fps", unit.fps}}));
        details.push_back(say("{time} remaining", {{"time", format_hms(seconds_left(unit))}}));
    }
    std::string text = head + " " + percent(unit);
    if (!details.empty()) {
        const std::string separator = say(", ", {});
        text += " (";
        for (std::size_t at = 0; at < details.size(); ++at) {
            text += (at == 0 ? "" : separator) + details[at];
        }
        text += ")";
    }
    return text;
}

// One metric's line in its video's tooltip, for example "GPU metrics 3 of
// 5: SSIMULACRA2 40.0% (0:00:15 remaining)" or "... (done)".
std::string unit_tooltip(const std::string& line, const MetricUnit& unit, bool paused, const Wording& wording) {
    const auto& say = wording.text;
    if (unit.state == "done") {
        return say("{label} (done)", {{"label", line}});
    }
    if (unit.state == "failed") {
        return say("{label} (failed)", {{"label", line}});
    }
    if (unit.state == "running") {
        const std::string text = line + " " + percent(unit);
        if (paused) {
            return text + " (" + say("paused", {}) + ")";
        }
        if (unit.fps > 0 && unit.whole > 0) {
            return text + " (" + say("{time} remaining", {{"time", format_hms(seconds_left(unit))}}) + ")";
        }
        return text;
    }
    if (unit.state == "starting") {
        return say("{kind} starting", {{"kind", line}});
    }
    return say("{label} (queued)", {{"label", line}});
}

}  // namespace

const Wording& english() {
    static const Wording wording{
        [](std::string_view text, const WordingValues& values) { return filled(text, values); },
        [](std::string_view message) { return std::string(message); },
    };
    return wording;
}

std::string step_text(const RunStatus& status, const Wording& wording) {
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
    return text.empty() ? text : wording.message(text);
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

MetricLine metric_line(const std::vector<TaskSnapshot>& tasks, bool paused, const Wording& wording) {
    const auto& say = wording.text;
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
    const auto number_of = [&order](const std::string& place, const std::string& key) {
        const std::vector<std::string>& keys = order[place];
        return static_cast<int>(std::find(keys.begin(), keys.end(), key) - keys.begin()) + 1;
    };
    // The start of a unit's text, such as "GPU metrics 1 of 2: VMAF NEG".
    const auto head = [&](const MetricUnit& unit) {
        std::vector<int> numbers;
        for (const std::string& key : unit.keys) {
            numbers.push_back(number_of(unit.place, key));
        }
        std::sort(numbers.begin(), numbers.end());
        std::vector<std::string> labelled;
        for (const int number : numbers) {
            labelled.push_back(order[unit.place][static_cast<std::size_t>(number) - 1]);
        }
        return say("{kind} {numbers} of {count}: {labels}",
                   {{"kind", say(place_name(unit.place), {})},
                    {"numbers", numbers_text(numbers)},
                    {"count", static_cast<long long>(order[unit.place].size())},
                    {"labels", labels_of(labelled)}});
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
    MetricLine line;
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
            line.parts.push_back(say("{kind} done", {{"kind", say(place_name(place), {})}}));
        }
        for (const MetricUnit* unit : shown) {
            line.parts.push_back(unit_text(head(*unit), *unit, paused, holder(*unit), wording));
        }
    }
    for (const std::string place : {"CPU", "GPU"}) {
        for (const std::string& key : order[place]) {
            const MetricUnit& unit = *std::find_if(units.begin(), units.end(), [&place, &key](const MetricUnit& unit) {
                return unit.place == place && contains(unit.keys, key);
            });
            const std::string metric = say("{kind} {numbers} of {count}: {labels}",
                                           {{"kind", say(place_name(place), {})},
                                            {"numbers", static_cast<long long>(number_of(place, key))},
                                            {"count", static_cast<long long>(order[place].size())},
                                            {"labels", std::string(metric_definition(key).label)}});
            line.tooltip += (line.tooltip.empty() ? "" : "\n") + unit_tooltip(metric, unit, paused, wording);
        }
    }
    // Each task decodes the videos itself. A finished task no longer
    // decodes, so it is left out unless every task has finished.
    std::vector<const TaskSnapshot*> decoding;
    for (const TaskSnapshot& task : tasks) {
        if (task.decode && task.state != "done" && task.state != "failed") {
            decoding.push_back(&task);
        }
    }
    if (decoding.empty()) {
        for (const TaskSnapshot& task : tasks) {
            if (task.decode) {
                decoding.push_back(&task);
            }
        }
    }
    for (const TaskSnapshot* task : decoding) {
        const std::string place = place_of(*task);
        std::string name = say(place_name(place), {});
        const bool differs = std::any_of(decoding.begin(), decoding.end(), [&](const TaskSnapshot* other) {
            return other != task && place_of(*other) == place && other->decode != task->decode;
        });
        if (differs) {
            std::vector<int> numbers;
            for (const std::string& key : task->cpu_keys.empty() ? task->metric_keys : task->cpu_keys) {
                if (contains(order[place], key)) {
                    numbers.push_back(number_of(place, key));
                }
            }
            name += " " + numbers_text(numbers);
        }
        // A name already there keeps its place and takes the later plan, as
        // a dict's key does.
        const auto existing = std::find_if(line.plans.begin(), line.plans.end(),
                                           [&name](const auto& plan) { return plan.first == name; });
        if (existing != line.plans.end()) {
            existing->second = *task->decode;
        } else {
            line.plans.emplace_back(name, *task->decode);
        }
    }
    return line;
}

std::string decoder_text(const std::vector<std::pair<std::string, HardwareDecoding>>& plans, bool source_only,
                         const Wording& wording) {
    // Where one video is decoded, "GPU" or "CPU", by task where they differ.
    const auto side = [&plans](bool source) {
        std::vector<std::pair<std::string, std::string>> found;  // each task's name and decoder
        for (const auto& [name, plan] : plans) {
            found.emplace_back(name, (source ? plan.source : plan.distorted) ? "GPU" : "CPU");
        }
        if (!found.empty() && std::all_of(found.begin(), found.end(),
                                          [&found](const auto& one) { return one.second == found.front().second; })) {
            return found.front().second;
        }
        std::string text;
        for (const auto& [name, decoder] : found) {
            text += (text.empty() ? "" : " / ") + decoder + " (" + name + ")";
        }
        return text;
    };
    if (source_only) {  // a resolution test decodes only the source
        return wording.text("Decoder: Source: {source}", {{"source", side(true)}});
    }
    return wording.text("Decoder: Source: {source}, test video: {test}",
                        {{"source", side(true)}, {"test", side(false)}});
}

}  // namespace vml
