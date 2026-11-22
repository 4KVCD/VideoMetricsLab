#include "loop.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "codes.h"
#include "common.h"
#include "compare.h"
#include "saved.h"
#include "weak_report.h"
#include "vml/append.h"
#include "vml/bd_rate.h"
#include "vml/bd_rate_report.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/results_summary.h"
#include "vml/settings.h"
#include "vml/time_format.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

Json number_or_null(const std::optional<double>& value) { return value ? Json(*value) : Json(nullptr); }

// Writes `document` where --json says: on standard output for -, which then
// stands for the text, or to its file after the text. The exit code.
int write_outputs(const std::optional<std::string>& json_path, const Json& document, const std::string& text) {
    if (json_path && *json_path == "-") {
        return print_results(python_json_dumps(document) + "\n") ? exit_ok : exit_failed;
    }
    if (!print_results(text)) {
        return exit_failed;
    }
    if (json_path) {
        if (const std::optional<std::string> failure = write_json_file(python_path(*json_path), document)) {
            print_error("The JSON file could not be written: " + *failure);
            return exit_failed;
        }
    }
    return exit_ok;
}

// A command that could not read what it was given (exit_input): `why`
// printed, and with --json the document of the error.
int refuse(const std::optional<std::string>& json_path, const char* format, const std::string& why,
           const char* code) {
    print_error(why);
    if (json_path) {
        Json document = document_head(format);
        document["error"] = why;
        document["error_code"] = code;
        if (*json_path == "-") {
            print_results(python_json_dumps(document) + "\n");
        } else if (const std::optional<std::string> failure = write_json_file(python_path(*json_path), document)) {
            print_error("The JSON file could not be written: " + *failure);
        }
    }
    return exit_input;
}

// ------------------------------------------------------------------ target

// An encode on the curve: what it was made with, its rate and its score.
struct TargetPoint {
    double setting = 0;
    double kbps = 0;
    double score = 0;
};

// A gap's stable word in the JSON.
const char* gap_code(BdRateGap gap) {
    switch (gap) {
        case BdRateGap::too_few_points:
            return "too_few_encodes";
        case BdRateGap::tied_scores:
            return "tied_scores";
        default:
            return "bad_number";
    }
}

struct TargetSequence {
    std::string name;
    std::vector<TargetPoint> points;
};

Json point_json(const TargetPoint& point) {
    return Json{{"setting", point.setting}, {"kbps", point.kbps}, {"score", point.score}};
}

}  // namespace

const std::vector<std::string>& summary_stats() {
    static const std::vector<std::string> stats{"score",           "median",          "min",
                                                "max",             "worst_10_percent", "worst_5_percent",
                                                "worst_1_percent", "worst_0.1_percent"};
    return stats;
}

TargetAnswers target_answers(const TargetArguments& arguments, const RateTable& table) {
    TargetAnswers answers;
    const bool by_rate = arguments.by == "kbps";
    // The sequences, in name order, with their encodes' points.
    std::map<std::string, TargetSequence> sequences;
    std::map<std::string, std::size_t> unscored;
    std::map<std::string, std::size_t> unset;
    for (const RateTable::Encode& encode : table.encodes) {
        if (arguments.sequence && encode.sequence != *arguments.sequence) {
            continue;
        }
        TargetSequence& sequence = sequences[encode.sequence];
        sequence.name = encode.sequence;
        const auto score = encode.scores.find(arguments.metric);
        if (score == encode.scores.end() || !score->second) {
            ++unscored[encode.sequence];
            continue;
        }
        std::optional<double> setting = encode.kbps;
        if (!by_rate) {
            const auto cell = encode.text.find(arguments.by);
            setting = cell == encode.text.end() ? std::nullopt : python_float(cell->second);
            if (!setting) {
                ++unset[encode.sequence];
                continue;
            }
        }
        sequence.points.push_back({*setting, encode.kbps, *score->second});
    }
    for (const auto& [name, count] : unscored) {
        answers.warnings.push_back(name + ": " + std::to_string(count) + " encode(s) without a " + arguments.metric +
                                   " " + arguments.stat + " left out");
    }
    for (const auto& [name, count] : unset) {
        answers.warnings.push_back(name + ": " + std::to_string(count) + " encode(s) without a number in " +
                                   arguments.by + " left out");
    }
    const MetricDefinition& metric = metric_definition(arguments.metric);
    const int decimals = by_rate ? 1 : 2;
    const auto point_text = [&](const TargetPoint& point) {
        return format_fixed(point.setting, decimals) + " (" + metric.format_value(point.score) + ")";
    };
    Json entries = Json::array();
    std::vector<std::vector<std::string>> rows{{"sequence", arguments.by, "encodes", "from"}};
    for (auto& [name, sequence] : sequences) {
        std::vector<SettingPoint> curve;
        for (const TargetPoint& point : sequence.points) {
            curve.push_back({point.setting, point.score});
        }
        const TargetResult result = setting_for_score(curve, arguments.value, by_rate);
        answers.all_answered = answers.all_answered && result.setting.has_value();
        Json entry{{"sequence", name},
                   {"encodes", sequence.points.size()},
                   {"setting", number_or_null(result.setting)},
                   {"extrapolated", result.extrapolated}};
        std::string from;
        if (result.bracket) {
            const TargetPoint& low = sequence.points[result.bracket->first];
            const TargetPoint& high = sequence.points[result.bracket->second];
            entry["bracket"] = Json::array({point_json(low), point_json(high)});
            from = result.bracket->first == result.bracket->second ? "an encode's"
                   : result.extrapolated ? "extrapolated from " + point_text(low) + " and " + point_text(high)
                                         : point_text(low) + " to " + point_text(high);
        } else {
            entry["bracket"] = nullptr;
            from = bd_rate_gap_text(result.gap);
        }
        entry["why"] = result.setting ? Json(nullptr) : Json(bd_rate_gap_text(result.gap));
        entry["why_code"] = result.setting ? Json(nullptr) : Json(gap_code(result.gap));
        entries.push_back(std::move(entry));
        rows.push_back({name, result.setting ? format_fixed(*result.setting, decimals) : "n/a",
                        std::to_string(sequence.points.size()), from});
    }
    answers.document = document_head(target_format);
    answers.document["method"] = "pchip";
    answers.document["files"] = arguments.scores;
    answers.document["metric"] = arguments.metric;
    answers.document["stat"] = arguments.stat;
    answers.document["value"] = arguments.value;
    answers.document["by"] = arguments.by;
    answers.document["sequences"] = std::move(entries);
    answers.document["warnings"] = answers.warnings;
    std::vector<std::string> lines{"Where " + std::string(metric.short_label) + " " +
                                   (arguments.stat == "score" ? "" : arguments.stat + " ") + "reaches " +
                                   format_general(arguments.value) + ", by " + arguments.by + ":"};
    add_table(lines, rows, true);
    for (const std::string& line : lines) {
        answers.text += line + "\n";
    }
    answers.sequences = sequences.size();
    return answers;
}

int target(const TargetArguments& arguments) {
    // Every file's encodes, read, as one table.
    RateTable table;
    std::string unread;
    for (const std::filesystem::path& path : expand_videos(arguments.scores, std::nullopt)) {
        try {
            if (const std::optional<std::string> nothing = found_nothing(path)) {
                throw std::runtime_error(platform::path_text(path) + ": " + *nothing);
            }
            RateTable file = read_rate_file(read_whole_file(path), platform::path_text(path), arguments.stat);
            for (const std::string& key : file.metric_keys) {
                if (std::ranges::count(table.metric_keys, key) == 0) {
                    table.metric_keys.push_back(key);
                }
            }
            for (const std::string& column : file.columns) {
                if (std::ranges::count(table.columns, column) == 0) {
                    table.columns.push_back(column);
                }
            }
            std::ranges::move(file.encodes, std::back_inserter(table.encodes));
        } catch (const std::exception& error) {
            unread += (unread.empty() ? "" : "\n") + std::string(error.what());
        }
    }
    if (!unread.empty()) {
        return refuse(arguments.json_path, target_format, unread, codes::rates_unreadable);
    }
    if (std::ranges::count(table.metric_keys, arguments.metric) == 0) {
        return refuse(arguments.json_path, target_format, "no file has scores of " + arguments.metric,
                      codes::no_metric_scores);
    }
    if (arguments.by != "kbps" && std::ranges::count(table.columns, arguments.by) == 0) {
        print_error("argument --by: no file has a column " + arguments.by +
                    " (a CSV file's own, such as crf from compare --batch's extra columns)");
        return exit_usage;
    }
    const TargetAnswers answers = target_answers(arguments, table);
    if (answers.sequences == 0) {
        return refuse(arguments.json_path, target_format, "no encode is of sequence " + arguments.sequence.value_or(""),
                      codes::no_sequence);
    }
    if (!arguments.quiet) {
        for (const std::string& warning : answers.warnings) {
            print_error("warning: " + warning);
        }
    }
    const int written = write_outputs(arguments.json_path, answers.document, answers.text);
    return written != exit_ok ? written : answers.all_answered ? exit_ok : exit_failed;
}

// -------------------------------------------------------------------- diff

SegmentMatch match_segments(const std::vector<WeakSegment>& old_segments,
                            const std::vector<WeakSegment>& new_segments) {
    // Groups by union-find over the old segments then the new ones.
    const std::size_t count = old_segments.size() + new_segments.size();
    std::vector<std::size_t> parent(count);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    const auto root = [&parent](std::size_t at) {
        while (parent[at] != at) {
            at = parent[at] = parent[parent[at]];
        }
        return at;
    };
    for (std::size_t o = 0; o < old_segments.size(); ++o) {
        for (std::size_t n = 0; n < new_segments.size(); ++n) {
            if (old_segments[o].first_frame <= new_segments[n].last_frame &&
                new_segments[n].first_frame <= old_segments[o].last_frame) {
                parent[root(old_segments.size() + n)] = root(o);
            }
        }
    }
    SegmentMatch match;
    std::map<std::size_t, SegmentMatch::Remaining> groups;  // by root, in the old segments' order
    std::vector<std::size_t> order;
    for (std::size_t at = 0; at < count; ++at) {
        const std::size_t group = root(at);
        if (!groups.contains(group)) {
            order.push_back(group);
        }
        SegmentMatch::Remaining& members = groups[group];
        if (at < old_segments.size()) {
            members.old_segments.push_back(at);
        } else {
            members.new_segments.push_back(at - old_segments.size());
        }
    }
    for (const std::size_t group : order) {
        SegmentMatch::Remaining& members = groups[group];
        if (members.new_segments.empty()) {
            append(match.fixed, members.old_segments);
        } else if (members.old_segments.empty()) {
            append(match.appeared, members.new_segments);
        } else {
            match.remaining.push_back(std::move(members));
        }
    }
    std::ranges::sort(match.fixed);
    std::ranges::sort(match.appeared);
    return match;
}

namespace {

// The metrics `result` has, in the app's order.
std::vector<std::string> metrics_of(const ComparisonResult& result) {
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {
        if (result.has_metric(std::string(metric.key))) {
            keys.emplace_back(metric.key);
        }
    }
    return keys;
}

Json run_json(const ComparisonResult& result, const std::string& item) {
    return Json{{"item", item},
                {"reference", platform::path_text(result.source)},
                {"test", platform::path_text(result.distorted)},
                {"frames", result.compared_frame_count},
                {"fps", result.fps}};
}

bool same_reference(const ComparisonResult& old_result, const ComparisonResult& new_result) {
    return platform::name_as_compared(platform::path_text(old_result.source.lexically_normal())) ==
           platform::name_as_compared(platform::path_text(new_result.source.lexically_normal()));
}

// A statistic's number in a summary, if it has it.
std::optional<double> stat_of(const std::optional<Json>& summary, const std::string& stat) {
    return summary && summary->contains(stat) ? summary_number(summary->at(stat)) : std::nullopt;
}

std::optional<double> change_of(const std::optional<double>& old_value, const std::optional<double>& new_value) {
    if (!old_value || !new_value || !std::isfinite(*old_value) || !std::isfinite(*new_value)) {
        return std::nullopt;
    }
    return *new_value - *old_value;
}

// Segments of a segments_json list, by index: "120-145 (0:00:05.005)", ...
std::string frames_text(const Json& segments, const Json& indexes) {
    std::string text;
    for (const Json& index : indexes) {
        const Json& segment = segments.at(index.get<std::size_t>());
        text += (text.empty() ? "" : ", ") + std::to_string(segment.at("first_frame").get<int>()) + "-" +
                std::to_string(segment.at("last_frame").get<int>()) + " (" +
                format_hms(segment.at("start_time").get<double>(), 3) + ")";
    }
    return text;
}

}  // namespace

Json diff_json(const ComparisonResult& old_result, const std::string& old_item, const ComparisonResult& new_result,
               const std::string& new_item, const SegmentRules& rules) {
    Json document = document_head(diff_format);
    document["old"] = run_json(old_result, old_item);
    document["new"] = run_json(new_result, new_item);
    document["same_reference"] = same_reference(old_result, new_result);
    const std::vector<std::string> old_keys = metrics_of(old_result);
    const std::vector<std::string> new_keys = metrics_of(new_result);
    Json scores = Json::object();
    Json only_old = Json::array();
    Json only_new = Json::array();
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        const bool in_old = std::ranges::count(old_keys, key) > 0;
        const bool in_new = std::ranges::count(new_keys, key) > 0;
        if (in_old != in_new) {
            (in_old ? only_old : only_new).push_back(key);
        }
        if (!in_old || !in_new) {
            continue;
        }
        const std::optional<Json> old_summary = metric_summary(old_result, key);
        const std::optional<Json> new_summary = metric_summary(new_result, key);
        Json stats = Json::object();
        for (const std::string& stat : summary_stats()) {
            const std::optional<double> old_value = stat_of(old_summary, stat);
            const std::optional<double> new_value = stat_of(new_summary, stat);
            if (old_value || new_value) {
                stats[stat] = Json{{"old", json_number(old_value)},
                                   {"new", json_number(new_value)},
                                   {"change", json_number(change_of(old_value, new_value))}};
            }
        }
        scores[key] = Json{{"better", metric.direction == MetricDirection::lower_is_better ? "lower" : "higher"},
                           {"stats", std::move(stats)}};
    }
    document["metrics"] = std::move(scores);
    document["only_old"] = std::move(only_old);
    document["only_new"] = std::move(only_new);
    if (rules.conditions.empty()) {
        document["segments"] = nullptr;
        return document;
    }
    std::vector<WeakCondition> old_left_out;
    std::vector<WeakCondition> new_left_out;
    const SegmentRules old_rules = scored_rules(old_result.metric_results, rules, &old_left_out);
    const SegmentRules new_rules = scored_rules(new_result.metric_results, rules, &new_left_out);
    // By the conditions on metrics each has scores for, as compare's
    const std::vector<WeakSegment> old_segments = weak_segments(old_result, old_rules);
    const std::vector<WeakSegment> new_segments = weak_segments(new_result, new_rules);
    const SegmentMatch match = match_segments(old_segments, new_segments);
    Json remaining = Json::array();
    for (const SegmentMatch::Remaining& group : match.remaining) {
        remaining.push_back(Json{{"old", group.old_segments}, {"new", group.new_segments}});
    }
    document["segments"] = Json{{"old", segments_json(old_segments, old_rules, old_left_out)},
                                {"new", segments_json(new_segments, new_rules, new_left_out)},
                                {"fixed", match.fixed},
                                {"remaining", std::move(remaining)},
                                {"appeared", match.appeared}};
    return document;
}

int diff(const SavedArguments& arguments) {
    CompareArguments output = arguments.output;
    const std::string& old_item = arguments.items.at(0);
    const std::string& new_item = arguments.items.at(1);
    std::optional<ComparisonResult> runs[2];
    std::string unread;
    for (int at = 0; at < 2; ++at) {
        try {
            runs[at] = read_comparison(arguments.items.at(at));
        } catch (const std::exception& error) {
            unread += (unread.empty() ? "" : "\n") + arguments.items.at(at) + ": " + error.what();
        }
    }
    if (!unread.empty()) {
        return refuse(output.json_path, diff_format, unread, codes::run_unreadable);
    }
    const ComparisonResult& old_result = *runs[0];
    const ComparisonResult& new_result = *runs[1];
    const auto warn = [&output](const std::string& text) {
        if (!output.quiet) {
            print_error("warning: " + text);
        }
    };
    std::vector<std::string> asked = metrics_of(old_result);
    for (const std::string& key : metrics_of(new_result)) {
        if (std::ranges::count(asked, key) == 0) {
            asked.push_back(key);
        }
    }
    if (output.segments_saved) {
        for (const std::string& condition : take_saved_segments(output, Settings::load(), asked)) {
            warn("--segments " + segments_source(output) + ": " + condition + " is left out: neither run has its metric");
        }
    }
    if (const std::optional<std::string> refused = segments_refusal(output.segments, asked)) {
        throw UsageError(*refused);
    }
    const SegmentRules rules = segment_rules(output);
    if (!same_reference(old_result, new_result)) {
        warn("the runs have different references (" + name_of(old_result.source) + ", " + name_of(new_result.source) +
             "): their frames are matched by number");
    }
    const Json document = diff_json(old_result, old_item, new_result, new_item, rules);

    // The text: the two runs, a row per metric, the segments' matching.
    std::vector<std::string> lines;
    for (const auto& [label, result, item] : {std::tuple{"Old", &old_result, &old_item},
                                              std::tuple{"New", &new_result, &new_item}}) {
        lines.push_back(std::string(label) + ": " + name_of(result->distorted) + " against " + name_of(result->source) +
                        ", " + with_thousands(result->compared_frame_count) + " frames (" + *item + ")");
    }
    lines.emplace_back();
    std::vector<std::vector<std::string>> rows{{"metric", "old", "new", "change", "worst 1% old", "new", "change"}};
    for (const auto& [key, entry] : document.at("metrics").items()) {
        const MetricDefinition& metric = metric_definition(key);
        std::vector<std::string> row{std::string(metric.short_label)};
        for (const char* stat : {"score", "worst_1_percent"}) {
            const Json& stats = entry.at("stats");
            if (!stats.contains(stat)) {
                append(row, std::vector<std::string>{"", "", ""});
                continue;
            }
            const Json& values = stats.at(stat);
            row.push_back(metric.format_value(summary_number(values.at("old"))));
            row.push_back(metric.format_value(summary_number(values.at("new"))));
            row.push_back(metric.format_delta(summary_number(values.at("change"))));
        }
        rows.push_back(std::move(row));
    }
    add_table(lines, rows, false);
    for (const auto& [key, label] : {std::pair{"only_old", "only in the old run: "},
                                     std::pair{"only_new", "only in the new run: "}}) {
        if (!document.at(key).empty()) {
            std::string keys;
            for (const Json& each : document.at(key)) {
                keys += (keys.empty() ? "" : ", ") + each.get<std::string>();
            }
            lines.push_back(label + keys);
        }
    }
    if (const Json& segments = document.at("segments"); !segments.is_null()) {
        std::string conditions;
        for (const std::string& condition : condition_texts(rules.conditions)) {
            conditions += (conditions.empty() ? "" : rules.all ? " and " : " or ") + condition;
        }
        const Json& old_list = segments.at("old").at("segments");
        const Json& new_list = segments.at("new").at("segments");
        lines.emplace_back();
        lines.push_back("Weak segments (" + conditions + "): " + std::to_string(old_list.size()) + " old, " +
                        std::to_string(new_list.size()) + " new; " + std::to_string(segments.at("fixed").size()) +
                        " fixed, " + std::to_string(segments.at("remaining").size()) + " remaining, " +
                        std::to_string(segments.at("appeared").size()) + " appeared");
        for (const Json& index : segments.at("fixed")) {
            lines.push_back("  fixed      was frames " + frames_text(old_list, Json::array({index})));
        }
        for (const Json& group : segments.at("remaining")) {
            lines.push_back("  remaining  frames " + frames_text(new_list, group.at("new")) + ", was " +
                            frames_text(old_list, group.at("old")));
        }
        for (const Json& index : segments.at("appeared")) {
            lines.push_back("  appeared   frames " + frames_text(new_list, Json::array({index})));
        }
    }
    std::string text;
    for (const std::string& line : lines) {
        text += line + "\n";
    }
    return write_outputs(output.json_path, document, text);
}

}  // namespace vml::cli
