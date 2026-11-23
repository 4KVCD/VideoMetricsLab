#include "gates.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "common.h"
#include "test_video.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/python_text.h"
#include "vml/results_summary.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

constexpr const char* no_weak_segments = "no weak segments";

// A summary's number: a number, or Python's "inf", "-inf" and "nan" as the
// results write PSNR of identical frames; nothing for null or another value.
std::optional<double> summary_number(const Json& value) {
    if (value.is_number()) {
        return value.get<double>();
    }
    if (value.is_string()) {
        const std::string text = value.get<std::string>();
        if (text == "inf") {
            return std::numeric_limits<double>::infinity();
        }
        if (text == "-inf") {
            return -std::numeric_limits<double>::infinity();
        }
        if (text == "nan") {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    return std::nullopt;
}

// The statistic of a metric's summary: its own, else (CVVDP, whose summary
// has its score and the rest per second) its per-second scores'.
std::optional<double> statistic_of(const Json& summary, const std::string& statistic) {
    if (summary.contains(statistic)) {
        return summary_number(summary.at(statistic));
    }
    if (summary.contains("per_second") && summary.at("per_second").contains(statistic)) {
        return summary_number(summary.at("per_second").at(statistic));
    }
    return std::nullopt;
}

// Why `video` has no score of `metric`: its failure, or its own error.
std::string why_not_scored(const TestVideo& video, const std::string& metric) {
    for (const auto& [key, reason] : video.failures) {
        if (key == metric) {
            return reason;
        }
    }
    if (!video.error.empty()) {
        return video.error;
    }
    return "it was not scored";
}

// Every metric's key, in the registry's order, as --metrics lists them.
std::string metric_keys() {
    std::string keys;
    for (const MetricDefinition& metric : metrics()) {
        keys += (keys.empty() ? "" : ", ") + std::string(metric.key);
    }
    return keys;
}

std::string value_text(const std::optional<double>& value) {
    if (!value) {
        return "-";
    }
    if (std::isinf(*value)) {
        return *value > 0 ? "inf" : "-inf";
    }
    return format_general(*value, 6);
}

}  // namespace

bool GateRule::met(double score) const {
    if (std::isnan(score)) {
        return false;
    }
    return comparison == ">=" ? score >= value
           : comparison == ">" ? score > value
           : comparison == "<=" ? score <= value
                                : score < value;
}

std::string GateRule::text() const {
    return metric + (statistic == "score" ? "" : "." + statistic) + comparison + format_general(value, 12);
}

const std::vector<std::string>& gate_statistics() {
    static const std::vector<std::string> names{"score", "median", "min", "max", "worst_10_percent",
                                                "worst_5_percent", "worst_1_percent", "worst_0.1_percent"};
    return names;
}

std::vector<GateRule> read_gate_rules(const std::string& text) {
    std::vector<GateRule> rules;
    for (std::size_t start = 0; start <= text.size();) {
        std::size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string part = ascii_lower(python_strip(text.substr(start, end - start)));
        start = end + 1;
        if (part.empty()) {
            continue;
        }
        const std::size_t at = part.find_first_of("<>");
        const std::string shape = "write METRIC>=VALUE or METRIC.STATISTIC>=VALUE (>=, >, <= or <), such as vmaf>=95";
        if (at == std::string::npos || at == 0) {
            throw std::invalid_argument("invalid rule " + python_string_repr(part) + ": " + shape);
        }
        GateRule rule;
        const bool or_equal = at + 1 < part.size() && part[at + 1] == '=';
        rule.comparison = std::string(1, part[at]) + (or_equal ? "=" : "");
        const std::optional<double> value = python_float(part.substr(at + rule.comparison.size()));
        if (!value || !std::isfinite(*value)) {
            throw std::invalid_argument("invalid rule " + python_string_repr(part) + ": " + shape);
        }
        rule.value = *value;
        const std::string named(python_strip(part.substr(0, at)));
        const std::size_t dot = named.find('.');
        rule.metric = named.substr(0, dot);
        rule.statistic = dot == std::string::npos ? "score" : named.substr(dot + 1);
        if (find_metric(rule.metric) == nullptr) {
            throw std::invalid_argument("unknown metric " + rule.metric + "; choose from: " + metric_keys());
        }
        const std::vector<std::string>& statistics = gate_statistics();
        if (std::find(statistics.begin(), statistics.end(), rule.statistic) == statistics.end()) {
            std::string listed;
            for (const std::string& name : statistics) {
                listed += (listed.empty() ? "" : ", ") + name;
            }
            throw std::invalid_argument("unknown statistic " + rule.statistic + " (" + rule.metric +
                                        "); choose from: " + listed);
        }
        rules.push_back(std::move(rule));
    }
    if (rules.empty()) {
        throw std::invalid_argument("no rule given: write METRIC>=VALUE, such as vmaf>=95");
    }
    return rules;
}

std::vector<GateCheck> gate_checks(const TestVideo& video, const std::map<std::string, Json>& summary,
                                   const std::vector<GateRule>& rules, const SegmentRules* segments) {
    std::vector<GateCheck> checks;
    for (const GateRule& rule : rules) {
        GateCheck& check = checks.emplace_back();
        check.rule = rule.text();
        check.metric = rule.metric;
        check.statistic = rule.statistic;
        const auto not_applicable =
            std::find_if(video.not_applicable.begin(), video.not_applicable.end(),
                         [&rule](const auto& item) { return item.first == rule.metric; });
        if (not_applicable != video.not_applicable.end()) {
            check.why = not_applicable->second.why;  // passed: none, which does not fail the gate
            continue;
        }
        const auto found = summary.find(rule.metric);
        if (found == summary.end() || !found->second.contains("score") || found->second.at("score").is_null()) {
            check.passed = false;
            check.why = why_not_scored(video, rule.metric);
            continue;
        }
        check.value = statistic_of(found->second, rule.statistic);
        if (!check.value) {
            check.passed = false;
            check.why = rule.metric + " has no " + rule.statistic + " here";
            continue;
        }
        check.passed = rule.met(*check.value);
    }
    if (segments != nullptr) {
        GateCheck& check = checks.emplace_back();
        check.rule = no_weak_segments;
        if (!video.result) {
            check.passed = false;
            check.why = video.error.empty() ? "it was not scored" : video.error;
        } else {
            const std::size_t found = weak_segments(*video.result, *segments).size();
            check.value = static_cast<double>(found);
            check.passed = found == 0;
        }
    }
    return checks;
}

bool gate_passed(const std::vector<GateCheck>& checks) {
    return std::none_of(checks.begin(), checks.end(),
                        [](const GateCheck& check) { return check.passed.has_value() && !*check.passed; });
}

Json gate_json(const std::vector<GateCheck>& checks) {
    Json listed = Json::array();
    for (const GateCheck& check : checks) {
        Json entry{{"rule", check.rule}};
        if (!check.metric.empty()) {
            entry["metric"] = check.metric;
            entry["statistic"] = check.statistic;
        }
        entry["value"] = check.value ? json_number(*check.value) : Json(nullptr);
        entry["passed"] = check.passed ? Json(*check.passed) : Json(nullptr);
        if (!check.why.empty()) {
            entry["why"] = check.why;
        }
        listed.push_back(std::move(entry));
    }
    return Json{{"passed", gate_passed(checks)}, {"checks", std::move(listed)}};
}

std::string gate_text(const std::vector<TestVideo>& videos) {
    std::vector<std::string> lines;
    for (const TestVideo& video : videos) {
        if (video.checks.empty()) {
            continue;
        }
        lines.emplace_back("");
        lines.push_back(video.name() + ": quality gate " + (gate_passed(video.checks) ? "passed" : "FAILED"));
        std::vector<std::vector<std::string>> rows{{"Rule", "Value", "Result"}};
        for (const GateCheck& check : video.checks) {
            const std::string result = !check.passed ? "not applicable" : *check.passed ? "passed" : "FAILED";
            rows.push_back({check.rule, value_text(check.value),
                            check.why.empty() ? result : result + ": " + check.why});
        }
        add_table(lines, rows, true);
    }
    std::string text;
    for (const std::string& line : lines) {
        text += line + "\n";
    }
    return text;
}

}  // namespace vml::cli
