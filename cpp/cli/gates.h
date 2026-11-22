// Quality gates: the bars a test video's scores must reach (--require) and
// whether it may have weak segments (--fail-on-segments), checked after the
// run; a video below a bar makes the command's exit code exit_gate_failed.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vml/weak_frames.h"

namespace vml::cli {

struct TestVideo;

// One bar: METRIC[.STATISTIC] OPERATOR VALUE ("vmaf>=95",
// "vmaf.worst_1_percent>=90", "butteraugli<=1.5").
struct GateRule {
    std::string metric;     // a metric's key
    std::string statistic;  // a key of its summary (results_summary.h): "score" unless given
    std::string comparison; // ">=", ">", "<=" or "<"
    double value = 0.0;

    bool met(double score) const;
    std::string text() const;  // as written back: "vmaf.worst_1_percent>=90"
};

// The statistics a rule can name: those every metric's summary has (CVVDP's
// of its per-second scores, but for its score).
const std::vector<std::string>& gate_statistics();

// The rules of a --require value, comma-separated. Throws
// std::invalid_argument saying what is wrong with one.
std::vector<GateRule> read_gate_rules(const std::string& text);

// How a video came out of one check.
struct GateCheck {
    std::string rule;  // GateRule::text, or "no weak segments"
    std::string metric;     // empty for the weak segments' check
    std::string statistic;  // empty for the weak segments' check
    // The video's figure: the statistic, or the number of weak segments;
    // none where there is none (the metric not scored, not applicable).
    std::optional<double> value;
    // Whether it holds; none where it does not apply (HDR VMAF of a video
    // that is not PQ), which does not fail the gate.
    std::optional<bool> passed;
    std::string why;  // for a check that failed with no value, or does not apply
};

// `video`'s checks: each rule against its summaries (`summary`: the metric's
// summary by key, results_summary.h), then, with `segments` (fail on weak
// segments), that it has none by those rules.
std::vector<GateCheck> gate_checks(const TestVideo& video,
                                   const std::map<std::string, nlohmann::ordered_json>& summary,
                                   const std::vector<GateRule>& rules, const SegmentRules* segments);

// Whether checks pass: none failed (those that do not apply are left out).
bool gate_passed(const std::vector<GateCheck>& checks);

// A video's "gate" in the JSON: {"passed": ..., "checks": [...]}.
nlohmann::ordered_json gate_json(const std::vector<GateCheck>& checks);

// The checks of each video that has any, as text after the tables.
std::string gate_text(const std::vector<TestVideo>& videos);

}  // namespace vml::cli
