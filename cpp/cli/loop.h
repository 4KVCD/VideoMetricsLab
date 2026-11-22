// Helpers for an encoding loop: the target command (the rate, or a setting
// such as CRF, at which a sequence's encodes reach a score, read from the
// curve bdrate integrates) and the diff command (how a new run's scores and
// weak segments differ from an old one's).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "arguments.h"
#include "vml/bd_rate_report.h"
#include "vml/results.h"
#include "vml/weak_frames.h"

namespace vml::cli {

// The JSON formats of target and diff.
inline constexpr const char* target_format = "target-1";
inline constexpr const char* diff_format = "diff-1";

// The statistics target --stat and diff take: a metric summary's names
// (results_summary.h), "score" the one the app shows.
const std::vector<std::string>& summary_stats();

// target's answers for `table`'s encodes (each file's, as one table): the
// JSON (docs/schemas/target.schema.json), the table, and the warnings of
// encodes left out (without the score, or a number in the --by column).
struct TargetAnswers {
    nlohmann::ordered_json document;
    std::string text;
    std::vector<std::string> warnings;
    std::size_t sequences = 0;  // those answered or not
    bool all_answered = true;
};
TargetAnswers target_answers(const TargetArguments& arguments, const RateTable& table);

// Runs the target command; returns the exit code: 0 when every sequence
// has an answer (extrapolated ones too); 1 when one has none, or the JSON
// file could not be written; 2 for a wrong command (a --by column no file
// has); 4 when a file cannot be read or has no scores of the metric.
int target(const TargetArguments& arguments);

// How two runs' weak segments match, by their frames: each group of old and
// new segments that overlap, chained (an old one overlapping two new ones
// joins them), as indexes into the two lists; the old ones nothing new
// overlaps (fixed) and the new ones nothing old overlaps (appeared).
struct SegmentMatch {
    struct Remaining {
        std::vector<std::size_t> old_segments;
        std::vector<std::size_t> new_segments;
    };
    std::vector<std::size_t> fixed;
    std::vector<Remaining> remaining;
    std::vector<std::size_t> appeared;
};
SegmentMatch match_segments(const std::vector<WeakSegment>& old_segments,
                            const std::vector<WeakSegment>& new_segments);

// diff's JSON (docs/schemas/diff.schema.json) of `old_result` against
// `new_result`: their videos, each statistic of each metric both have (old,
// new and the change), the metrics only one has, and with `rules` holding
// conditions, both runs' weak segments and how they match.
nlohmann::ordered_json diff_json(const ComparisonResult& old_result, const std::string& old_item,
                                 const ComparisonResult& new_result, const std::string& new_item,
                                 const SegmentRules& rules);

// Runs the diff command; returns the exit code: 0; 1 when the JSON file
// could not be written; 2 for a wrong command; 4 when a run cannot be read.
int diff(const SavedArguments& arguments);

}  // namespace vml::cli
