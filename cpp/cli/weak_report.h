// --worst and --segments (issue #24): each test video's worst frames and
// weak segments, as tables, in the --json and as files.
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "arguments.h"
#include "test_video.h"
#include "vml/segment_profiles.h"
#include "vml/settings.h"
#include "vml/weak_frames.h"

namespace vml::cli {

// The rules --segments and its options give.
SegmentRules segment_rules(const CompareArguments& arguments);

// What --segments named in place of conditions: "saved", "profile:NAME" or
// "preset:KEY".
std::string segments_source(const CompareArguments& arguments);

// --segments saved (or profile:NAME, or preset:KEY): the conditions of the
// weak segment profile in use (or NAME) in the window's Settings > Weak
// segments, or of the preset
// (`settings`) on the metrics compared (`asked`) put in `arguments`, and
// its any/every choice, shortest segment and gap where the command gave
// none (--segments-all or --segments-any). Conditions given after it
// replace its condition on their metric, or are added. Those on metrics
// not compared are refused later, as given conditions are
// (segments_refusal). The saved conditions left
// out, on metrics not compared, are returned. Throws
// std::invalid_argument when none is left.
std::vector<std::string> take_saved_segments(CompareArguments& arguments, const Settings& settings,
                                             const std::vector<std::string>& asked);

// Why --segments cannot be answered with the metrics asked for: a
// condition's metric that is not one of them. Nothing when it can.
std::optional<std::string> segments_refusal(const std::vector<WeakCondition>& conditions,
                                            const std::vector<std::string>& asked);

// Each video's `count` worst dips per metric, as tables.
std::string worst_text(const std::vector<TestVideo>& videos, std::size_t count);

// Each video's weak segments, as tables: up to `shown` rows each, then how
// many more there are.
std::string segments_text(const std::vector<TestVideo>& videos, const SegmentRules& rules, std::size_t shown);

// Adds each video's segments (segments_json) to its entry of `document`
// (results_document's, whose "videos" are in `videos`' order).
void add_segments_json(nlohmann::ordered_json& document, const std::vector<TestVideo>& videos,
                       const SegmentRules& rules);

// --segments-report: every video of `videos` (with its reference) in one
// report at `file`, JSON or, for a .csv name, CSV (segments_report_json and
// _csv). Nothing when written, else why not.
std::optional<std::string> write_segments_report(
    const std::vector<std::pair<const TestVideo*, std::filesystem::path>>& videos, const SegmentRules& rules,
    const std::filesystem::path& file);

// --segments-dir: each video's <video>.<metrics>.segments.json and .csv in
// `folder`, under names no file there has (each video's segment_files says
// where they went). Whether every one was written.
bool write_segment_files(std::vector<TestVideo>& videos, const SegmentRules& rules,
                         const std::filesystem::path& folder,
                         const std::function<void(const std::string&)>& message,
                         const std::function<void(const std::string&)>& failed);

}  // namespace vml::cli
