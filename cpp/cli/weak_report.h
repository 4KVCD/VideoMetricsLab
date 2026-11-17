// --worst and --segments (issue #24): each test video's worst frames and
// weak segments, as tables, in the --json and as files.
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "arguments.h"
#include "test_video.h"
#include "vml/weak_frames.h"

namespace vml::cli {

// The rules --segments and its options give.
SegmentRules segment_rules(const CompareArguments& arguments);

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

// --segments-dir: each video's <video>.segments.json and .segments.csv in
// `folder`, under names no file there has. Whether every one was written.
bool write_segment_files(const std::vector<TestVideo>& videos, const SegmentRules& rules,
                         const std::filesystem::path& folder,
                         const std::function<void(const std::string&)>& message,
                         const std::function<void(const std::string&)>& failed);

}  // namespace vml::cli
