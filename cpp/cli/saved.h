// The saved results without the videos: the saved command (list, show and
// delete the comparisons the app's results cache holds, as the window's
// Saved Results tab does) and the report command (run files, .metrics.json).
// show and report give what compare gives for the same scores: the tables,
// --json, --csv, --frames-json, --run-dir, the weak segments and the worst
// dips, as compare's report.h and weak_report.h make them.
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "arguments.h"
#include "vml/saved_results.h"
#include "vml/settings.h"

namespace vml::cli {

// saved list --json's format.
inline constexpr const char* saved_list_format = "saved-list-1";

// A saved comparison's ID: its folder's name (64 hex digits, as the results
// cache names them).
std::string saved_id(const SavedComparison& comparison);

// The comparisons `item` names: those whose ID begins with it (at least
// `min_prefix` hex digits), or, for a file's path, those whose test video it
// is. Throws UsageError for a prefix shorter than `min_prefix` that is not a
// path either.
std::vector<const SavedComparison*> saved_named(const std::vector<SavedComparison>& saved, const std::string& item,
                                                std::size_t min_prefix);

// saved list's lines and JSON (docs/schemas/saved-list.schema.json): each
// comparison's ID, test video, reference (and whether it is still there),
// when it was saved, its size and its metrics, newest first.
std::string saved_list_text(const std::vector<const SavedComparison*>& saved);
nlohmann::ordered_json saved_list_json(const std::vector<const SavedComparison*>& saved);

// --segments saved (profile:NAME, preset:KEY) for saved show and report:
// the conditions taken from `settings` into `output` as compare takes them
// (take_saved_segments), with its notes said by `message`: conditions left
// out, on metrics none of the videos has (`asked`), and the condition a
// profile with none on them takes. Throws UsageError where compare exits 2
// with the message (no such preset, or a preset with no condition left).
void take_saved_segments_of_scores(CompareArguments& output, const Settings& settings,
                                   const std::vector<std::string>& asked,
                                   const std::function<void(const std::string&)>& message);

// The comparison `item` names: a run file's (.metrics.json), or the saved
// one whose ID begins with it (at least 6 hex digits). Throws
// std::runtime_error saying why there is none, or more than one.
ComparisonResult read_comparison(const std::string& item, const std::filesystem::path& folder,
                                 const Settings& settings);

// Runs the saved command; returns the exit code: 0; 1 when a file asked for
// could not be written, or a comparison could not be read or deleted; 2 for
// a wrong command; 4 when an item names no saved comparison.
// `settings`: as --settings or --no-settings choose them (the saved
// results' folder, unless --cache-dir gives one; the weak segment profiles).
int saved(const SavedArguments& arguments, const Settings& settings);

// Runs the report command: the run files as compare reports their scores.
// Returns the exit code: 0; 1 when a file could not be written; 4 when a run
// file could not be read (the others are still reported).
int report(const SavedArguments& arguments, const Settings& settings);

}  // namespace vml::cli
