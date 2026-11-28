// The saved results (metric_cache.h) browsed: each comparison saved under
// a folder, with its score files, read back without a request (the Saved
// results tab). A metric can have more than one saved score (CVVDP for two
// displays; SSIMULACRA2 on the GPU and the CPU): the newest is taken unless
// another is chosen.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/metric_cache.h"
#include "vml/results.h"

namespace vml {

struct Settings;

// One score file of a saved comparison: <key>_<hash>.npz.
struct SavedScoreFile {
    std::filesystem::path file;
    std::string key;
    std::chrono::system_clock::time_point saved;
    std::uintmax_t bytes = 0;
};

struct SavedComparison {
    std::filesystem::path folder;
    std::string label;
    std::filesystem::path source;
    std::filesystem::path distorted;
    std::chrono::system_clock::time_point saved;  // its newest file's
    std::uintmax_t bytes = 0;                     // its files'
    // Whether its reference is still where its scores were saved from
    // (metric_cache.h's reference_where_saved). Where not, its scores are the
    // only ones there are: they could not be calculated again.
    bool reference_found = true;
    // Whether reference_found was found out (ReferenceCheck::later leaves
    // it to references_found), and what that needs: the reference's
    // identity and sampled hash as its scores were saved (identity.json).
    bool reference_checked = true;
    std::string source_identity;
    std::string source_sampled;
    // By metric in the app's order, each metric's newest first.
    std::vector<SavedScoreFile> scores;

    // Its metrics, each once, in the app's order.
    std::vector<std::string> keys() const;
    // What the Saved Results filter (the tab's, and the command line's
    // `saved list --filter`) looks in: its label and its two files' paths.
    std::vector<std::string> filter_texts() const;
};

// Every comparison saved under `base` (its v2 folder), newest first; a
// folder without a context.json this app can read is left out. `stopped`,
// where given, is asked before each folder: once it says so, the listing
// ends with what it has (a listing nobody waits for any more, on a slow
// share, stops rather than reading every folder).
//
// Each one's reference is looked for (reference_found) before the list is
// given back, unless `references` says later: references are often on a
// share, where a first look waits on the server (seconds, its disks woken
// from sleep), and the list itself does not. Then reference_checked is
// false and references_found does it. A listing read from the folders, the
// index unusable, has always looked.
enum class ReferenceCheck { now, later };
std::vector<SavedComparison> list_saved_comparisons(const std::filesystem::path& base,
                                                    const std::function<bool()>& stopped = {},
                                                    ReferenceCheck references = ReferenceCheck::now);

// Whether each comparison's reference is still where its scores were saved
// from, in the list's order: each reference once (comparisons share them),
// several at a time (each waits on its share's server, not on the others).
// Those already checked keep what they have. Empty once `stopped` says so.
std::vector<bool> references_found(const std::vector<SavedComparison>& list,
                                   const std::function<bool()>& stopped = {});

// The comparison's result and label with one score per metric: the file
// `chosen` gives for it, else its newest. A file that cannot be read, or
// whose score the Videos tab refuses (read_saved_score's `outdated`), is
// left out (the next newest of its metric stands in). Nothing when the
// context cannot be read. Each file is read by `read`: read_saved_score, or
// what was read of it already.
std::optional<std::pair<ComparisonResult, std::string>> load_saved_comparison(
    const SavedComparison& comparison, const std::map<std::string, std::filesystem::path>& chosen = {},
    const std::function<std::optional<SavedScore>(const std::filesystem::path&)>& read = read_saved_score);

// A saved score of a metric to choose among: its file, when it was saved,
// and how it was made (none where its file can't be read, or is out of
// date: never chosen).
struct SavedScoreChoice {
    std::filesystem::path file;
    std::chrono::system_clock::time_point saved;
    std::optional<MetricProvenance> provenance;
};

// The rule that picks one (Settings::saved_score_pick, _device, _engine).
struct SavedScoreRule {
    std::string pick = "videos";  // "videos", "newest" or "newest_app"
    std::string device = "any";   // "any", "gpu" or "cpu": first, with "newest" and "newest_app"
    std::string engine = "any";   // "any", "app" or "ffmpeg": first, likewise
};

// The rule Settings > Saved Results gives.
SavedScoreRule saved_score_rule(const Settings& settings);

// The score `rule` uses among `choices` (those of one metric), or none where
// none is usable. "videos": the newest of those `videos_would_use` takes
// (the Videos tab's lookup, the window's), else as "newest". "newest": the
// newest; "newest_app": the one made by the newest version of the app
// (MetricProvenance::made_by), newest among equals; each with the device's,
// then the engine's, first.
std::optional<std::filesystem::path> pick_saved_score(
    const std::vector<SavedScoreChoice>& choices, const SavedScoreRule& rule,
    const std::function<bool(const SavedScoreChoice&)>& videos_would_use = {});

// The file of `key`'s score among `comparison`'s that `rule` picks
// (pick_saved_score): where it has several, each read by `score_of` (none:
// it can't be read) to see how it was made; with one, that one, usable or
// not; none where it has no score of `key`.
std::optional<std::filesystem::path> saved_score_by_rule(
    const SavedComparison& comparison, const std::string& key, const SavedScoreRule& rule,
    const std::function<const SavedScore*(const std::filesystem::path&)>& score_of,
    const std::function<bool(const SavedScoreChoice&)>& videos_would_use = {});

// Score files read once: a file read to choose among a metric's several
// scores is not read again to load it (readable or not).
using SavedScoreReads = std::map<std::filesystem::path, std::optional<SavedScore>>;

// Each metric's file `rule` picks (saved_score_by_rule), the files read to
// choose kept in `reads`.
std::map<std::string, std::filesystem::path> saved_scores_by_rule(const SavedComparison& comparison,
                                                                 const SavedScoreRule& rule,
                                                                 SavedScoreReads& reads);

// Deletes the comparison's folder, which must be one under `base`'s v2
// folder: whether it is gone. Whole or not at all: where one of its files
// is in use, either nothing of it is deleted (it is still listed), or it is
// moved out of the comparisons (v2/.deleted) and what was in use is deleted
// at the next delete.
bool delete_saved_comparison(const std::filesystem::path& base, const SavedComparison& comparison);

// Deletes each, as delete_saved_comparison does, the index brought up to
// date once for them all: whether each is gone, in the list's order.
std::vector<bool> delete_saved_comparisons(const std::filesystem::path& base,
                                           const std::vector<SavedComparison>& comparisons);

// Deletes the comparisons under `base` last saved (their newest file's time)
// from `from` up to, not including, `to` (none: no bound on that side), each
// as delete_saved_comparison does: every one cache_summary counts, those
// whose reference is gone and those whose context.json can't be read too
// (the question before it said so). `removed` counts those
// deleted; one that could not be deleted (a file in use) is not counted.
ClearedResults delete_saved_between(const std::filesystem::path& base,
                                    std::optional<std::chrono::system_clock::time_point> from,
                                    std::optional<std::chrono::system_clock::time_point> to);

}  // namespace vml
