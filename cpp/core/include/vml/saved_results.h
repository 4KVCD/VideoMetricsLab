// The saved results (metric_cache.h) browsed: each comparison saved under
// a folder, with its score files, read back without a request (the Saved
// results tab). A metric can have more than one saved score (CVVDP for two
// displays; SSIMULACRA2 on the GPU and the CPU): the newest is taken unless
// another is chosen.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/results.h"

namespace vml {

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
    // By metric in the app's order, each metric's newest first.
    std::vector<SavedScoreFile> scores;

    // Its metrics, each once, in the app's order.
    std::vector<std::string> keys() const;
};

// Every comparison saved under `base` (its v2 folder), newest first; a
// folder without a context.json this app can read is left out.
std::vector<SavedComparison> list_saved_comparisons(const std::filesystem::path& base);

// The comparison's result and label with one score per metric: the file
// `chosen` gives for it, else its newest. A file that cannot be read is
// left out (the next newest of its metric stands in). Nothing when the
// context cannot be read.
std::optional<std::pair<ComparisonResult, std::string>> load_saved_comparison(
    const SavedComparison& comparison, const std::map<std::string, std::filesystem::path>& chosen = {});

// Deletes the comparison's folder, which must be one under `base`'s v2
// folder: whether it is gone.
bool delete_saved_comparison(const std::filesystem::path& base, const SavedComparison& comparison);

}  // namespace vml
