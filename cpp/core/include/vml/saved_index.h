// The index of the saved results (metric_cache.h): one SQLite database
// beside the comparisons' folders, <base>/saved_results.sqlite, that knows
// each comparison (its label, files, when it was saved, its size) and each
// of its score files, so that what lists them (the Saved Results tab,
// Settings, `vml-cli saved`) reads one file rather than every folder. With
// 20,000 comparisons, the folders took 1.7 s to list on a RAM drive and
// about 4.5 s on an SSD at each visit; the index, a few tens of ms.
//
// The folders stay what the scores are: the index is made from them, and
// can be made again from them at any time (rebuild). Each listing checks it
// against the folder of comparisons in one read of that folder's names: a
// comparison saved or deleted by another program (an older version, a
// folder copied or deleted by hand) is read or dropped then. What this app
// and vml-cli change, they write to the index at once (changed), and a
// change cut short by a crash is read again at the next listing
// (changing). A change another program makes inside a comparison's folder
// is not seen until the index is made again (rebuild: Settings > Saved
// Results).
//
// Several processes may use it at once (the window and vml-cli): each waits
// for the other's writes. On a network drive, where SQLite's write-ahead log
// cannot be used, it keeps a rollback journal instead.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vml {

// A comparison as the index has it.
struct IndexedComparison {
    std::string name;      // its folder's name, under <base>/v2
    bool readable = true;  // a context.json this app reads (else listed by Settings only)
    std::string label;
    std::string source;     // the reference's path, as context.json says it
    std::string distorted;  // the test video's
    // What its identity.json recorded of the reference: the file's identity
    // (file_identity) and sampled hash; empty where it has none.
    std::string source_identity;
    std::string source_sampled;
    std::chrono::system_clock::time_point saved;  // its newest file's time
    std::uintmax_t bytes = 0;                     // its files'
    struct Score {
        std::string file;  // its file's name in the folder
        std::string key;
        std::chrono::system_clock::time_point saved;
        std::uintmax_t bytes = 0;
    };
    std::vector<Score> scores;  // its .npz files, in no order
};

namespace saved_index {

// The index's file for the saved results in `base`.
std::filesystem::path file(const std::filesystem::path& base);

// Every comparison saved under `base` (a folder of v2 with a context.json),
// newest first, from the index after it is checked against the folders (the
// first time, every folder is read into it: as long as a listing took
// before). Nothing where the index cannot be used (the database cannot be
// opened or written: a read-only folder, a newer version's database):
// callers then read the folders themselves. `stopped`, asked between
// folders read, ends a first reading early (what was read is kept: the
// next listing goes on from there) and returns nothing.
std::optional<std::vector<IndexedComparison>> comparisons(const std::filesystem::path& base,
                                                          const std::function<bool()>& stopped = {});

// About to change the comparison in `folder` (<base>/v2/<name>): should the
// process end before changed() (a crash), the next listing reads it again.
void changing(const std::filesystem::path& folder);

// The comparison in `folder` changed (saved, scores added or deleted, its
// files recorded) or is gone: read again into the index now.
void changed(const std::filesystem::path& folder);
// The same for several folders of one base, in one transaction (a Delete of
// many: one each was 25 ms a comparison on a VM runner's disk, most of it
// opening and committing).
void changed(const std::vector<std::filesystem::path>& folders);

// Reads every folder under `base` into the index again, as a first listing
// does: Settings' Rebuild. How many comparisons it holds then, or nothing
// where the index cannot be used (or `stopped` said so).
std::optional<int> rebuild(const std::filesystem::path& base, const std::function<bool()>& stopped = {});

}  // namespace saved_index

}  // namespace vml
