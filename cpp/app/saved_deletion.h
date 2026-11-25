// The one question before saved results are deleted, the Saved Results
// tab's Delete... and Settings > Saved Results' Delete... alike
// (docs/ARCHITECTURE.md, "Removing and deleting"): a danger message saying
// what goes (how many comparisons, their size and dates, the folder), what
// that costs, and how many of them can't be scored again; Cancel Enter's
// answer and Esc's.
#pragma once

#include <QString>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include "app_dialog.h"
#include "vml/saved_results.h"

namespace vml::app {

struct SavedDeletion {
    int comparisons = 0;
    std::uintmax_t bytes = 0;
    // The first and last of their saved times; none with none.
    std::optional<std::chrono::system_clock::time_point> first;
    std::optional<std::chrono::system_clock::time_point> last;
    int without_reference = 0;  // whose reference video is gone: not to be scored again
    // Every saved result in the folder: the bitrate readings and the black
    // bars' answers kept with them go too.
    bool everything = false;
    std::filesystem::path folder;
};

// What deleting `comparisons` would delete, from `folder`.
SavedDeletion saved_deletion(const std::vector<SavedComparison>& comparisons, const std::filesystem::path& folder);

// The question, its first button deleting.
Message saved_deletion_message(const SavedDeletion& deletion);

// A size as Saved Results says it ("512 KB", "212.4 MB"), and a date in the
// system's short format.
QString saved_size_text(std::uintmax_t bytes);
QString saved_date_text(std::chrono::system_clock::time_point when);

}  // namespace vml::app
