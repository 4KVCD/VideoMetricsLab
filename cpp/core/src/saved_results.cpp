#include "vml/saved_results.h"

#include <algorithm>
#include <system_error>
#include <variant>

#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

// A file's time on the system clock: file_time_type's clock is not the
// system's everywhere, so its age is carried over.
std::chrono::system_clock::time_point system_time(std::filesystem::file_time_type time) {
    const auto age = std::filesystem::file_time_type::clock::now() - time;
    return std::chrono::system_clock::now() -
           std::chrono::duration_cast<std::chrono::system_clock::duration>(age);
}

// The metric's place in the app's order (unknown metrics last).
std::size_t order_of(const std::string& key) {
    const auto& all = metrics();
    const auto found = std::find_if(all.begin(), all.end(), [&key](const MetricDefinition& d) { return d.key == key; });
    return static_cast<std::size_t>(found - all.begin());
}

}  // namespace

std::vector<std::string> SavedComparison::keys() const {
    std::vector<std::string> found;
    for (const SavedScoreFile& score : scores) {
        if (std::find(found.begin(), found.end(), score.key) == found.end()) {
            found.push_back(score.key);
        }
    }
    return found;
}

std::vector<SavedComparison> list_saved_comparisons(const std::filesystem::path& base) {
    std::vector<SavedComparison> list;
    std::error_code error;
    std::filesystem::directory_iterator folder(base / "v2", error);
    for (; !error && folder != std::filesystem::directory_iterator(); folder.increment(error)) {
        std::error_code ignored;
        if (!folder->is_directory(ignored)) {
            continue;
        }
        const std::optional<std::pair<ComparisonResult, std::string>> context = saved_comparison(folder->path());
        if (!context) {
            continue;
        }
        SavedComparison saved;
        saved.folder = folder->path();
        saved.label = context->second;
        saved.source = context->first.source;
        saved.distorted = context->first.distorted;
        saved.reference_found = reference_where_saved(folder->path());
        std::filesystem::directory_iterator file(folder->path(), ignored);
        for (; !ignored && file != std::filesystem::directory_iterator(); file.increment(ignored)) {
            std::error_code unread;
            const std::uintmax_t bytes = file->file_size(unread);
            saved.bytes += unread ? 0 : bytes;
            const auto when = system_time(file->last_write_time(unread));
            saved.saved = std::max(saved.saved, when);
            const std::string name = platform::path_text(file->path().filename());
            const std::size_t hash = name.rfind('_');
            if (name.ends_with(".npz") && hash != std::string::npos && hash > 0) {
                saved.scores.push_back({file->path(), name.substr(0, hash), when, bytes});
            }
        }
        std::stable_sort(saved.scores.begin(), saved.scores.end(),
                         [](const SavedScoreFile& a, const SavedScoreFile& b) {
                             const std::size_t x = order_of(a.key);
                             const std::size_t y = order_of(b.key);
                             return x != y ? x < y : a.key != b.key ? a.key < b.key : a.saved > b.saved;
                         });
        list.push_back(std::move(saved));
    }
    std::stable_sort(list.begin(), list.end(),
                     [](const SavedComparison& a, const SavedComparison& b) { return a.saved > b.saved; });
    return list;
}

std::optional<std::pair<ComparisonResult, std::string>> load_saved_comparison(
    const SavedComparison& comparison, const std::map<std::string, std::filesystem::path>& chosen,
    const std::function<std::optional<SavedScore>(const std::filesystem::path&)>& read) {
    std::optional<std::pair<ComparisonResult, std::string>> loaded = saved_comparison(comparison.folder);
    if (!loaded) {
        return std::nullopt;
    }
    for (const std::string& key : comparison.keys()) {
        // The chosen file first, then the metric's others, newest first.
        std::vector<std::filesystem::path> files;
        if (const auto pick = chosen.find(key); pick != chosen.end()) {
            files.push_back(pick->second);
        }
        for (const SavedScoreFile& score : comparison.scores) {
            if (score.key == key && std::find(files.begin(), files.end(), score.file) == files.end()) {
                files.push_back(score.file);
            }
        }
        for (const std::filesystem::path& file : files) {
            if (auto score = read(file); score && !score->outdated) {
                std::visit([&loaded](auto&& metric) { loaded->first.metric_results.add(std::move(metric)); },
                           std::move(score->metric));
                break;
            }
        }
    }
    return loaded;
}

bool delete_saved_comparison(const std::filesystem::path& base, const SavedComparison& comparison) {
    // Only a comparison's own folder, directly under v2: never more.
    std::error_code error;
    const std::filesystem::path v2 = std::filesystem::weakly_canonical(base / "v2", error);
    const std::filesystem::path folder = std::filesystem::weakly_canonical(comparison.folder, error);
    if (error || folder.parent_path() != v2 || !std::filesystem::exists(folder / "context.json", error)) {
        return false;
    }
    std::filesystem::remove_all(folder, error);
    return !error && !std::filesystem::exists(folder, error);
}

}  // namespace vml
