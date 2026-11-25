#include "vml/saved_results.h"

#include <algorithm>
#include <random>
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

std::vector<std::string> SavedComparison::filter_texts() const {
    return {label, platform::path_text(distorted), platform::path_text(source)};
}

std::vector<SavedComparison> list_saved_comparisons(const std::filesystem::path& base,
                                                    const std::function<bool()>& stopped) {
    std::vector<SavedComparison> list;
    std::error_code error;
    std::filesystem::directory_iterator folder(base / "v2", error);
    for (; !error && folder != std::filesystem::directory_iterator(); folder.increment(error)) {
        if (stopped && stopped()) {
            break;
        }
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
    // Moved aside first, whole or not at all, then deleted: deleted where it
    // is, file by file, a file open without leave to delete it (a details
    // reader, a lookup, an antivirus) stopped it half-way, its context.json
    // already gone, and what was left was listed by nothing and deletable by
    // nothing. Moved, it is no comparison of v2's any more (none is listed
    // two folders down); whatever of it cannot be deleted now is at the next
    // delete.
    //
    // Each delete moves its folder under a name of its own: a leftover of the
    // same comparison (one of its files still open then) does not block it,
    // and a delete from vml-cli and one from the app at once do not take
    // each other's .deleted folder away (#221's review). Only what is in it
    // is cleared, never the folder itself, which another delete may be
    // moving into.
    const std::filesystem::path aside = v2 / ".deleted";
    std::error_code ignored;
    std::filesystem::directory_iterator left(aside, ignored);
    for (; !ignored && left != std::filesystem::directory_iterator(); left.increment(ignored)) {
        std::error_code still_open;
        std::filesystem::remove_all(left->path(), still_open);  // what deletes before left
    }
    ignored.clear();
    std::filesystem::create_directories(aside, error);
    std::random_device random;
    const std::filesystem::path moved =
        aside / platform::path_from_text(platform::path_text(folder.filename()) + "." + std::to_string(random()));
    if (!error) {
        std::filesystem::rename(folder, moved, error);
    }
    if (error) {
        return false;  // refused (one of its files in use): nothing of it deleted
    }
    saved_comparison_deleted(folder);  // gone from v2: the kept listing reads it again
    std::filesystem::remove_all(moved, ignored);
    return true;
}

}  // namespace vml
