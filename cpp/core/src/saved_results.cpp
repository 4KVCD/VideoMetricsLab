#include "vml/saved_results.h"

#include <algorithm>
#include <random>
#include <system_error>
#include <variant>

#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/settings.h"

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

// A YUV metric's Y plane is saved in its plain metric's file (metric_path):
// listed under the Y's key too, from that file, where the folder has the
// YUV metric's planes and no file of the Y's own (one saved before). The
// file's bytes are counted once, under the plain metric.
void list_luma_planes(std::vector<SavedScoreFile>& scores) {
    const auto has = [&scores](const std::string& key) {
        return std::any_of(scores.begin(), scores.end(), [&key](const SavedScoreFile& score) { return score.key == key; });
    };
    std::vector<SavedScoreFile> planes;
    for (const SavedScoreFile& chroma : scores) {
        if (!chroma.key.ends_with("_u")) {
            continue;
        }
        const std::string luma = chroma.key.substr(0, chroma.key.size() - 1) + "y";
        const std::string_view metric = saved_as(luma);
        if (metric == luma || has(luma)) {
            continue;
        }
        for (const SavedScoreFile& score : scores) {
            if (score.key == metric) {
                planes.push_back({score.file, luma, score.saved, 0});
            }
        }
    }
    scores.insert(scores.end(), planes.begin(), planes.end());
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
        list_luma_planes(saved.scores);
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
                // Under the key listed: a YUV metric's Y is read from its plain metric's file (list_luma_planes).
                std::visit(
                    [&loaded, &key](auto&& metric) {
                        metric.key = key;
                        loaded->first.metric_results.add(std::move(metric));
                    },
                    std::move(score->metric));
                break;
            }
        }
    }
    return loaded;
}

namespace {

// "2.6 (0123456789ab)" as numbers, {2, 6}: compared part by part; none known: empty, oldest.
std::vector<int> app_version_of(const std::string& made_by) {
    std::vector<int> parts;
    int number = -1;
    for (const char c : made_by) {
        if (c >= '0' && c <= '9') {
            number = (number < 0 ? 0 : number * 10) + (c - '0');
        } else {
            if (number >= 0) {
                parts.push_back(number);
                number = -1;
            }
            if (c != '.') {
                break;  // the version ends at the first that is neither a digit nor a dot
            }
        }
    }
    if (number >= 0) {
        parts.push_back(number);
    }
    return parts;
}

}  // namespace

SavedScoreRule saved_score_rule(const Settings& settings) {
    return {settings.saved_score_pick, settings.saved_score_device, settings.saved_score_engine};
}

std::optional<std::filesystem::path> pick_saved_score(const std::vector<SavedScoreChoice>& choices,
                                                      const SavedScoreRule& rule,
                                                      const std::function<bool(const SavedScoreChoice&)>& videos_would_use) {
    std::vector<const SavedScoreChoice*> usable;
    for (const SavedScoreChoice& choice : choices) {
        if (choice.provenance) {
            usable.push_back(&choice);
        }
    }
    if (usable.empty()) {
        return std::nullopt;
    }
    // Newest first, whatever order they came in.
    std::stable_sort(usable.begin(), usable.end(),
                     [](const SavedScoreChoice* a, const SavedScoreChoice* b) { return a->saved > b->saved; });
    if (rule.pick == "videos" && videos_would_use) {
        for (const SavedScoreChoice* choice : usable) {
            if (videos_would_use(*choice)) {
                return choice->file;
            }
        }
        // None the Videos tab would take: the newest.
    }
    const auto device_rank = [&rule](const MetricProvenance& made) {
        return rule.device == "any" || made.compute_backend == rule.device ? 0 : 1;
    };
    const auto engine_rank = [&rule](const MetricProvenance& made) {
        if (rule.engine == "any") {
            return 0;
        }
        const bool ffmpeg = made.implementation.rfind("ffmpeg", 0) == 0;
        return (rule.engine == "ffmpeg") == ffmpeg ? 0 : 1;
    };
    const bool by_version = rule.pick == "newest_app";
    const auto better = [&](const SavedScoreChoice* a, const SavedScoreChoice* b) {
        if (rule.pick != "videos") {
            if (device_rank(*a->provenance) != device_rank(*b->provenance)) {
                return device_rank(*a->provenance) < device_rank(*b->provenance);
            }
            if (engine_rank(*a->provenance) != engine_rank(*b->provenance)) {
                return engine_rank(*a->provenance) < engine_rank(*b->provenance);
            }
        }
        if (by_version) {
            const std::vector<int> x = app_version_of(a->provenance->made_by);
            const std::vector<int> y = app_version_of(b->provenance->made_by);
            if (x != y) {
                return x > y;
            }
        }
        return false;  // as they are: newest first
    };
    std::stable_sort(usable.begin(), usable.end(), better);
    return usable.front()->file;
}

std::optional<std::filesystem::path> saved_score_by_rule(
    const SavedComparison& comparison, const std::string& key, const SavedScoreRule& rule,
    const std::function<const SavedScore*(const std::filesystem::path&)>& score_of,
    const std::function<bool(const SavedScoreChoice&)>& videos_would_use) {
    std::vector<SavedScoreChoice> choices;
    for (const SavedScoreFile& file : comparison.scores) {
        if (file.key == key) {
            choices.push_back({file.file, file.saved, std::nullopt});
        }
    }
    if (choices.size() < 2) {  // nothing to choose; usable or not, it's the one there is
        return choices.empty() ? std::nullopt : std::optional(choices.front().file);
    }
    for (SavedScoreChoice& choice : choices) {
        if (const SavedScore* score = score_of(choice.file); score != nullptr && !score->outdated) {
            choice.provenance = std::visit([](const auto& metric) { return metric.provenance; }, score->metric);
        }
    }
    return pick_saved_score(choices, rule, videos_would_use).value_or(choices.front().file);
}

std::map<std::string, std::filesystem::path> saved_scores_by_rule(const SavedComparison& comparison,
                                                                 const SavedScoreRule& rule,
                                                                 SavedScoreReads& reads) {
    const auto score_of = [&reads](const std::filesystem::path& file) -> const SavedScore* {
        const auto [kept, first_read] = reads.try_emplace(file);
        if (first_read) {
            kept->second = read_saved_score(file);
        }
        return kept->second ? &*kept->second : nullptr;
    };
    std::map<std::string, std::filesystem::path> chosen;
    for (const std::string& key : comparison.keys()) {
        if (const std::optional<std::filesystem::path> file = saved_score_by_rule(comparison, key, rule, score_of)) {
            chosen[key] = *file;
        }
    }
    return chosen;
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
    // Only a folder of v2's own, or none yet: were it a link (a junction or
    // symlink someone made), clearing it would empty the folder it points to,
    // and the comparison would be moved there.
    std::error_code ignored;
    const std::filesystem::file_type aside_type = std::filesystem::symlink_status(aside, ignored).type();
    if (aside_type != std::filesystem::file_type::directory && aside_type != std::filesystem::file_type::not_found) {
        return false;
    }
    ignored.clear();
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

ClearedResults delete_saved_between(const std::filesystem::path& base,
                                    std::optional<std::chrono::system_clock::time_point> from,
                                    std::optional<std::chrono::system_clock::time_point> to) {
    // The comparisons Settings counted and said would go (cache_summary): every
    // folder with a context.json, one this app can't read among them, which
    // list_saved_comparisons leaves out.
    ClearedResults done;
    for (const CacheSummary::Comparison& comparison : cache_summary(base).saved) {
        if ((from && comparison.saved < *from) || (to && comparison.saved >= *to)) {
            continue;
        }
        SavedComparison folder;
        folder.folder = comparison.folder;
        if (delete_saved_comparison(base, folder)) {
            ++done.removed;
        }
    }
    return done;
}

}  // namespace vml
