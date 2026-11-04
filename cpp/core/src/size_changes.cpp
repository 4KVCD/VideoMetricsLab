#include "vml/size_changes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <system_error>

#include "vml/app_log.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.size_changes");

// The videos found to change size, by their keys (noted_key), each with
// whether it was only rebuilt at its own size.
struct Noted {
    std::mutex mutex;
    std::map<std::string, bool, std::less<>> keeps_its_size;
};

Noted& noted() {
    static Noted videos;
    return videos;
}

// A video as the noted videos know it: its path, its size and when it was
// written, so that a file written again since (an encode made again under
// its name) is not taken for the one found to change.
std::string noted_key(const std::filesystem::path& video) {
    std::error_code error;
    const std::uintmax_t bytes = std::filesystem::file_size(video, error);
    const std::optional<long long> written = platform::modified_ns(video);
    return platform::path_text(video) + "|" + (error ? "" : std::to_string(bytes)) + "|" +
           (written ? std::to_string(*written) : "");
}

// Notes that `video`'s picture size changes, for this process's runs of it.
// What is noted first stays: a run marked by the note has not looked again.
void note_size_change(const std::filesystem::path& video, bool keeps_its_size) {
    std::string key = noted_key(video);
    Noted& videos = noted();
    const std::lock_guard lock(videos.mutex);
    videos.keeps_its_size.try_emplace(std::move(key), keeps_its_size);
}

// Whether `video` was found to change size, and if so whether it was only
// rebuilt at its own size.
std::optional<bool> noted_keeps_its_size(const std::filesystem::path& video) {
    const std::string key = noted_key(video);
    Noted& videos = noted();
    const std::lock_guard lock(videos.mutex);
    const auto found = videos.keeps_its_size.find(key);
    return found != videos.keeps_its_size.end() ? std::optional(found->second) : std::nullopt;
}

constexpr std::string_view rebuild_words = "Reconfiguring filter graph";

}  // namespace

bool filters_rebuilt(std::string_view ffmpeg_line) {
    return ffmpeg_line.find(rebuild_words) != std::string_view::npos;
}

std::optional<std::pair<int, int>> rebuilt_size(std::string_view ffmpeg_output) {
    // "... changed to yuv420p(tv, smpte170m), 640x360, unspecified alpha"
    const std::size_t at = ffmpeg_output.find(rebuild_words);
    if (at == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string line(ffmpeg_output.substr(at, ffmpeg_output.find('\n', at) - at));
    static const std::regex size_words(R"(, (\d{1,6})x(\d{1,6})(,|\s|$))");
    std::smatch found;
    if (!std::regex_search(line, found, size_words)) {
        return std::nullopt;
    }
    return std::pair(std::stoi(found[1].str()), std::stoi(found[2].str()));
}

void FilterRebuild::note(std::string_view ffmpeg_line) {
    if (rebuilt_ || !filters_rebuilt(ffmpeg_line)) {
        return;
    }
    if (const auto size = rebuilt_size(ffmpeg_line)) {
        width_ = size->first;
        height_ = size->second;
    }
    rebuilt_ = true;  // last: whoever sees it sees the size
}

std::optional<std::pair<int, int>> FilterRebuild::size() const {
    return rebuilt_ && width_ > 0 ? std::optional(std::pair<int, int>(width_, height_)) : std::nullopt;
}

std::function<void(std::string_view)> rebuild_watcher(std::shared_ptr<FilterRebuild> rebuild) {
    return [rebuild = std::move(rebuild)](std::string_view line) { rebuild->note(line); };
}

std::string error_lines_of(std::string_view error_output) {
    // FFmpeg's levels, each written after a line's context ("[in#0 @ 0x...]
    // [error] ..."); the first that a line has is its own.
    static constexpr std::array<std::string_view, 9> levels{"[quiet] ", "[panic] ",   "[fatal] ",
                                                            "[error] ", "[warning] ", "[info] ",
                                                            "[verbose] ", "[debug] ", "[trace] "};
    static constexpr std::size_t shown = 4;  // "-loglevel error" shows the levels up to "[error] "
    std::string kept;
    // A line without a level goes on the message before it (a metadata
    // value's later lines in FFmpeg's description of its input), or, before
    // any, is not FFmpeg's log at all, and may say why it failed.
    bool keeping = true;
    while (!error_output.empty()) {
        const std::size_t end = error_output.find('\n');
        const std::string_view line = error_output.substr(0, end == std::string_view::npos ? end : end + 1);
        error_output.remove_prefix(line.size());
        std::size_t at = std::string_view::npos;
        std::size_t level = 0;
        for (std::size_t index = 0; index < levels.size(); ++index) {
            if (const std::size_t found = line.find(levels[index]); found < at) {
                at = found;
                level = index;
            }
        }
        if (at == std::string_view::npos) {
            if (keeping) {
                kept += line;
            }
            continue;
        }
        keeping = level < shown;
        if (keeping) {
            kept.append(line.substr(0, at)).append(line.substr(at + levels[level].size()));
        }
    }
    return kept;
}

std::pair<int, int> size_change_target(const VideoInfo& test, const VideoInfo& source) {
    const int width = source.frame_size().first;
    if (test.width <= 0 || test.height <= 0) {
        return {width, source.frame_size().second};
    }
    const double height = static_cast<double>(width) * test.height / test.width;
    return {width, std::max(2, 2 * static_cast<int>(std::lround(height / 2)))};
}

void mark_size_changes(VideoInfo& source, VideoInfo& test) {
    if (!source.scaled_to && noted_keeps_its_size(source.path)) {
        source.scaled_to = std::pair(source.width, source.height);
    }
    if (!test.scaled_to) {
        if (const std::optional<bool> keeps_its_size = noted_keeps_its_size(test.path)) {
            test.scaled_to = *keeps_its_size ? std::pair(test.width, test.height) : size_change_target(test, source);
        }
    }
}

SizeChangeMarks::SizeChangeMarks(VideoInfo& source, VideoInfo& test) : source_(source), test_(test) {
    mark_size_changes(source_, test_);
}

bool SizeChangeMarks::after(const VideoSizeChanged& changed, const std::string& scale_algorithm) {
    if (const std::optional<std::filesystem::path>& video = changed.video()) {
        VideoInfo* rebuilt = *video == test_.path ? &test_ : *video == source_.path ? &source_ : nullptr;
        if (rebuilt != nullptr) {
            if (rebuilt->scaled_to) {
                return false;  // its FFmpeg rebuilt them though it was told to keep them
            }
            mark(*rebuilt, changed.new_size(), scale_algorithm);
            if (rebuilt == &test_) {
                test_guessed_ = false;
            }
            return true;
        }
    }
    if (!test_.scaled_to) {
        mark(test_, changed.new_size(), scale_algorithm);
        test_guessed_ = !source_.scaled_to;  // with the source marked, only the test is left to rebuild
        return true;
    }
    if (!source_.scaled_to) {
        // With the test video marked, it was the source.
        mark(source_, changed.new_size(), scale_algorithm);
        if (test_guessed_) {
            test_.scaled_to.reset();  // tried again: it is marked once it rebuilds with the source marked
            test_guessed_ = false;
            test_keeps_its_size_ = false;
            logger.info(platform::path_text(test_.path.filename()) + ": scored at its own size again; it is " +
                        platform::path_text(source_.path.filename()) + " whose size changes");
        }
        return true;
    }
    return false;
}

void SizeChangeMarks::finished() const {
    if (source_.scaled_to) {
        note_size_change(source_.path, false);  // as a test video later, it is scaled to its reference
    }
    if (test_.scaled_to) {
        note_size_change(test_.path, test_keeps_its_size_);
    }
}

void SizeChangeMarks::mark(VideoInfo& video, std::optional<std::pair<int, int>> new_size,
                           const std::string& scale_algorithm) {
    const std::pair own_size(video.width, video.height);
    const bool keeps_its_size = new_size == own_size;
    if (&video == &test_) {
        test_keeps_its_size_ = keeps_its_size;
    }
    video.scaled_to = &video == &test_ && !keeps_its_size ? size_change_target(test_, source_) : own_size;
    const std::string scored_at = "every frame is scored at " + std::to_string(video.scaled_to->first) + "x" +
                                  std::to_string(video.scaled_to->second);
    logger.info(platform::path_text(video.path.filename()) +
                (keeps_its_size ? ": FFmpeg rebuilt its filters partway through at the video's own size; " + scored_at
                                : ": its picture size changes partway through; " + scored_at + ", scaled with " +
                                      scale_algorithm));
}

}  // namespace vml
