#include "vml/size_changes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <system_error>

#include "vml/app_log.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.size_changes");

// The videos found to change size, by their keys (noted_key).
struct Noted {
    std::mutex mutex;
    std::set<std::string, std::less<>> keys;
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
void note_size_change(const std::filesystem::path& video) {
    std::string key = noted_key(video);
    Noted& videos = noted();
    const std::lock_guard lock(videos.mutex);
    videos.keys.insert(std::move(key));
}

// Whether `video`'s picture size was found to change.
bool size_changes(const std::filesystem::path& video) {
    const std::string key = noted_key(video);
    Noted& videos = noted();
    const std::lock_guard lock(videos.mutex);
    return videos.keys.contains(key);
}

}  // namespace

bool filters_rebuilt(std::string_view ffmpeg_line) {
    return ffmpeg_line.find("Reconfiguring filter graph") != std::string_view::npos;
}

std::function<void(std::string_view)> rebuild_watcher(std::shared_ptr<std::atomic<bool>> rebuilt) {
    return [rebuilt = std::move(rebuilt)](std::string_view line) {
        if (filters_rebuilt(line)) {
            *rebuilt = true;
        }
    };
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
    if (!source.scaled_to && size_changes(source.path)) {
        source.scaled_to = std::pair(source.width, source.height);
    }
    if (!test.scaled_to && size_changes(test.path)) {
        test.scaled_to = size_change_target(test, source);
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
            mark(*rebuilt, scale_algorithm);
            if (rebuilt == &test_) {
                test_guessed_ = false;
            }
            return true;
        }
    }
    if (!test_.scaled_to) {
        mark(test_, scale_algorithm);
        test_guessed_ = !source_.scaled_to;  // with the source marked, only the test is left to rebuild
        return true;
    }
    if (!source_.scaled_to) {
        // With the test video marked, it was the source.
        mark(source_, scale_algorithm);
        if (test_guessed_) {
            test_.scaled_to.reset();  // tried again: it is marked once it rebuilds with the source marked
            test_guessed_ = false;
            logger.info(platform::path_text(test_.path.filename()) + ": scored at its own size again; it is " +
                        platform::path_text(source_.path.filename()) + " whose size changes");
        }
        return true;
    }
    return false;
}

void SizeChangeMarks::finished() const {
    for (const VideoInfo* video : {&source_, &test_}) {
        if (video->scaled_to) {
            note_size_change(video->path);
        }
    }
}

void SizeChangeMarks::mark(VideoInfo& video, const std::string& scale_algorithm) {
    video.scaled_to = &video == &test_ ? size_change_target(test_, source_) : std::pair(video.width, video.height);
    logger.info(platform::path_text(video.path.filename()) +
                ": its picture size changes partway through; every frame is scored at " +
                std::to_string(video.scaled_to->first) + "x" + std::to_string(video.scaled_to->second) +
                ", scaled with " + scale_algorithm);
}

}  // namespace vml
