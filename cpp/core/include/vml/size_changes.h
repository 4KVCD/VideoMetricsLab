// Test videos whose picture size changes partway through (an encoder's
// adaptive resolution: segments of another size, each starting with a
// keyframe and new parameters). The whole video is scored at the source's
// size: every frame of such a video is scaled first to the source's width,
// at the video's own shape (size_change_target), with the run's scaler
// (VideoInfo::scaled_to, geometry.h's size_change_scale), so that a 1080p
// stretch of a 4K encode is judged as a 4K player shows it, and a stream
// that starts small is still scored at the source's size.
//
// Nothing tells such a video apart before it is decoded. FFmpeg tells it as
// it goes: it rebuilds its filters for the new size ("Reconfiguring filter
// graph"), which restarts a metric's filter or loses the run's scaler. A run
// that sees it stops (VideoSizeChanged) and is made again with the video
// marked (SizeChangeMarks), and the videos marked are remembered for this
// process's later runs of them (as long as the file is not written again).
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "vml/video.h"

namespace vml {

// FFmpeg rebuilt its filters partway through a run: a video's picture size
// (or format) changed. `video` is the one it rebuilt them for, where the
// FFmpeg decoded only it; nothing where one FFmpeg decoded both.
class VideoSizeChanged : public std::runtime_error {
public:
    explicit VideoSizeChanged(std::optional<std::filesystem::path> video = std::nullopt)
        : VideoSizeChanged("FFmpeg rebuilt its filters partway through the video", std::move(video)) {}
    VideoSizeChanged(const std::string& message, std::optional<std::filesystem::path> video)
        : std::runtime_error(message), video_(std::move(video)) {}

    const std::optional<std::filesystem::path>& video() const { return video_; }

private:
    std::optional<std::filesystem::path> video_;
};

// Whether a line of FFmpeg's error output says that it rebuilt its filters.
bool filters_rebuilt(std::string_view ffmpeg_line);

// A ProcessOptions::on_error_line for an FFmpeg watched for filters_rebuilt:
// sets `rebuilt` when FFmpeg says it (shared, as the line is read on a
// thread of its own).
std::function<void(std::string_view)> rebuild_watcher(std::shared_ptr<std::atomic<bool>> rebuilt);

// The -loglevel of an FFmpeg watched for filters_rebuilt where only its
// errors are wanted: FFmpeg says it rebuilt at the info level, and with each
// line's level written, error_lines_of keeps what "-loglevel error" shows.
inline constexpr const char* watched_log_level = "level+info";

// The lines of an FFmpeg's error output, written at watched_log_level, that
// "-loglevel error" writes too (errors and worse, and the lines that go on
// one of their messages), without their levels.
std::string error_lines_of(std::string_view error_output);

// The size a test video's frames are scaled to: the source's frame width,
// and the height that keeps the test video's own shape (its probed size), even,
// so that a cropped encode of a letterboxed source is not stretched.
std::pair<int, int> size_change_target(const VideoInfo& test, const VideoInfo& source);

// The pair as this process found it: a video noted by an earlier run that
// finished (SizeChangeMarks::finished) is scaled_to its target, the test
// video to size_change_target, the source to its own probed size.
void mark_size_changes(VideoInfo& source, VideoInfo& test);

// The marks a pair's runs are made with (run_vmaf, run_perceptual_task):
// those this process found first (mark_size_changes), then those its runs
// find. A run that stops names the video FFmpeg rebuilt its filters for
// where it knows it (one FFmpeg for each video), and that one is marked.
// Where one FFmpeg decoded both, the test video is tried first, then the
// source alone, then both: a marked video's FFmpeg keeps its filters, so
// FFmpeg rebuilding them again tells the guess was wrong.
class SizeChangeMarks {
public:
    SizeChangeMarks(VideoInfo& source, VideoInfo& test);

    // After a run stopped (`changed`): marks the pair for the next run, and
    // logs each video marked, its frames scaled with `scale_algorithm`. False
    // when nothing is left to try.
    bool after(const VideoSizeChanged& changed, const std::string& scale_algorithm);

    // After a run that finished: notes its marked videos for this process's
    // later runs of them.
    void finished() const;

private:
    void mark(VideoInfo& video, const std::string& scale_algorithm);

    VideoInfo& source_;
    VideoInfo& test_;
    bool test_guessed_ = false;  // the test video is marked on a guess (one FFmpeg decoded both)
};

// Why a pair cannot be scored when FFmpeg rebuilt its filters with both
// videos marked.
inline constexpr const char* size_change_unfixed =
    "FFmpeg rebuilt its filters partway through the videos even with each scaled to one size; they cannot be "
    "compared";

}  // namespace vml
