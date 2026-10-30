// A video's pictures decoded by the app and fed to an FFmpeg as a rawvideo
// input through a pipe: how FFmpeg's routes (its libvmaf and xpsnr filters,
// the resolution test, the PNG pairs) score a video whose film grain only the
// app's own decoder adds as it should (VVC's H.274, by VVdeC:
// film_grain_scoring.h's only_the_app_adds). Every other input, filter and
// output of the command is as for a video FFmpeg decodes itself.
//
// The decoder runs in a process of its own (isolated.h, task "grain-feed"),
// as the app's other decoding for scores does: one that crashes ends that
// process and fails the run, not the window. It makes the pipe, says so,
// decodes the video from its start and writes each picture; FFmpeg is
// started once the pipe is there.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vml/cancel.h"
#include "vml/hardware_decoding.h"
#include "vml/video.h"

namespace vml {

// Whether FFmpeg gets `info`'s pictures from the app (its film grain only the
// app adds as it is scored), rather than decoding the file itself. With
// VML_FEED=all, every video the app's software decoder takes, for proving
// that a fed video scores as the file does.
bool fed_to_ffmpeg(const VideoInfo& info);

// Why `info`'s pictures cannot be fed (a size or format the app's software
// decoder does not take, its decoder missing, a video scaled on the way,
// frames paired by timestamp, another system than Windows), or nothing when
// they can. FFmpeg's routes refuse the metric with it (FilmGrainRefusal).
std::optional<std::string> feed_refusal(const VideoInfo& info);

// The input options of a fed video's pictures, before "-i" and the pipe's
// path: rawvideo of its size, format and frame rate, with its colour tags,
// as FFmpeg's decoding of the file would give them. Nothing when a tag has no
// name the rawvideo decoder takes.
std::optional<std::vector<std::string>> feed_input_options(const VideoInfo& info);

// One feeding of a video's pictures to one FFmpeg (a pipe has one reader:
// each FFmpeg run, retries too, has its own). Made before the FFmpeg is
// started, and destroyed after it has ended: a feeder blocked on a pipe whose
// reader never came is let go then.
class PictureFeed {
public:
    // Starts the feeding process and waits for its pipe. Throws
    // FilmGrainRefusal with feed_refusal's reason, Cancelled, and what the
    // feeding process threw before its pipe was there.
    PictureFeed(const VideoInfo& info, const std::string& ffmpeg, const std::string& ffprobe,
                const CancelToken& cancel);
    ~PictureFeed();
    PictureFeed(const PictureFeed&) = delete;
    PictureFeed& operator=(const PictureFeed&) = delete;

    // The FFmpeg arguments of the fed input: feed_input_options, "-i" and
    // the pipe's path (given to FFmpeg as it is: opening it to look at it
    // would be its one reader).
    const std::vector<std::string>& input_arguments() const;

    // Once the FFmpeg has ended: waits for the feeding process. Throws
    // FilmGrainRefusal when the video could not be decoded to its end (the
    // app's decoder is the only one for its pictures, so no other route is
    // tried, and FFmpeg, finding its input's end, would have scored part of
    // the video as the whole), and Cancelled. FFmpeg stopping to read first
    // (a duration limit, or its own failure, its caller's to report) is no
    // error of the feeding.
    void finish();

private:
    struct State;
    std::unique_ptr<State> state_;
};

// The feeding of one FFmpeg run's two videos (either may be nothing), each
// one fed_to_ffmpeg says: made before the FFmpeg and kept until it has
// ended.
class PairFeeds {
public:
    PairFeeds(const VideoInfo* source, const VideoInfo* distorted, const std::string& ffmpeg,
              const std::string& ffprobe, const CancelToken& cancel);

    // Each video's feed, or nothing where FFmpeg decodes the file.
    const PictureFeed* source() const { return source_ ? &*source_ : nullptr; }
    const PictureFeed* distorted() const { return distorted_ ? &*distorted_ : nullptr; }

    // `decoding` for the run: no GPU decoding for a fed video (FFmpeg reads
    // its pictures as they come).
    HardwareDecoding decoding(HardwareDecoding decoding) const;

    // Once the FFmpeg has ended, before anything it wrote is read: each
    // feed's finish().
    void finish();

private:
    std::optional<PictureFeed> distorted_;
    std::optional<PictureFeed> source_;
};

// The feeding process's side (isolated.cpp's task "grain-feed"): makes the
// pipe, calls `on_ready`, then decodes and writes. Its result: the pictures
// written, and whether the reader closed its end first.
nlohmann::json picture_feed_task(const nlohmann::json& request, const std::function<void()>& on_ready);

}  // namespace vml
