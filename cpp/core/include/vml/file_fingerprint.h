// Recognising a video file by what it holds rather than where it is, for
// the saved scores (metric_cache.h): a file renamed, moved or copied, and a
// reference remuxed into another container, keep their saved scores.
//
// Two fingerprints, cheapest first:
// - sampled_file_hash: the file's size and bytes from its start, its end
//   and evenly between (a few megabytes read, whatever the file's size).
//   The same file wherever it is; any change to the file (a remux, a tag
//   edit in its header) makes another.
// - video_fingerprint: the first video stream as the app reads and decodes
//   it, whatever container holds it: its codec, picture size, pixel
//   format, frame rate, colour as read and length in whole seconds, and a
//   digest of five decoded pictures spread through it. A remux of the same
//   stream into MKV or MP4, with any other tracks, gives the same one.
//   About a second or two: it decodes a few pictures.
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vml/cancel.h"
#include "vml/video.h"

namespace vml {

// The sampled file hash, or nothing when the file cannot be read.
std::optional<std::string> sampled_file_hash(const std::filesystem::path& file);

// The times (seconds from the stream's start) of the pictures
// video_fingerprint decodes: five, at a sixth of the length apart (the
// length in whole seconds, so that a container's rounding moves none), each
// in the middle of its picture's time, so that a container's millisecond
// rounding of timestamps lands on the same picture.
std::vector<double> fingerprint_times(double duration, double fps);

// The video fingerprint of `file`, read with `info` (probe_video's) and
// decoded by `ffmpeg`; nothing when it cannot be decoded or is cancelled.
std::optional<std::string> video_fingerprint(const std::filesystem::path& file, const VideoInfo& info,
                                             const std::string& ffmpeg, const CancelToken& cancel = {});

// A fingerprinter for the saved scores (set_video_fingerprinter): each file
// probed by `ffprobe` and fingerprinted with `ffmpeg`; nothing for one that
// cannot be read.
std::function<std::optional<std::string>(const std::filesystem::path&)> fingerprinter_using(std::string ffmpeg,
                                                                                         std::string ffprobe);

}  // namespace vml
