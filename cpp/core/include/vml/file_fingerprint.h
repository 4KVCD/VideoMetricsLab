// Recognising a video file by what it holds rather than where it is, for
// the saved scores (metric_cache.h): a file renamed, moved or copied, and a
// reference remuxed into another container, keep their saved scores.
//
// Three fingerprints, cheapest first:
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
// - stream_fingerprint: the first video stream's coded pictures, byte for
//   byte, whatever holds them: a raw stream (.264, .265, .266, .obu), MKV
//   or MP4. Only the units that carry pictures are hashed (H.264, HEVC and
//   VVC's VCL NAL units, in Annex B form; AV1's frame, frame header and
//   tile group OBUs), so that the parameter sets, SEI and temporal
//   delimiters a container moves, rewrites or drops change nothing. Any
//   re-encode, of any part, makes another. It reads the whole stream; its
//   first packets alone make a cheap first check.
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

// The stream fingerprint of `file`, whose first video stream is coded with
// `codec` (VideoInfo::codec_name), read by `ffmpeg`: of its first `packets`
// packets, or of all of them for 0. Nothing when it cannot be read or is
// cancelled, and for raw pictures (rawvideo: a raw YUV file, whose scores
// are not saved). A codec other than H.264, HEVC, VVC and AV1 has its
// packets hashed as they are: the same in every container that keeps them
// so.
std::optional<std::string> stream_fingerprint(const std::filesystem::path& file, const std::string& codec,
                                              const std::string& ffmpeg, int packets = 0,
                                              const CancelToken& cancel = {});

// A stream fingerprinter for the saved scores (set_stream_fingerprinter):
// each file probed by `ffprobe` and read with `ffmpeg`; nothing for one that
// cannot be read.
std::function<std::optional<std::string>(const std::filesystem::path&, int, const CancelToken&)>
stream_fingerprinter_using(std::string ffmpeg, std::string ffprobe);

// A fingerprinter for the saved scores (set_video_fingerprinter): each file
// probed by `ffprobe` and fingerprinted with `ffmpeg`; nothing for one that
// cannot be read.
std::function<std::optional<std::string>(const std::filesystem::path&)> fingerprinter_using(std::string ffmpeg,
                                                                                         std::string ffprobe);

}  // namespace vml
