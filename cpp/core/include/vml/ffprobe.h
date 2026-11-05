// Reading a video's details with ffprobe. A port of vmaf_app/core/ffprobe.py.
#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

#include "vml/cancel.h"
#include "vml/video.h"

namespace vml {

// ffprobe could not read a file; the message says why, as the Python app's
// does.
class ProbeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// How every ffprobe the app starts begins. ffprobe decodes a few pictures to fill
// in what a file does not say, and FFmpeg 9.0's own VVC decoder crashes adding
// film grain with chroma to a 4:0:0 picture (cpp/tests/media/README.md). Asked to
// export the grain, it adds none, and what ffprobe reports is otherwise the same.
std::vector<std::string> ffprobe_command_start(const std::string& ffprobe);

// Whether ffprobe's listing of frames (frame=width,height, csv=p=0) has one of
// another size than this.
bool listed_sizes_differ(std::string_view listed, int width, int height);

// The video's details, read by running `ffprobe` (the program to run) on
// `path`. Throws ProbeError when they cannot be read.
VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe);
// The same, ended by `cancel`: ffprobe on a large file over a slow or
// network disk can take many seconds, and closing the window or stopping a
// scan must not wait for it. Throws Cancelled once cancelled.
VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe, const CancelToken& cancel);

// The frame rate a raw stream of `codec_name` stores in its headers, given
// what ffprobe reports of it (r_frame_rate) when told to read the stream at a
// rate too high to be one: see frame_rate_in_stream. Nothing where it stores none.
std::optional<double> stored_frame_rate(std::string_view codec_name, double reported);

// The frame rate stored in the headers of `info`, a raw stream's details
// (VideoInfo::is_raw_stream); nothing where it stores none, or its codec's
// rate has not been shown to be the stream's own. Never FFmpeg's guess at 25.
// Throws Cancelled once cancelled.
std::optional<double> frame_rate_in_stream(const VideoInfo& info, const std::string& ffprobe,
                                           const CancelToken& cancel);

// `info`, a raw stream's details, with what reading it finds: its frames
// counted, and timed at the rate in its headers, or with no rate at all
// (VideoInfo::frame_rate_unknown). Throws ProbeError when its frames cannot be
// counted, Cancelled once cancelled.
VideoInfo raw_stream_as_found(VideoInfo info, const std::string& ffprobe, const CancelToken& cancel);

// `info`, a raw stream's details, timed at `fps` frames a second, the rate the
// command line's --fps gives or frame_rate_in_stream finds. Its frames are counted with `ffprobe` (without
// decoding them), and its length and bitrate follow from them. Throws
// ProbeError when they cannot be counted, Cancelled once cancelled.
VideoInfo timed_raw_stream(VideoInfo info, double fps, const std::string& ffprobe,
                           const CancelToken& cancel = CancelToken());

// The frame rate someone typed: a fraction of whole numbers below 2^53, as many
// as a double holds exactly ("24000/1001"), or a number ("23.976"). Nothing
// where it is neither; what it comes to is not checked.
std::optional<double> typed_frame_rate(const std::string& text);
// The rates accepted from someone typing one: outside them, a mistyped value,
// which FFmpeg would take or fail on oddly.
inline constexpr double lowest_typed_frame_rate = 0.001;
inline constexpr double highest_typed_frame_rate = 10000;

// A raw YUV file's details (a .yuv file: pictures and nothing else, FFmpeg's
// rawvideo), read as pictures of `size` ("1920x1080") in `pix_fmt` at `fps`
// frames a second, the command line's --size, --pix-fmt and --fps, and
// timed as timed_raw_stream times a raw stream. Throws ProbeError when
// ffprobe cannot read it so, Cancelled once cancelled.
// The frame count a file states: its stream's, or Matroska's NUMBER_OF_FRAMES tag
// (which a stream copy may have left stale). Nothing where it states none.
std::optional<long long> stated_frame_count(const std::filesystem::path& path, const std::string& ffprobe,
                                            const CancelToken& cancel);

// The packets of the file's first video stream, read through without decoding.
long long count_video_packets(const std::filesystem::path& path, const std::string& ffprobe,
                              const CancelToken& cancel);

VideoInfo probe_raw_yuv(const std::filesystem::path& path, const std::string& size, const std::string& pix_fmt,
                        double fps, const std::string& ffprobe, const CancelToken& cancel = CancelToken());

// The video's details in ffprobe's JSON output for `path`
// (-print_format json -show_format -show_streams). Throws ProbeError when
// the output cannot be read or has no video stream.
VideoInfo video_info_from_probe(const std::filesystem::path& path, std::string_view probe_json);

}  // namespace vml
