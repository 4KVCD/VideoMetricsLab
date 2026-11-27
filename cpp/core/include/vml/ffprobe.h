// Reading a video's details with ffprobe. A port of vmaf_app/core/ffprobe.py.
#pragma once

#include <filesystem>
#include <map>
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

// Input options for an ffprobe or FFmpeg that lists or copies packets: the
// file opened with no stream analysis, the decoding of its first frames for
// details the packets do not need. The same packets, 0.25 to 0.35 s sooner
// for a 1080p H.264 Matroska file. Never for the details (probe_video):
// without the analysis Matroska files lose their start times.
std::vector<std::string> no_stream_analysis();

// Whether ffprobe's listing of frames (frame=width,height, csv=p=0) has one of
// another size than this.
bool listed_sizes_differ(std::string_view listed, int width, int height);

// ffprobe's timestamps (csv=p=0, one a line), in the order listed; nothing
// when a line has none (N/A) or no number.
std::optional<std::vector<long long>> listed_timestamps(std::string_view listed);

// The earliest of the times in ffprobe's listing of packets (packet=pts_time,
// csv=p=0, one a line): with a stream's first packets listed, the time its
// first shown picture has. Lines with none (N/A) are passed over; nothing
// when no line has one.
std::optional<double> earliest_listed_time(std::string_view listed);

// Whether a file's packets, as stored, look stamped in the order they are
// decoded rather than shown: in a video whose pictures are reordered, stamps
// that only ever rise. A file stamped right stores a reordered picture's
// packet before its earlier-shown neighbours. Too few packets say nothing.
bool stamps_rise_as_stored(const std::vector<long long>& packet_stamps);

// Whether a video's decoded pictures, which come out in the order they are
// shown, carry stamps that go back: the stamps are not the pictures' times.
bool stamps_go_back_as_shown(const std::vector<long long>& picture_stamps);

// The video's details, read by running `ffprobe` (the program to run) on
// `path`. Throws ProbeError when they cannot be read, and for a video whose
// timestamps are in decoding order (a raw stream copied into MP4 or Matroska
// with none of its own): no decoder can place its pictures.
VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe);
// The same, ended by `cancel`: ffprobe on a large file over a slow or
// network disk can take many seconds, and closing the window or stopping a
// scan must not wait for it. Throws Cancelled once cancelled.
//
// Where the video states no rate of its own, a short stretch of the file's
// packets is read for the other streams' and the container's share of the
// file's rate (0.1 to 0.6 s): with RateSample::later it is not, and the rate
// is the file's, or the file's less the audio's stated rates (marked as an
// estimate either way), until sampled_video_rate gives it.
enum class RateSample { now, later };
// What sampled_video_rate takes of a video's reading with RateSample::later:
// ffprobe's report of its format and streams and its Matroska writing app,
// so that the sample reads only the packets, not the file's streams again.
struct ProbeForRateSample {
    std::string probe;
    std::optional<std::string> writer;
};
// `for_sample`: where given, filled for sampled_video_rate when the video
// needs_rate_sample (RateSample::later).
VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe, const CancelToken& cancel,
                      RateSample sample = RateSample::now, ProbeForRateSample* for_sample = nullptr);

// Whether a video read with RateSample::later has a better rate to be had
// from sampled_video_rate: its own rate is not stated, and it is not a raw
// stream (whose rate is the file's, exact).
bool needs_rate_sample(const VideoInfo& info);

// The video's details with the rate RateSample::now gives (only the rate is
// meant to be taken from them: VideoInfo::take_bit_rate), or nothing where the
// packets cannot be read and the rate stays as it was. `read`: from the
// video's probe_video. Throws Cancelled once cancelled.
std::optional<VideoInfo> sampled_video_rate(const std::filesystem::path& path, const ProbeForRateSample& read,
                                            const std::string& ffprobe, const CancelToken& cancel);

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

// The frames a decoder delivers from a file, from the count it states, where
// that can be trusted: its stream's own (an MP4's samples), or Matroska's
// NUMBER_OF_FRAMES tag unless FFmpeg wrote the file (its encoder tag is
// "Lavf..."; FFmpeg copies a stream's tags as they were: an encode it made
// of a Blu-ray remux states the remux's count). Less the pictures among its
// first 64 packets that are not shown (count_shown_video_packets), which a
// count of stored pictures includes. Nothing where it states none.
std::optional<long long> trusted_frame_count(const std::filesystem::path& path, const std::string& ffprobe,
                                             const CancelToken& cancel);

// The packets of the file's first video stream that are shown, read through
// without decoding: those with a presentation time and not flagged to be
// discarded. A stream cut by copying can begin with pictures that refer to
// ones before the cut (an open GOP's leading pictures), which Matroska gives
// no time and MP4 the discard flag, and which no decoder delivers; they are
// left out, so the count is of the frames a decoder delivers.
long long count_shown_video_packets(const std::filesystem::path& path, const std::string& ffprobe,
                                    const CancelToken& cancel);

VideoInfo probe_raw_yuv(const std::filesystem::path& path, const std::string& size, const std::string& pix_fmt,
                        double fps, const std::string& ffprobe, const CancelToken& cancel = CancelToken());

// What the packets of the first `seconds` of a file hold: each stream's
// bytes, by its index in the file (ffprobe's "index"), and the share of the
// bytes they span that are the container's own (Matroska's cluster and block
// headers, MPEG-TS's packet headers, tables and padding), 0 to 0.5.
struct PacketSample {
    std::map<int, long long> bytes;
    double overhead = 0.0;
    double seconds = 0.0;  // the stretch read (the file's length where it is shorter)
};

// How many seconds of a file of `bits_per_second` (0 if unknown) a
// PacketSample reads: about 16 MB of it, at least 5 s and at most 2 minutes,
// so that a high-rate file is not read for long.
int packet_sample_seconds(long long bits_per_second);

// The video's details in ffprobe's JSON output for `path`
// (-print_format json -show_format -show_streams). Where the video's own
// bitrate is not recorded, it is estimated from the file's: with `sample`,
// less its attachments (fonts), the container's own bytes and every other
// stream (audio, subtitles, data), each at its stated rate or else its rate
// in the sample; without, less its audio tracks' stated rates. `writing_app`
// is the program that wrote the file (matroska_writing_app), where known: its
// Matroska statistics are checked only where that is FFmpeg. Throws
// ProbeError when the output cannot be read or has no video stream.
VideoInfo video_info_from_probe(const std::filesystem::path& path, std::string_view probe_json,
                                const std::optional<PacketSample>& sample = std::nullopt,
                                const std::optional<std::string>& writing_app = std::nullopt);

// The sample in ffprobe's lines of packets (-show_entries
// packet=stream_index,size,pos -of csv=p=0) of the first `seconds` of a file.
PacketSample packet_sample_from_csv(std::string_view packets, double seconds);

}  // namespace vml
