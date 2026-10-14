// A video stream's packets read with ffprobe, and its bitrate graphs: per
// frame, per second and per group of pictures. A port of
// vmaf_app/core/bitrate.py.
//
// Only the first video stream is read. Audio, subtitles and the container's
// own bytes are left out on purpose: the Bitrate tab shows how the encoded
// video spends its bits, not how large the whole file is.
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "vml/cancel.h"
#include "vml/video.h"

namespace vml {

// The video's packets could not be read; the message says why.
class BitrateError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A video's average, lowest and highest bitrate, in kb/s (the lowest and
// highest per second).
struct BitrateSummary {
    double average_kbps = 0;
    double minimum_kbps = 0;
    double maximum_kbps = 0;
};

// A video stream's packets in presentation order: each one's time and
// duration in seconds, its size in bytes, and whether it is a keyframe.
// Made by bitrate_data, which works the summary out once: the table asks
// for it on every refresh, for every row.
struct BitrateData {
    std::filesystem::path path;
    std::vector<double> times;
    std::vector<double> durations;
    std::vector<long long> sizes;
    std::vector<bool> keyframes;
    BitrateSummary summary;

    long long frame_count() const { return static_cast<long long>(times.size()); }
    // The first packet's time.
    double start_time() const;
    // When the last packet ends.
    double end_time() const;
    // The stream's length.
    double duration() const;
    long long total_bytes() const;
    // The average over the whole stream.
    double average_kbps() const;
};

// The packets given, with their summary. Throws std::invalid_argument when
// the columns are not all as long.
BitrateData bitrate_data(std::filesystem::path path, std::vector<double> times, std::vector<double> durations,
                         std::vector<long long> sizes, std::vector<bool> keyframes);

// The one-second bins of a stream's bitrate: each one's start and end, and
// its kb/s. With adjust_start, the stream starts at 0; without, the bins
// fall on the file's whole seconds.
struct SecondBins {
    std::vector<double> starts;
    std::vector<double> ends;
    std::vector<float> rates;
};
SecondBins second_bins(const BitrateData& data, bool adjust_start);

// A graph line: its times and values, and the labels for its axis and its
// values.
struct BitratePlot {
    std::vector<double> times;
    std::vector<float> values;
    std::string axis_label;
    std::string value_label;
};

// Each frame's size, in kilobits.
BitratePlot frame_plot(const BitrateData& data, bool adjust_start = true);
// The bitrate in each second, in kb/s, as steps.
BitratePlot second_plot(const BitrateData& data, bool adjust_start = true);
// Each group of pictures' bitrate (keyframe to keyframe), in kb/s, as steps.
BitratePlot gop_plot(const BitrateData& data, bool adjust_start = true);

// ffprobe's lines of packets, read one at a time as they come
// (-show_packets in compact form), into the stream's packets in
// presentation order.
class PacketList {
public:
    // A packet with no duration is given one frame's at `fps` (none if 0).
    explicit PacketList(double fps);
    // Reads one line; a line that is not a packet (no whole size) is left.
    void add(std::string_view line);
    std::size_t size() const { return packets_.size(); }
    // The packets in presentation order: B-frames come in decode order.
    // Throws BitrateError when there are none.
    BitrateData finish(const std::filesystem::path& path) const;

private:
    struct Packet {
        std::optional<double> pts;
        std::optional<double> dts;
        double duration;
        long long size;
        long long position;
        bool keyframe;
    };
    double fallback_duration_;
    std::vector<Packet> packets_;
};

// The command reading a video's packets: its first video stream's, each
// one's times, size, position and flags.
std::vector<std::string> bitrate_command(const std::string& ffprobe, const std::filesystem::path& path);

// Reads the first video stream's packets with `ffprobe`, ordered by
// presentation time; `on_progress` hears the packets read so far and the
// frames expected, every 500 packets and at the end. Throws BitrateError
// when they cannot be read, Cancelled once `cancel` is.
BitrateData analyze_video_bitrate(const std::string& ffprobe, const VideoInfo& info, const CancelToken& cancel,
                                  const std::function<void(long long, long long)>& on_progress = nullptr);

}  // namespace vml
