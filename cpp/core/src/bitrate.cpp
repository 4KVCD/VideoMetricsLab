#include "vml/bitrate.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>

#include "vml/ffmpeg_tools.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// One line of ffprobe's compact output: the value of `key`, the last one
// given, or nothing when the line has none.
std::optional<std::string_view> field(std::string_view line, std::string_view key) {
    std::optional<std::string_view> found;
    line = python_strip(line);
    std::size_t start = 0;
    while (start <= line.size()) {
        const std::size_t end = std::min(line.find('|', start), line.size());
        const std::string_view item = line.substr(start, end - start);
        const std::size_t equals = item.find('=');
        if (equals != std::string_view::npos && item.substr(0, equals) == key) {
            found = item.substr(equals + 1);
        }
        start = end + 1;
    }
    return found;
}

// A number as ffprobe printed it, or nothing for none ("N/A") or one that is
// not finite.
std::optional<double> number(std::optional<std::string_view> value) {
    if (!value || value->empty() || *value == "N/A") {
        return std::nullopt;
    }
    const std::optional<double> parsed = python_float(*value);
    return parsed && std::isfinite(*parsed) ? parsed : std::nullopt;
}

// Turns (start, end, value) spans into the points of a stepped line.
void step_points(const std::vector<double>& starts, const std::vector<double>& ends, const std::vector<float>& values,
                 BitratePlot& plot) {
    plot.times.reserve(values.size() * 2);
    plot.values.reserve(values.size() * 2);
    for (std::size_t index = 0; index < values.size(); ++index) {
        plot.times.push_back(starts[index]);
        plot.times.push_back(ends[index]);
        plot.values.push_back(values[index]);
        plot.values.push_back(values[index]);
    }
}

}  // namespace

// ------------------------------------------------------------------ the data

double BitrateData::start_time() const { return times.empty() ? 0.0 : times.front(); }

double BitrateData::end_time() const {
    double end = 0.0;
    for (std::size_t index = 0; index < times.size(); ++index) {
        const double packet_end = times[index] + durations[index];
        end = index == 0 ? packet_end : std::max(end, packet_end);
    }
    return end;
}

double BitrateData::duration() const { return std::max(0.0, end_time() - start_time()); }

long long BitrateData::total_bytes() const { return std::accumulate(sizes.begin(), sizes.end(), 0LL); }

double BitrateData::average_kbps() const {
    const double length = duration();
    return length > 0 ? static_cast<double>(total_bytes() * 8) / length / 1000 : 0.0;
}

BitrateData bitrate_data(std::filesystem::path path, std::vector<double> times, std::vector<double> durations,
                         std::vector<long long> sizes, std::vector<bool> keyframes) {
    if (durations.size() != times.size() || sizes.size() != times.size() || keyframes.size() != times.size()) {
        throw std::invalid_argument("Bitrate packet columns must have the same length");
    }
    BitrateData data{std::move(path), std::move(times), std::move(durations), std::move(sizes), std::move(keyframes),
                     {}};
    const std::vector<float> rates = second_bins(data, true).rates;
    data.summary.average_kbps = data.average_kbps();
    if (!rates.empty()) {
        data.summary.minimum_kbps = *std::min_element(rates.begin(), rates.end());
        data.summary.maximum_kbps = *std::max_element(rates.begin(), rates.end());
    }
    return data;
}

// ------------------------------------------------------------------ the graphs

SecondBins second_bins(const BitrateData& data, bool adjust_start) {
    SecondBins bins;
    if (data.times.empty()) {
        return bins;
    }
    const double offset = adjust_start ? data.start_time() : std::floor(data.start_time());
    const double relative_start = data.start_time() - offset;
    const double relative_end = data.end_time() - offset;
    const auto first_bin = static_cast<long long>(std::floor(relative_start));
    const auto last_bin = static_cast<long long>(std::ceil(relative_end));
    for (long long bin = first_bin; bin < last_bin; ++bin) {
        bins.starts.push_back(static_cast<double>(bin));
        bins.ends.push_back(static_cast<double>(bin) + 1.0);
    }
    // Packets that all start and end at one whole second have no bin: the
    // Python app fails there (an index into nothing), and no ffprobe output
    // seen gives such packets.
    if (bins.starts.empty()) {
        return bins;
    }
    const auto last_index = static_cast<long long>(bins.starts.size()) - 1;
    std::vector<double> byte_counts(bins.starts.size(), 0.0);
    for (std::size_t packet = 0; packet < data.times.size(); ++packet) {
        const double start = data.times[packet] - offset;
        const double duration = data.durations[packet];
        const auto size = static_cast<double>(data.sizes[packet]);
        const double end = start + duration;
        if (duration <= 0) {
            const long long index =
                std::min(last_index, std::max(0LL, static_cast<long long>(std::floor(start)) - first_bin));
            byte_counts[static_cast<std::size_t>(index)] += size;
            continue;
        }
        // Shared between the seconds it spans, in proportion.
        const long long begin_index = std::max(0LL, static_cast<long long>(std::floor(start)) - first_bin);
        const long long end_index = std::min(last_index, static_cast<long long>(std::ceil(end)) - 1 - first_bin);
        for (long long index = begin_index; index <= end_index; ++index) {
            const auto at = static_cast<std::size_t>(index);
            const double overlap = std::max(0.0, std::min(end, bins.ends[at]) - std::max(start, bins.starts[at]));
            byte_counts[at] += size * overlap / duration;
        }
    }
    // The first and last seconds count only the part the stream covers.
    bins.rates.reserve(byte_counts.size());
    for (std::size_t index = 0; index < byte_counts.size(); ++index) {
        const double coverage = std::max(
            1e-12, std::min(bins.ends[index], relative_end) - std::max(bins.starts[index], relative_start));
        bins.rates.push_back(static_cast<float>(byte_counts[index] * 8 / coverage / 1000));
    }
    if (!adjust_start) {
        // The file's own timeline on the axis.
        for (std::size_t index = 0; index < bins.starts.size(); ++index) {
            bins.starts[index] += offset;
            bins.ends[index] += offset;
        }
    }
    return bins;
}

BitratePlot frame_plot(const BitrateData& data, bool adjust_start) {
    const double offset = adjust_start ? data.start_time() : 0.0;
    BitratePlot plot{{}, {}, "Frame size (kbit)", "kbit"};
    plot.times.reserve(data.times.size());
    plot.values.reserve(data.sizes.size());
    for (std::size_t index = 0; index < data.times.size(); ++index) {
        plot.times.push_back(data.times[index] - offset);
        plot.values.push_back(static_cast<float>(static_cast<double>(data.sizes[index]) * 8 / 1000));
    }
    return plot;
}

BitratePlot second_plot(const BitrateData& data, bool adjust_start) {
    const SecondBins bins = second_bins(data, adjust_start);
    BitratePlot plot{{}, {}, "Video bitrate (kb/s)", "kb/s"};
    step_points(bins.starts, bins.ends, bins.rates, plot);
    return plot;
}

BitratePlot gop_plot(const BitrateData& data, bool adjust_start) {
    BitratePlot plot{{}, {}, "GOP bitrate (kb/s)", "kb/s"};
    if (data.times.empty()) {
        return plot;
    }
    // A group starts at the first packet and at every later keyframe.
    std::vector<std::size_t> group_starts{0};
    for (std::size_t index = 1; index < data.keyframes.size(); ++index) {
        if (data.keyframes[index]) {
            group_starts.push_back(index);
        }
    }
    const double offset = adjust_start ? data.start_time() : 0.0;
    std::vector<double> starts;
    std::vector<double> ends;
    std::vector<float> rates;
    for (std::size_t group = 0; group < group_starts.size(); ++group) {
        const std::size_t begin = group_starts[group];
        const std::size_t finish = group + 1 < group_starts.size() ? group_starts[group + 1] : data.times.size();
        const double start = data.times[begin];
        double end = data.times[begin] + data.durations[begin];
        long long size = 0;
        for (std::size_t index = begin; index < finish; ++index) {
            end = std::max(end, data.times[index] + data.durations[index]);
            size += data.sizes[index];
        }
        const double duration = end - start;
        starts.push_back(start - offset);
        ends.push_back(end - offset);
        rates.push_back(static_cast<float>(duration > 0 ? static_cast<double>(size * 8) / duration / 1000 : 0.0));
    }
    step_points(starts, ends, rates, plot);
    return plot;
}

// ------------------------------------------------------------- the packets

PacketList::PacketList(double fps) : fallback_duration_(fps > 0 ? 1.0 / fps : 0.0) {}

void PacketList::add(std::string_view line) {
    const std::optional<long long> size = python_int(field(line, "size").value_or(""));
    if (!size) {
        return;
    }
    // A duration of nothing, or of 0, is one frame's.
    const std::optional<double> given = number(field(line, "duration_time"));
    const double duration = given && *given != 0.0 ? *given : fallback_duration_;
    const auto count = static_cast<long long>(packets_.size());
    const std::optional<std::string_view> pos = field(line, "pos");
    const long long position = pos ? python_int(*pos).value_or(count) : count;
    const std::string_view flags = field(line, "flags").value_or("");
    packets_.push_back({number(field(line, "pts_time")), number(field(line, "dts_time")), std::max(0.0, duration),
                        *size, position, flags.find('K') != std::string_view::npos});
}

BitrateData PacketList::finish(const std::filesystem::path& path) const {
    if (packets_.empty()) {
        throw BitrateError("ffprobe returned no packets for the first video stream.");
    }
    struct Resolved {
        double time;
        double duration;
        long long size;
        long long position;
        bool keyframe;
    };
    // A packet without a presentation time takes its decode time, and one
    // with neither follows the packets before it.
    std::vector<Resolved> resolved;
    resolved.reserve(packets_.size());
    double next_time = 0.0;
    for (const Packet& packet : packets_) {
        const double time = packet.pts ? *packet.pts : packet.dts ? *packet.dts : next_time;
        resolved.push_back({time, packet.duration, packet.size, packet.position, packet.keyframe});
        next_time = std::max(next_time, time + packet.duration);
    }
    // Packets come in decode order. The graphs and the groups of pictures go
    // in presentation order, so B-frames are put back in place.
    std::stable_sort(resolved.begin(), resolved.end(), [](const Resolved& a, const Resolved& b) {
        return a.time != b.time ? a.time < b.time : a.position < b.position;
    });
    std::vector<double> times;
    std::vector<double> durations;
    std::vector<long long> sizes;
    std::vector<bool> keyframes;
    for (const Resolved& packet : resolved) {
        times.push_back(packet.time);
        durations.push_back(packet.duration);
        sizes.push_back(packet.size);
        keyframes.push_back(packet.keyframe);
    }
    return bitrate_data(path, std::move(times), std::move(durations), std::move(sizes), std::move(keyframes));
}

// ------------------------------------------------------------- reading a video

std::vector<std::string> bitrate_command(const std::string& ffprobe, const std::filesystem::path& path) {
    return {ffprobe,       "-v",          "error", "-select_streams",
            std::string(video_stream), "-show_packets", "-show_entries",
            "packet=pts_time,dts_time,duration_time,size,pos,flags", "-of", "compact=p=0:nk=0",
            platform::path_text(path)};
}

BitrateData analyze_video_bitrate(const std::string& ffprobe, const VideoInfo& info, const CancelToken& cancel,
                                  const std::function<void(long long, long long)>& on_progress) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(info.path, error)) {
        throw BitrateError("Video file is missing: " + platform::path_text(info.path));
    }
    cancel.throw_if_cancelled();
    // ffprobe's complaints, read as they come, the last 200 lines kept: a
    // damaged file can report one for every packet.
    struct Complaints {
        std::mutex mutex;
        std::deque<std::string> lines;
    };
    const auto complaints = std::make_shared<Complaints>();
    platform::ProcessOptions options;
    options.on_error_line = [complaints](std::string_view line) {
        const std::lock_guard lock(complaints->mutex);
        complaints->lines.push_back(std::string(line) + "\n");
        if (complaints->lines.size() > 200) {
            complaints->lines.pop_front();
        }
    };
    std::optional<platform::ChildProcess> process;
    try {
        process.emplace(platform::ChildProcess::start(bitrate_command(ffprobe, info.path), options));
    } catch (const platform::ProcessError& failure) {
        throw BitrateError(std::string("Could not start ffprobe: ") + failure.what());
    }
    PacketList packets(info.fps);
    const long long total = std::max(1LL, info.estimated_frame_count());
    int exit_code = 0;
    {
        const CancelToken::Attached attached = cancel.attach(*process);
        while (const std::optional<std::string> line = process->read_line()) {
            const std::size_t before = packets.size();
            packets.add(*line);
            if (on_progress && packets.size() != before && packets.size() % 500 == 0) {
                on_progress(static_cast<long long>(packets.size()), total);
            }
        }
        exit_code = process->wait();
    }
    if (exit_code != 0) {
        cancel.throw_if_cancelled();
        std::string said;
        {
            const std::lock_guard lock(complaints->mutex);
            for (const std::string& line : complaints->lines) {
                said += line;
            }
        }
        const std::string detail(python_strip(replace_invalid_utf8(said)));
        throw BitrateError(detail.empty() ? "ffprobe exited with code " + std::to_string(exit_code) + "."
                                          : last_characters(detail, 2000));
    }
    BitrateData data = packets.finish(info.path);
    if (on_progress) {
        on_progress(data.frame_count(), total);
    }
    return data;
}

}  // namespace vml
