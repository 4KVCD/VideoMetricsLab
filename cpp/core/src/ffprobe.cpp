#include "vml/ffprobe.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "vml/append.h"
#include "vml/av2_probe.h"
#include "vml/matroska.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

// Keys in the order ffprobe wrote them, as Python's dicts keep them: a tag
// is found by the first key that matches.
using Json = nlohmann::ordered_json;

std::string text(const std::filesystem::path& path) { return platform::path_text(path); }

// Python's truthiness of a value read from JSON: missing, null, false, 0
// and "" are false.
bool truthy(const Json* value) {
    if (value == nullptr || value->is_null()) {
        return false;
    }
    if (value->is_string()) {
        return !value->get_ref<const std::string&>().empty();
    }
    if (value->is_boolean()) {
        return value->get<bool>();
    }
    if (value->is_number()) {
        return value->get<double>() != 0.0;
    }
    return !value->empty();
}

const Json* member(const Json& object, const char* key) {
    if (!object.is_object()) {
        return nullptr;
    }
    const auto found = object.find(key);
    return found == object.end() ? nullptr : &*found;
}

// A string member, or `fallback` when it is missing (Python's dict.get).
std::string string_member(const Json& object, const char* key, const std::string& fallback = "") {
    const Json* value = member(object, key);
    return value != nullptr && value->is_string() ? value->get<std::string>() : fallback;
}

// The first of these members that is truthy, as Python's "a or b or c".
std::string first_truthy(const Json& object, std::initializer_list<const char*> keys, const std::string& fallback) {
    for (const char* key : keys) {
        const Json* value = member(object, key);
        if (truthy(value) && value->is_string()) {
            return value->get<std::string>();
        }
    }
    return fallback;
}

// Python's float(value) of a value read from JSON, or nothing where it
// would raise.
std::optional<double> float_of(const Json* value) {
    if (value == nullptr) {
        return std::nullopt;
    }
    if (value->is_string()) {
        return python_float(value->get_ref<const std::string&>());
    }
    if (value->is_boolean()) {
        return value->get<bool>() ? 1.0 : 0.0;
    }
    if (value->is_number()) {
        return value->get<double>();
    }
    return std::nullopt;
}

// A number read as a whole number, or 0 (_whole: int(float(value))).
long long whole(const Json* value) {
    const std::optional<double> number = float_of(value);
    if (!number || !std::isfinite(*number)) {
        return 0;
    }
    return static_cast<long long>(std::trunc(*number));
}

// A rate as ffprobe prints it ("24000/1001", "25"); 0 for one it could not
// tell ("N/A", "0/0").
double parse_frame_rate(const std::string& rate) {
    const std::size_t slash = rate.find('/');
    if (slash != std::string::npos) {
        const std::optional<double> numerator = python_float(std::string_view(rate).substr(0, slash));
        const std::optional<double> denominator = python_float(std::string_view(rate).substr(slash + 1));
        if (!numerator || !denominator) {
            return 0.0;
        }
        return *denominator != 0.0 ? *numerator / *denominator : 0.0;
    }
    return python_float(rate).value_or(0.0);
}

// Whether a stream is cover art or a thumbnail: a picture or a few, which
// FFmpeg lists as a video stream (in Matroska, an attachment), not a stream
// with a rate.
bool is_picture(const Json& stream) {
    const Json* disposition = member(stream, "disposition");
    return disposition != nullptr &&
           (truthy(member(*disposition, "attached_pic")) || truthy(member(*disposition, "timed_thumbnails")));
}

// The first video stream that is a video, not cover art or a thumbnail
// (the same stream FFmpeg's commands read: "V:0").
const Json* video_stream(const Json& streams) {
    if (!streams.is_array()) {
        return nullptr;
    }
    for (const Json& stream : streams) {
        if (string_member(stream, "codec_type") == "video" && !is_picture(stream)) {
            return &stream;
        }
    }
    return nullptr;
}

// A stream tag, by its name in any case (Matroska writers differ), or else
// by its name with a language: a Matroska tag given a language reaches
// ffprobe as NAME-lang, as the statistics tags of a track with one (BPS-eng,
// DURATION-eng, NUMBER_OF_FRAMES-eng: mkvmerge before v40, 2019).
const Json* tag(const Json& stream, std::string_view name) {
    const Json* tags = member(stream, "tags");
    if (tags == nullptr || !tags->is_object()) {
        return nullptr;
    }
    const Json* with_language = nullptr;
    for (auto item = tags->begin(); item != tags->end(); ++item) {
        const std::string key = ascii_upper(item.key());
        if (key == name) {
            return &item.value();
        }
        // NAME-xx or NAME-xxx (ISO 639), or NAME-xx-YY (a BCP 47 tag)
        if (with_language == nullptr && key.size() > name.size() + 1 && key.starts_with(name) &&
            key[name.size()] == '-') {
            const std::string_view language = std::string_view(key).substr(name.size() + 1);
            if (language.size() <= 8 && std::all_of(language.begin(), language.end(), [](char c) {
                    return (c >= 'A' && c <= 'Z') || c == '-';
                })) {
                with_language = &item.value();
            }
        }
    }
    return with_language;
}

// A tag's text, by its name in any case; empty where it is none or not text.
std::string tag_text(const Json& tagged, std::string_view name) {
    const Json* value = tag(tagged, name);
    return value != nullptr && value->is_string() ? value->get<std::string>() : std::string();
}

// "01:45:36.289000000" in seconds, or nothing.
std::optional<double> clock_seconds(const Json* value) {
    if (value == nullptr || !value->is_string()) {
        return std::nullopt;
    }
    const std::string& text = value->get_ref<const std::string&>();
    const std::size_t first = text.find(':');
    const std::size_t second = first == std::string::npos ? first : text.find(':', first + 1);
    if (second == std::string::npos || text.find(':', second + 1) != std::string::npos) {
        return std::nullopt;
    }
    const std::optional<long long> hours = python_int(std::string_view(text).substr(0, first));
    const std::optional<long long> minutes = python_int(std::string_view(text).substr(first + 1, second - first - 1));
    const std::optional<double> seconds = python_float(std::string_view(text).substr(second + 1));
    if (!hours || !minutes || !seconds) {
        return std::nullopt;
    }
    return static_cast<double>(*hours * 3600 + *minutes * 60) + *seconds;
}

// The stream's own length. A Matroska file gives no stream duration, and the
// container's is its longest stream's (an audio track running on after the
// picture). Matroska writers record each track's length as a DURATION tag;
// one longer than the container is stale (left by a remux that cut the
// file) and is not used.
double stream_duration(const Json& stream, double container) {
    const Json* duration = member(stream, "duration");
    if (truthy(duration)) {
        if (const std::optional<double> seconds = float_of(duration)) {
            return *seconds;
        }
    }
    const std::optional<double> tagged = clock_seconds(tag(stream, "DURATION"));
    if (tagged && *tagged > 0 && (container <= 0 || *tagged <= container + 0.5)) {
        return *tagged;
    }
    return container;
}

// One stream's own bitrate, from what the container records: its bit_rate,
// or Matroska's statistics tags (BPS, or NUMBER_OF_BYTES over its length).
// 0 when there is none. The statistics are used only if they are the file's
// (see statistics_are_true) and fit in it.
long long stream_bit_rate(const Json& stream, double duration, long long file_size, bool statistics_are_true) {
    const long long rate = whole(member(stream, "bit_rate"));
    if (rate > 0) {
        return rate;
    }
    if (!statistics_are_true) {
        return 0;
    }
    const long long size = whole(tag(stream, "NUMBER_OF_BYTES"));
    long long tagged = whole(tag(stream, "BPS"));
    if (tagged == 0) {
        tagged = size > 0 && duration > 0
                     ? static_cast<long long>(std::nearbyint(static_cast<double>(size * 8) / duration))
                     : 0;
    }
    if (tagged <= 0 || file_size <= 0 || size > file_size ||
        static_cast<double>(tagged) * duration / 8 > static_cast<double>(file_size) * 1.02) {
        return 0;
    }
    return tagged;
}

// Whether a file's statistics tags describe it. FFmpeg writes none of its own,
// and copies its input's into what it writes, even through a re-encode, where
// they can be far from the truth. Who wrote the file is its Matroska
// WritingApp (`writing_app`, matroska_writing_app), not its ENCODER tag:
// mkvmerge keeps its input's, so a file mkvmerge wrote, with statistics of its
// own, from one FFmpeg made says "Lavf" there. Without the WritingApp (a file
// not read, or not Matroska), the ENCODER tag says.
//
// In a file FFmpeg wrote, the tags are used only where they add up: every
// stream but the attachments and cover art has its size, the streams but the
// subtitles are the file but for its container's own bytes (at least 99.5 %
// of it: a 2-hour Matroska film's are 0.3 %, and no more than all of it), and
// the video is at least half of it. A stale set passes only where the new
// video is at most 0.5 % of the file larger than the old: its rate is then at
// most about 1 % off. The subtitles are left out because their sizes are their
// packets' as read: a subtitle track may be stored compressed (zlib, as
// mkvmerge stores PGS) and counted uncompressed, more than it takes in the file.
bool statistics_are_true(const Json& streams, const Json& format, const Json& video,
                         const std::optional<std::string>& writing_app) {
    const Json* encoder = tag(format, "ENCODER");
    const std::string writer = writing_app                                   ? *writing_app
                               : encoder != nullptr && encoder->is_string() ? encoder->get<std::string>()
                                                                             : std::string();
    if (!writer.starts_with("Lavf")) {
        return true;
    }
    const long long file_size = whole(member(format, "size"));
    if (file_size <= 0 || !streams.is_array()) {
        return false;
    }
    long long stored_as_is = 0;  // all but the subtitles, which may be stored compressed
    long long video_bytes = 0;
    for (const Json& stream : streams) {
        const std::string type = string_member(stream, "codec_type");
        if (type == "attachment" || is_picture(stream)) {
            continue;  // fonts and covers, as attachments or as pictures: Matroska keeps no statistics of them
        }
        const Json* bytes = tag(stream, "NUMBER_OF_BYTES");
        if (bytes == nullptr) {
            return false;
        }
        const long long size = std::max(0LL, whole(bytes));
        if (type != "subtitle") {
            stored_as_is += size;
        }
        if (&stream == &video) {
            video_bytes = size;
        }
    }
    return stored_as_is <= file_size && stored_as_is * 1000 >= file_size * 995 && video_bytes * 2 >= file_size;
}

// What the video's display matrix says a player turns it by: 0, 90, 180 or 270
// degrees (ffprobe gives it as, say, -180).
int display_rotation(const Json& video) {
    const Json* side_data = member(video, "side_data_list");
    if (side_data == nullptr || !side_data->is_array()) {
        return 0;
    }
    for (const Json& item : *side_data) {
        if (const Json* rotation = member(item, "rotation"); rotation != nullptr && rotation->is_number()) {
            const int whole_turns_off = static_cast<int>(rotation->get<double>()) % 360;
            return whole_turns_off < 0 ? whole_turns_off + 360 : whole_turns_off;
        }
    }
    return 0;
}

// The video stream's bitrate, and where it comes from.
struct VideoRate {
    long long bits_per_second = 0;
    bool whole_file = false;  // the file's, every other stream included
    bool estimated = false;   // the file's less what is not the video's
};

// The video's rate as the file's (`total`, over `container` seconds) less
// its attachments (fonts, whose bytes FFmpeg gives as their extradata), then
// less the container's own share of the rest (the sample's overhead), then
// less every other stream: at its stated rate, else at its rate in the
// sample (none where it has no packet there: a subtitle track with no line
// in the first minutes). Cover art and thumbnails, which video_stream skips
// too, are left out: a picture or a few, not a stream with a rate.
// Nothing where that leaves no rate.
std::optional<long long> rate_less_the_rest(const Json& video, const Json& streams, long long total,
                                            double container, long long file_size, bool statistics,
                                            const PacketSample& sample) {
    if (container <= 0 || !streams.is_array()) {
        return std::nullopt;
    }
    const double sample_length = std::min(sample.seconds, container);
    if (sample_length <= 0) {
        return std::nullopt;
    }
    long long attachment_bytes = 0;
    long long other_rates = 0;
    for (const Json& stream : streams) {
        if (&stream == &video) {
            continue;
        }
        if (string_member(stream, "codec_type") == "attachment") {
            attachment_bytes += std::max(0LL, whole(member(stream, "extradata_size")));
            continue;
        }
        if (is_picture(stream)) {
            continue;
        }
        long long rate = stream_bit_rate(stream, stream_duration(stream, container), file_size, statistics);
        if (const auto sampled = sample.bytes.find(static_cast<int>(whole(member(stream, "index"))));
            rate == 0 && sampled != sample.bytes.end()) {
            rate = std::llround(static_cast<double>(sampled->second) * 8 / sample_length);
        }
        other_rates += rate;
    }
    const double packets_rate =
        (static_cast<double>(total) - static_cast<double>(attachment_bytes) * 8 / container) * (1 - sample.overhead);
    const long long rate = std::llround(packets_rate) - other_rates;
    return rate > 0 ? std::optional(rate) : std::nullopt;
}

// The video stream's own rate where the file records it, or a raw stream's
// (the file is the stream). Otherwise an estimate from the file's: with a
// sample, rate_less_the_rest; without, the file's less its audio tracks'
// stated rates when every one states one (with no audio, the file's). Where
// neither gives one, the file's rate is all there is.
VideoRate video_bit_rate(const Json& video, const Json& streams, const Json& format, double duration,
                         bool raw_stream, const std::optional<PacketSample>& sample, bool statistics) {
    const long long file_size = whole(member(format, "size"));
    if (const long long own = stream_bit_rate(video, duration, file_size, statistics); own != 0) {
        return {own};
    }
    const long long total = whole(member(format, "bit_rate"));
    if (total <= 0) {
        return {};
    }
    if (raw_stream) {
        return {total};
    }
    double container = duration;
    const Json* format_duration = member(format, "duration");
    if (truthy(format_duration)) {
        container = float_of(format_duration).value_or(duration);
    }
    if (sample) {
        if (const std::optional<long long> rate =
                rate_less_the_rest(video, streams, total, container, file_size, statistics, *sample)) {
            return {*rate, false, true};
        }
        return {total, true, false};
    }
    long long audio_rates = 0;
    bool every_rate_known = true;
    if (streams.is_array()) {
        for (const Json& stream : streams) {
            if (string_member(stream, "codec_type") != "audio") {
                continue;
            }
            const long long rate = stream_bit_rate(stream, stream_duration(stream, container), file_size, statistics);
            every_rate_known = every_rate_known && rate != 0;
            audio_rates += rate;
        }
    }
    if (every_rate_known && total > audio_rates) {
        return {total - audio_rates, false, true};
    }
    return {total, true, false};
}

// How long after the file's start the video's first frame comes; 0 where
// either is not known.
double start_offset(const Json& stream, const Json& format) {
    const std::optional<double> stream_start = float_of(member(stream, "start_time"));
    const std::optional<double> file_start = float_of(member(format, "start_time"));
    if (!stream_start || !file_start) {
        return 0.0;
    }
    return std::max(0.0, *stream_start - *file_start);
}

// The part of a Matroska video's length (as stream_duration took it) before
// its first frame. Matroska's segment duration, and the DURATION tags
// FFmpeg and mkvmerge write, run from the file's zero to the last frame's
// end, so a video whose first frame comes at 0.125 s holds 0.125 s fewer
// frames than its length says. Only a length equal to the container's is
// taken to count from zero: a stream's own duration, or a tag shorter than
// the container, may count from the first frame already.
double matroska_lead_in(const Json& stream, const Json& format, double length, double container) {
    const bool matroska = string_member(format, "format_name").find("matroska") != std::string::npos;
    if (!matroska || truthy(member(stream, "duration"))) {
        return 0.0;
    }
    constexpr double same_length = 0.01;
    const std::optional<double> first_frame = float_of(member(stream, "start_time"));
    if (!first_frame || *first_frame <= 0.0 || *first_frame >= length || std::abs(length - container) > same_length) {
        return 0.0;
    }
    return *first_frame;
}

int integer_member(const Json& object, const char* key) {
    const Json* value = member(object, key);
    if (value == nullptr) {
        return 0;
    }
    if (value->is_string()) {
        return static_cast<int>(python_int(value->get_ref<const std::string&>()).value_or(0));
    }
    return value->is_number() ? static_cast<int>(value->get<double>()) : 0;
}

}  // namespace

VideoInfo video_info_from_probe(const std::filesystem::path& path, std::string_view probe_json,
                                const std::optional<PacketSample>& sample,
                                const std::optional<std::string>& writing_app) {
    Json data;
    try {
        data = Json::parse(replace_invalid_utf8(probe_json));
    } catch (const Json::parse_error&) {
        throw ProbeError("Could not parse ffprobe output for " + text(path));
    }
    const Json empty = Json::object();
    const Json* found_streams = member(data, "streams");
    const Json& streams = found_streams != nullptr ? *found_streams : empty;
    const Json* video = video_stream(streams);
    if (video == nullptr) {
        throw ProbeError("No video stream found in " + text(path));
    }
    const Json* found_format = member(data, "format");
    const Json& format = found_format != nullptr && found_format->is_object() ? *found_format : empty;

    VideoInfo info;
    info.path = path;
    info.fps = parse_frame_rate(first_truthy(*video, {"avg_frame_rate", "r_frame_rate"}, "0/1"));
    if (info.fps <= 0) {
        info.fps = parse_frame_rate(first_truthy(*video, {"r_frame_rate"}, "0/1"));
    }
    info.nominal_fps = parse_frame_rate(first_truthy(*video, {"r_frame_rate"}, "0/1"));
    info.average_fps = parse_frame_rate(first_truthy(*video, {"avg_frame_rate"}, "0/1"));

    double container = 0.0;
    if (const Json* duration = member(format, "duration"); truthy(duration)) {
        container = float_of(duration).value_or(0.0);
    }
    info.duration = stream_duration(*video, container);
    info.lead_in = matroska_lead_in(*video, format, info.duration, container);

    if (const Json* frames = member(*video, "nb_frames"); truthy(frames)) {
        info.nb_frames = frames->is_string() ? python_int(frames->get_ref<const std::string&>()).value_or(0)
                                             : static_cast<long long>(frames->get<double>());
    }
    info.format_name = string_member(format, "format_name");
    const bool statistics = statistics_are_true(streams, format, *video, writing_app);
    const VideoRate rate =
        video_bit_rate(*video, streams, format, info.duration, info.is_raw_stream(), sample, statistics);
    info.bit_rate = rate.bits_per_second;
    info.bit_rate_whole_file = rate.whole_file;
    info.bit_rate_estimated = rate.estimated;

    info.rotation = display_rotation(*video);
    info.width = integer_member(*video, "width");
    info.height = integer_member(*video, "height");
    info.codec_name = string_member(*video, "codec_name", "unknown");
    info.profile = string_member(*video, "profile");
    info.level = integer_member(*video, "level");
    if (statistics) {
        info.stream_bytes = std::max(0LL, whole(tag(*video, "NUMBER_OF_BYTES")));
    }
    info.writing_library = tag_text(format, "ENCODER");
    info.encoder = tag_text(*video, "ENCODER");
    for (const Json* tagged : {&*video, &format}) {
        for (const char* name : {"ENCODER_SETTINGS", "ENCODING_SETTINGS"}) {
            if (info.encoder_settings.empty()) {
                info.encoder_settings = tag_text(*tagged, name);
            }
        }
    }
    info.sar = first_truthy(*video, {"sample_aspect_ratio"}, "1:1");
    info.pix_fmt = string_member(*video, "pix_fmt");
    info.color_range = string_member(*video, "color_range");
    info.color_space = string_member(*video, "color_space");
    info.color_transfer = string_member(*video, "color_transfer");
    info.color_primaries = string_member(*video, "color_primaries");
    info.chroma_location = string_member(*video, "chroma_location");
    info.start_offset = start_offset(*video, format);
    return info;
}

int packet_sample_seconds(long long bits_per_second) {
    constexpr double sample_bytes = 16.0 * 1024 * 1024;
    constexpr double shortest = 5;
    constexpr double longest = 120;
    if (bits_per_second <= 0) {
        return static_cast<int>(longest);
    }
    return static_cast<int>(std::clamp(std::round(sample_bytes * 8 / static_cast<double>(bits_per_second)),
                                       shortest, longest));
}

PacketSample packet_sample_from_csv(std::string_view packets, double seconds) {
    // The most of the bytes the packets span that may be the container's: a
    // larger share is no container's (MPEG-TS's is about 5 %), but a stretch
    // the sample's packets do not cover.
    constexpr double most_overhead = 0.5;
    PacketSample sample;
    sample.seconds = seconds;
    long long all_bytes = 0;
    long long first_byte = std::numeric_limits<long long>::max();  // the span of the packets with a place
    long long past_last_byte = 0;
    int placed = 0;
    for (const std::string_view line : python_splitlines(packets)) {
        // ffprobe's order, not the one asked for: the stream's index, the
        // packet's size, its place in the file ("N/A" for a packet with
        // none: MPEG-TS's audio frames split from a PES packet), then
        // possibly more fields (side data).
        const std::size_t first = line.find(',');
        const std::size_t second = first == line.npos ? line.npos : line.find(',', first + 1);
        if (second == line.npos) {
            continue;
        }
        const std::size_t third = line.find(',', second + 1);
        const std::optional<long long> index = python_int(line.substr(0, first));
        const std::optional<long long> bytes = python_int(line.substr(first + 1, second - first - 1));
        if (!index || !bytes || *bytes < 0) {
            continue;
        }
        sample.bytes[static_cast<int>(*index)] += *bytes;
        all_bytes += *bytes;
        const std::string_view place_text =
            python_strip(third == line.npos ? line.substr(second + 1) : line.substr(second + 1, third - second - 1));
        if (const std::optional<long long> place = python_int(place_text); place && *place >= 0) {
            first_byte = std::min(first_byte, *place);
            past_last_byte = std::max(past_last_byte, *place + *bytes);
            ++placed;
        }
    }
    if (placed >= 2 && past_last_byte > first_byte + all_bytes) {
        const double span = static_cast<double>(past_last_byte - first_byte);
        sample.overhead = std::min(most_overhead, 1 - static_cast<double>(all_bytes) / span);
    }
    return sample;
}

VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe) {
    return probe_video(path, ffprobe, CancelToken());
}

namespace {

// ffprobe's output for `command` (ffprobe reading `path`), within `limit`.
// Throws ProbeError, with why, when it cannot be had, and Cancelled.
std::string run_ffprobe(const std::vector<std::string>& command, const std::filesystem::path& path,
                        std::chrono::seconds limit, const CancelToken& cancel) {
    const std::string& ffprobe = command.front();
    platform::ProcessOutput output;
    try {
        output = platform::run_program(command, limit, {}, [&cancel] { return cancel.cancelled(); });
    } catch (const platform::ProcessError& error) {
        if (!platform::find_on_path(ffprobe) && !std::filesystem::exists(platform::path_from_text(ffprobe))) {
            throw ProbeError(
                "ffprobe was not found. Make sure ffmpeg is installed and on PATH, "
                "or set a custom ffmpeg folder in Settings.");
        }
        throw ProbeError("Could not start ffprobe for " + text(path) + ": " + error.what());
    }
    if (output.stopped) {
        throw Cancelled();
    }
    if (output.timed_out) {
        throw ProbeError("ffprobe timed out while reading " + text(path));
    }
    if (output.exit_code != 0) {
        std::string error = output.standard_error;
        while (!error.empty() && std::isspace(static_cast<unsigned char>(error.back()))) {
            error.pop_back();
        }
        const std::size_t start = error.find_first_not_of(" \t\r\n");
        // It says nothing where it crashed.
        const std::string said = start == std::string::npos
                                     ? "it ended with exit code " + platform::exit_code_text(output.exit_code)
                                     : error.substr(start);
        throw ProbeError("ffprobe failed for " + text(path) + ":\n" + said);
    }
    return output.standard_output;
}

// The packets of the file's first seconds (packet_sample_seconds of a file
// of `bits_per_second`), for the rates of the streams that state none (Opus
// and subtitles in Matroska) and the container's own share: the whole file
// is not read (a 2-hour film's first two minutes, 7 to 13 MB, took 0.14 to
// 0.6 s; seeking to other points in an FFmpeg-written Matroska file read
// most of it). A start that is not typical of the film is a few percent off
// its average (audio 108.7 against 112.9 kb/s), under 1 % of the video's
// rate. None where they cannot be read.
std::optional<PacketSample> sampled_packets(const std::filesystem::path& path, long long bits_per_second,
                                            const std::string& ffprobe, const CancelToken& cancel) {
    const int seconds = packet_sample_seconds(bits_per_second);
    // Without the stream analysis first; with it where that reads nothing
    for (const bool analysed : {false, true}) {
        std::vector<std::string> command = ffprobe_command_start(ffprobe);
        if (!analysed) {
            append(command, no_stream_analysis());
        }
        command.insert(command.end(), {"-read_intervals", "%+" + std::to_string(seconds), "-show_entries",
                                       "packet=stream_index,size,pos", "-of", "csv=p=0", text(path)});
        try {
            PacketSample sample =
                packet_sample_from_csv(run_ffprobe(command, path, std::chrono::seconds(60), cancel), seconds);
            if (!sample.bytes.empty()) {
                return sample;
            }
        } catch (const ProbeError&) {
            // tried again with the analysis; after that, the rate stays as without the sample
        }
    }
    return std::nullopt;
}

}  // namespace

std::vector<std::string> ffprobe_command_start(const std::string& ffprobe) {
    return {ffprobe, "-v", "error", "-export_side_data", "film_grain"};
}

std::vector<std::string> no_stream_analysis() {
    return {"-probesize", "32", "-analyzeduration", "0"};  // 32: FFmpeg's least
}

bool listed_sizes_differ(std::string_view listed, int width, int height) {
    const std::string header = std::to_string(width) + "," + std::to_string(height);
    for (const std::string_view line : python_splitlines(listed)) {
        std::string_view size = python_strip(line);
        // Width and height, then a column for each kind of side data a frame has (HDR videos' do).
        const std::size_t after_width = size.find(',');
        if (const std::size_t end = after_width == size.npos ? size.npos : size.find(',', after_width + 1);
            end != size.npos) {
            size = size.substr(0, end);
        }
        if (!size.empty() && size != header) {
            return true;
        }
    }
    return false;
}

namespace {

// Whether any of an AV1 video's first pictures is coded at another size than
// its header says (VideoInfo::coded_at_another_size), as decoded in software.
// Not known, as for a file ffprobe cannot decode: as its header says.
bool av1_coded_at_another_size(const VideoInfo& info, const std::string& ffprobe, const CancelToken& cancel) {
    std::string listed;
    try {
        listed = run_ffprobe({ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#8",
                              "-show_entries", "frame=width,height", "-of", "csv=p=0", text(info.path)},
                             info.path, std::chrono::seconds(30), cancel);
    } catch (const ProbeError&) {
        return false;
    }
    return listed_sizes_differ(listed, info.width, info.height);
}

// The packets and pictures read to judge a file's timestamps: enough to pass
// a group of reordered pictures several times over.
constexpr const char* stamps_judged = "%+#32";

// Whether the video stream in ffprobe's JSON says its pictures are reordered
// (has_b_frames). Some H.264 files say 0 until decoded: those are not judged.
bool reorders_pictures(std::string_view probe_json) {
    const Json data = Json::parse(replace_invalid_utf8(probe_json), nullptr, false);
    const Json* streams = data.is_discarded() ? nullptr : member(data, "streams");
    const Json* video = streams != nullptr ? video_stream(*streams) : nullptr;
    return video != nullptr && integer_member(*video, "has_b_frames") > 0;
}

// The first packets' or pictures' timestamps (`entry`: packet=pts or
// frame=pts); nothing when ffprobe cannot list them.
std::optional<std::vector<long long>> first_timestamps(const VideoInfo& info, const char* entry,
                                                       const std::string& ffprobe, const CancelToken& cancel) {
    // Begun as every ffprobe is (ffprobe_command_start): listing pictures decodes them.
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-select_streams", "v:0", "-read_intervals", stamps_judged, "-show_entries", entry,
                                   "-of", "csv=p=0", text(info.path)});
    try {
        return listed_timestamps(run_ffprobe(command, info.path, std::chrono::seconds(30), cancel));
    } catch (const ProbeError&) {
        return std::nullopt;
    }
}

// Whether a file's timestamps are in decoding order: its packets' stamps
// only rise as stored (cheap: read, not decoded), and, decoded, its pictures'
// go back. Raw streams carry no stamps of their own and are not asked.
bool stamped_in_decoding_order(const VideoInfo& info, std::string_view probe_json, const std::string& ffprobe,
                               const CancelToken& cancel) {
    if (info.is_raw_stream() || !reorders_pictures(probe_json)) {
        return false;
    }
    const auto packets = first_timestamps(info, "packet=pts", ffprobe, cancel);
    if (!packets || !stamps_rise_as_stored(*packets)) {
        return false;
    }
    const auto pictures = first_timestamps(info, "frame=pts", ffprobe, cancel);
    return pictures && stamps_go_back_as_shown(*pictures);
}

}  // namespace

std::optional<std::vector<long long>> listed_timestamps(std::string_view listed) {
    std::vector<long long> stamps;
    for (const std::string_view line : python_splitlines(listed)) {
        const std::string_view stamp = python_strip(line);
        if (stamp.empty()) {
            continue;
        }
        const std::optional<long long> value = python_int(std::string(stamp));
        if (!value) {
            return std::nullopt;  // N/A: no stamp to judge by
        }
        stamps.push_back(*value);
    }
    return stamps;
}

bool stamps_rise_as_stored(const std::vector<long long>& packet_stamps) {
    constexpr std::size_t fewest_judged = 16;
    return packet_stamps.size() >= fewest_judged &&
           std::adjacent_find(packet_stamps.begin(), packet_stamps.end(), std::greater_equal<>()) ==
               packet_stamps.end();
}

bool stamps_go_back_as_shown(const std::vector<long long>& picture_stamps) {
    return std::adjacent_find(picture_stamps.begin(), picture_stamps.end(), std::greater<>()) != picture_stamps.end();
}

namespace {

// ffprobe's JSON of the file's format and streams.
std::string format_and_streams(const std::filesystem::path& path, const std::string& ffprobe,
                               const CancelToken& cancel) {
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-print_format", "json", "-show_format", "-show_streams", text(path)});
    return run_ffprobe(command, path, std::chrono::seconds(60), cancel);
}

}  // namespace

bool needs_rate_sample(const VideoInfo& info) {
    return (info.bit_rate_whole_file || info.bit_rate_estimated) && !info.is_raw_stream();
}

std::optional<VideoInfo> sampled_video_rate(const std::filesystem::path& path, const std::string& ffprobe,
                                            const CancelToken& cancel) {
    if (is_av2_file(path)) {
        return std::nullopt;  // ffprobe cannot read AV2
    }
    try {
        const std::string probe = format_and_streams(path, ffprobe, cancel);
        const std::optional<std::string> writer = matroska_writing_app(path);
        const VideoInfo info = video_info_from_probe(path, probe, std::nullopt, writer);
        if (const std::optional<PacketSample> sample = sampled_packets(path, info.bit_rate, ffprobe, cancel)) {
            return video_info_from_probe(path, probe, sample, writer);
        }
    } catch (const ProbeError&) {
        // The file changed or went since it was read: its rate stays.
    }
    return std::nullopt;
}

VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe, const CancelToken& cancel,
                      RateSample rate_sample) {
    if (is_av2_file(path)) {
        return probe_av2(path, cancel);  // ffprobe cannot read AV2
    }
    const std::string probe = format_and_streams(path, ffprobe, cancel);
    const std::optional<std::string> writer = matroska_writing_app(path);
    VideoInfo info = video_info_from_probe(path, probe, std::nullopt, writer);
    if (rate_sample == RateSample::now && needs_rate_sample(info)) {
        // The video states no rate of its own: a short stretch of the
        // file's packets gives the other streams' and the container's share.
        // The rate found so far (the file's, or the file's less the stated
        // audio) is near enough to size the stretch.
        if (const std::optional<PacketSample> sample = sampled_packets(path, info.bit_rate, ffprobe, cancel)) {
            info = video_info_from_probe(path, probe, sample, writer);
        }
    }
    if (stamped_in_decoding_order(info, probe, ffprobe, cancel)) {
        throw ProbeError(text(path) +
                         " has its timestamps in decoding order, not in the order its pictures are shown (a raw "
                         "stream copied into a container with no timestamps of its own), so its pictures cannot be "
                         "placed. Add the raw stream instead, or mux it with a tool that writes the pictures' "
                         "display times.");
    }
    info.coded_at_another_size = ascii_lower(info.codec_name) == "av1" && av1_coded_at_another_size(info, ffprobe, cancel);
    return info;
}

std::optional<long long> stated_frame_count(const std::filesystem::path& path, const std::string& ffprobe,
                                            const CancelToken& cancel) {
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-select_streams", "v:0", "-print_format", "json", "-show_entries",
                                   "stream=nb_frames:stream_tags=NUMBER_OF_FRAMES,NUMBER_OF_FRAMES-eng", text(path)});
    const Json probe = Json::parse(run_ffprobe(command, path, std::chrono::seconds(60), cancel));
    const Json* streams = member(probe, "streams");
    if (streams == nullptr || streams->empty()) {
        return std::nullopt;
    }
    const auto count_in = [](const Json& holder, const char* key) -> std::optional<long long> {
        const Json* value = member(holder, key);
        return value != nullptr && value->is_string() ? python_int(value->get_ref<const std::string&>())
                                                      : std::optional<long long>();
    };
    const Json& stream = streams->at(0);
    if (const std::optional<long long> count = count_in(stream, "nb_frames"); count && *count > 0) {
        return count;
    }
    const Json* tags = member(stream, "tags");
    if (tags == nullptr) {
        return std::nullopt;
    }
    for (const char* key : {"NUMBER_OF_FRAMES", "NUMBER_OF_FRAMES-eng"}) {
        if (const std::optional<long long> count = count_in(*tags, key); count && *count > 0) {
            return count;
        }
    }
    return std::nullopt;
}

long long count_video_packets(const std::filesystem::path& path, const std::string& ffprobe,
                              const CancelToken& cancel) {
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-select_streams", "v:0", "-count_packets", "-print_format", "json",
                                   "-show_entries", "stream=nb_read_packets", text(path)});
    const Json probe = Json::parse(run_ffprobe(command, path, std::chrono::minutes(5), cancel));
    const Json* streams = member(probe, "streams");
    const Json* packets = streams != nullptr && !streams->empty() ? member(streams->at(0), "nb_read_packets") : nullptr;
    return packets != nullptr && packets->is_string()
               ? python_int(packets->get_ref<const std::string&>()).value_or(0)
               : 0;
}

namespace {

// In ffprobe's "pts,flags" listing of packets (csv=p=0), how many are shown
// (with a presentation time, not flagged to be discarded) and how many are
// not. A stream cut by copying can begin with pictures that refer to ones
// before the cut (an open GOP's leading pictures): Matroska gives them no
// time, MP4 a time before the start and the discard flag, and no decoder
// delivers them.
std::pair<long long, long long> shown_and_unshown(const std::string& listed) {
    long long shown = 0;
    long long unshown = 0;
    std::size_t start = 0;
    while (start < listed.size()) {
        std::size_t end = listed.find('\n', start);
        end = end == std::string::npos ? listed.size() : end;
        std::string_view line(listed.data() + start, end - start);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
            line.remove_suffix(1);
        }
        start = end + 1;
        if (line.empty()) {
            continue;
        }
        const std::size_t comma = line.find(',');
        const std::string_view pts = line.substr(0, comma);
        const std::string_view flags = comma == std::string_view::npos ? std::string_view() : line.substr(comma + 1);
        if (pts.empty() || pts == "N/A" || flags.find('D') != std::string_view::npos) {
            ++unshown;
        } else {
            ++shown;
        }
    }
    return {shown, unshown};
}

// The packets' "pts,flags" listing of `path`'s first video stream, the
// first `first_packets` of them when above 0, otherwise all.
std::string listed_packets(const std::filesystem::path& path, const std::string& ffprobe, int first_packets,
                           const CancelToken& cancel) {
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-select_streams", "v:0"});
    if (first_packets > 0) {
        command.insert(command.end(), {"-read_intervals", "%+#" + std::to_string(first_packets)});
    }
    command.insert(command.end(), {"-show_entries", "packet=pts,flags", "-print_format", "csv=p=0", text(path)});
    // A film on a network drive is read through: minutes, not seconds
    return run_ffprobe(command, path, first_packets > 0 ? std::chrono::seconds(60) : std::chrono::minutes(30),
                       cancel);
}

}  // namespace

std::optional<long long> trusted_frame_count(const std::filesystem::path& path, const std::string& ffprobe,
                                             const CancelToken& cancel) {
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-select_streams", "v:0", "-print_format", "json", "-show_entries",
                                   "stream=nb_frames:stream_tags=NUMBER_OF_FRAMES,NUMBER_OF_FRAMES-eng:"
                                   "format_tags=encoder,ENCODER",
                                   text(path)});
    const Json probe = Json::parse(run_ffprobe(command, path, std::chrono::seconds(60), cancel));
    const Json* streams = member(probe, "streams");
    if (streams == nullptr || streams->empty()) {
        return std::nullopt;
    }
    const auto count_in = [](const Json& holder, const char* key) -> std::optional<long long> {
        const Json* value = member(holder, key);
        return value != nullptr && value->is_string() ? python_int(value->get_ref<const std::string&>())
                                                      : std::optional<long long>();
    };
    const Json& stream = streams->at(0);
    std::optional<long long> stated = count_in(stream, "nb_frames");
    if (!stated || *stated <= 0) {
        stated.reset();
        // FFmpeg copies a stream's tags into what it writes as they were
        const Json* format = member(probe, "format");
        const Json* format_tags = format != nullptr ? member(*format, "tags") : nullptr;
        if (format_tags != nullptr) {
            for (const char* key : {"encoder", "ENCODER"}) {
                const Json* encoder = member(*format_tags, key);
                if (encoder != nullptr && encoder->is_string() &&
                    encoder->get_ref<const std::string&>().starts_with("Lavf")) {
                    return std::nullopt;
                }
            }
        }
        if (const Json* tags = member(stream, "tags")) {
            for (const char* key : {"NUMBER_OF_FRAMES", "NUMBER_OF_FRAMES-eng"}) {
                if (const std::optional<long long> count = count_in(*tags, key); count && *count > 0) {
                    stated = count;
                    break;
                }
            }
        }
    }
    if (!stated) {
        return std::nullopt;
    }
    // A count states every picture stored: less the leading pictures a cut
    // left, which are only at the start, read from its first packets.
    const long long unshown = shown_and_unshown(listed_packets(path, ffprobe, 64, cancel)).second;
    return *stated - unshown;
}

long long count_shown_video_packets(const std::filesystem::path& path, const std::string& ffprobe,
                                    const CancelToken& cancel) {
    return shown_and_unshown(listed_packets(path, ffprobe, 0, cancel)).first;
}

namespace {

// Each codec whose raw streams' stored rate FFmpeg reports, and how many of
// what it reports make a frame a second. Measured on streams made at known
// rates, 24000/1001 among them: H.264's and MPEG-1's are reported doubled, as
// H.264's headers count in fields, also for progressive video. Any other codec
// goes unlisted because it stores none (EVC, APV, MJPEG), or what is reported
// is its format's clock and not the video's (H.261 and H.263 read 30000/1001
// made at 15 too), or no stream of it could be made to check (VC-1, CAVS, AVS3).
struct StoredRate {
    std::string_view codec;
    int reports_per_frame;
};
constexpr std::array<StoredRate, 9> stored_rates{{{"h264", 2},
                                                  {"mpeg1video", 2},
                                                  {"hevc", 1},
                                                  {"vvc", 1},
                                                  {"av1", 1},
                                                  {"mpeg2video", 1},
                                                  {"mpeg4", 1},
                                                  {"dirac", 1},
                                                  {"avs2", 1}}};

const StoredRate* stored_rate_of(std::string_view codec_name) {
    const auto found = std::find_if(stored_rates.begin(), stored_rates.end(),
                                    [codec_name](const StoredRate& rate) { return rate.codec == codec_name; });
    return found == stored_rates.end() ? nullptr : &*found;
}

// What is reported above this was not stored. Told to read a stream at the
// rate below, FFmpeg reports that rate where a stream has none (or its demuxer's
// 1200000 ticks a second), and leaves a stored one as it is.
constexpr double highest_reported_frame_rate = 1000;
constexpr const char* rate_no_stream_stores = "1999";

// The frames of `info`, a raw stream. ffprobe's parser splits the stream into
// its frames without decoding them (3.4 s for a 4K VVC film of 151,919 frames,
// 800 MB), but reads the whole file, so it has as long as a slow disk needs.
// (The csv format adds a comma after some streams' count: "480,".) A raw YUV
// file is read with its size and pixel format (probe_options).
long long counted_frames(const VideoInfo& info, const std::string& ffprobe, const CancelToken& cancel) {
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    const std::vector<std::string> input = info.probe_options();
    append(command, input);
    command.insert(command.end(), {"-select_streams", "v:0", "-count_packets", "-show_entries",
                                   "stream=nb_read_packets", "-of", "default=noprint_wrappers=1:nokey=1",
                                   text(info.path)});
    const std::optional<long long> frames =
        python_int(python_strip(run_ffprobe(command, info.path, std::chrono::hours(1), cancel)));
    if (!frames || *frames <= 0) {
        throw ProbeError("ffprobe found no frames in " + text(info.path));
    }
    return *frames;
}

}  // namespace

std::optional<double> stored_frame_rate(std::string_view codec_name, double reported) {
    const StoredRate* rate = stored_rate_of(codec_name);
    if (rate == nullptr || reported <= 0 || reported > highest_reported_frame_rate) {
        return std::nullopt;
    }
    return reported / rate->reports_per_frame;
}

std::optional<double> frame_rate_in_stream(const VideoInfo& info, const std::string& ffprobe,
                                           const CancelToken& cancel) {
    if (stored_rate_of(info.codec_name) == nullptr) {
        return std::nullopt;
    }
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-framerate", rate_no_stream_stores, "-select_streams", "v:0", "-show_entries",
                                   "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
                                   text(info.path)});
    try {
        const std::string reported(python_strip(run_ffprobe(command, info.path, std::chrono::seconds(60), cancel)));
        return stored_frame_rate(info.codec_name, parse_frame_rate(reported));
    } catch (const ProbeError&) {
        return std::nullopt;  // it was read a moment ago: only its rate stays unknown
    }
}

VideoInfo raw_stream_as_found(VideoInfo info, const std::string& ffprobe, const CancelToken& cancel) {
    const std::optional<double> rate = frame_rate_in_stream(info, ffprobe, cancel);
    if (info.codec_name != "av2") {  // its units were counted as it was probed (av2_probe.h)
        info.nb_frames = counted_frames(info, ffprobe, cancel);
    }
    if (rate) {
        info.set_frame_rate(*rate);
    } else {
        info.forget_frame_rate();
    }
    return info;
}

VideoInfo timed_raw_stream(VideoInfo info, double fps, const std::string& ffprobe, const CancelToken& cancel) {
    info.timed_raw = true;  // a raw YUV file is read at it (probe_options)
    info.fps = fps;
    if (info.codec_name != "av2") {  // its units were counted as it was probed (av2_probe.h)
        info.nb_frames = counted_frames(info, ffprobe, cancel);
    }
    info.set_frame_rate(fps);
    return info;
}

std::optional<double> typed_frame_rate(const std::string& text) {
    static const std::regex fraction("([0-9]+)/([0-9]+)");
    const std::string stripped(python_strip(text));
    std::smatch parts;
    if (!std::regex_match(stripped, parts, fraction)) {
        return python_float(text);
    }
    constexpr long long exact = 1LL << 53;
    const std::optional<long long> numerator = python_int(parts[1].str());
    const std::optional<long long> denominator = python_int(parts[2].str());
    if (!numerator || !denominator || *denominator == 0 || *numerator >= exact || *denominator >= exact) {
        return std::nullopt;
    }
    return static_cast<double>(*numerator) / static_cast<double>(*denominator);
}

VideoInfo probe_raw_yuv(const std::filesystem::path& path, const std::string& size, const std::string& pix_fmt,
                        double fps, const std::string& ffprobe, const CancelToken& cancel) {
    std::vector<std::string> command = ffprobe_command_start(ffprobe);
    command.insert(command.end(), {"-f", "rawvideo", "-video_size", size, "-pixel_format", pix_fmt, "-framerate",
                                   python_repr(fps), "-print_format", "json", "-show_format", "-show_streams",
                                   text(path)});
    return timed_raw_stream(video_info_from_probe(path, run_ffprobe(command, path, std::chrono::seconds(60), cancel)),
                            fps, ffprobe, cancel);
}

}  // namespace vml
