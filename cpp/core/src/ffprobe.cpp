#include "vml/ffprobe.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

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

// The first video stream that is a video, not cover art or a thumbnail,
// which FFmpeg lists as video streams too (the same stream FFmpeg's
// commands read: "V:0").
const Json* video_stream(const Json& streams) {
    if (!streams.is_array()) {
        return nullptr;
    }
    for (const Json& stream : streams) {
        const Json* disposition = member(stream, "disposition");
        const auto disposed = [disposition](const char* key) {
            return disposition != nullptr && truthy(member(*disposition, key));
        };
        if (string_member(stream, "codec_type") == "video" && !disposed("attached_pic") &&
            !disposed("timed_thumbnails")) {
            return &stream;
        }
    }
    return nullptr;
}

// A stream tag, by its name in any case (Matroska writers differ).
const Json* tag(const Json& stream, std::string_view name) {
    const Json* tags = member(stream, "tags");
    if (tags == nullptr || !tags->is_object()) {
        return nullptr;
    }
    for (auto item = tags->begin(); item != tags->end(); ++item) {
        const std::string key = ascii_upper(item.key());
        if (key == name) {
            return &item.value();
        }
    }
    return nullptr;
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
// 0 when there is none. The statistics are used only if they fit in the
// file: FFmpeg copies them from its input into what it writes, even through
// a re-encode.
long long stream_bit_rate(const Json& stream, double duration, long long file_size) {
    const long long rate = whole(member(stream, "bit_rate"));
    if (rate > 0) {
        return rate;
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

// The video stream's bitrate, and whether it is the whole file's instead.
// Without the video's own, the audio tracks' rates are taken from the
// file's when every one is known; otherwise the file's rate is all there is.
std::pair<long long, bool> video_bit_rate(const Json& video, const Json& streams, const Json& format,
                                          double duration) {
    const long long file_size = whole(member(format, "size"));
    if (const long long own = stream_bit_rate(video, duration, file_size); own != 0) {
        return {own, false};
    }
    const long long total = whole(member(format, "bit_rate"));
    if (total <= 0) {
        return {0, false};
    }
    double container = duration;
    const Json* format_duration = member(format, "duration");
    if (truthy(format_duration)) {
        container = float_of(format_duration).value_or(duration);
    }
    long long audio_rates = 0;
    bool every_rate_known = true;
    if (streams.is_array()) {
        for (const Json& stream : streams) {
            if (string_member(stream, "codec_type") != "audio") {
                continue;
            }
            const long long rate = stream_bit_rate(stream, stream_duration(stream, container), file_size);
            every_rate_known = every_rate_known && rate != 0;
            audio_rates += rate;
        }
    }
    if (every_rate_known && total > audio_rates) {
        return {total - audio_rates, false};
    }
    return {total, true};
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

VideoInfo video_info_from_probe(const std::filesystem::path& path, std::string_view probe_json) {
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

    if (const Json* frames = member(*video, "nb_frames"); truthy(frames)) {
        info.nb_frames = frames->is_string() ? python_int(frames->get_ref<const std::string&>()).value_or(0)
                                             : static_cast<long long>(frames->get<double>());
    }
    const auto [rate, whole_file] = video_bit_rate(*video, streams, format, info.duration);
    info.bit_rate = rate;
    info.bit_rate_whole_file = whole_file;

    info.width = integer_member(*video, "width");
    info.height = integer_member(*video, "height");
    info.codec_name = string_member(*video, "codec_name", "unknown");
    info.sar = first_truthy(*video, {"sample_aspect_ratio"}, "1:1");
    info.pix_fmt = string_member(*video, "pix_fmt");
    info.color_range = string_member(*video, "color_range");
    info.color_space = string_member(*video, "color_space");
    info.color_transfer = string_member(*video, "color_transfer");
    info.color_primaries = string_member(*video, "color_primaries");
    info.chroma_location = string_member(*video, "chroma_location");
    info.format_name = string_member(format, "format_name");
    info.start_offset = start_offset(*video, format);
    return info;
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
        throw ProbeError("ffprobe failed for " + text(path) + ":\n" +
                         (start == std::string::npos ? std::string() : error.substr(start)));
    }
    return output.standard_output;
}

}  // namespace

VideoInfo probe_video(const std::filesystem::path& path, const std::string& ffprobe, const CancelToken& cancel) {
    const std::vector<std::string> command{ffprobe, "-v", "error", "-print_format", "json", "-show_format",
                                           "-show_streams", text(path)};
    return video_info_from_probe(path, run_ffprobe(command, path, std::chrono::seconds(60), cancel));
}

VideoInfo timed_raw_stream(VideoInfo info, double fps, const std::string& ffprobe, const CancelToken& cancel) {
    info.timed_raw = true;
    info.fps = fps;
    // ffprobe's parser splits the stream into its frames without decoding
    // them (3.4 s for a 4K VVC film of 151,919 frames, 800 MB), but reads the
    // whole file, so it has as long as a slow disk needs. (The csv format
    // adds a comma after some streams' count: "480,".) A raw YUV file is
    // read with its size and pixel format (input_options).
    std::vector<std::string> command{ffprobe, "-v", "error"};
    const std::vector<std::string> input = info.input_options();
    command.insert(command.end(), input.begin(), input.end());
    command.insert(command.end(), {"-select_streams", "v:0", "-count_packets", "-show_entries",
                                   "stream=nb_read_packets", "-of", "default=noprint_wrappers=1:nokey=1",
                                   text(info.path)});
    const std::optional<long long> frames =
        python_int(python_strip(run_ffprobe(command, info.path, std::chrono::hours(1), cancel)));
    if (!frames || *frames <= 0) {
        throw ProbeError("ffprobe found no frames in " + text(info.path));
    }
    info.nominal_fps = fps;
    info.average_fps = fps;
    info.nb_frames = *frames;
    info.duration = static_cast<double>(*frames) / fps;
    // A raw stream's file is its video and nothing else.
    std::error_code unknown;
    const std::uintmax_t bytes = std::filesystem::file_size(info.path, unknown);
    info.bit_rate = unknown ? 0 : std::llround(static_cast<double>(bytes) * 8 / info.duration);
    info.bit_rate_whole_file = false;
    return info;
}

VideoInfo probe_raw_yuv(const std::filesystem::path& path, const std::string& size, const std::string& pix_fmt,
                        double fps, const std::string& ffprobe, const CancelToken& cancel) {
    const std::vector<std::string> command{
        ffprobe, "-v", "error", "-f", "rawvideo", "-video_size", size, "-pixel_format", pix_fmt, "-framerate",
        python_repr(fps), "-print_format", "json", "-show_format", "-show_streams", text(path)};
    return timed_raw_stream(video_info_from_probe(path, run_ffprobe(command, path, std::chrono::seconds(60), cancel)),
                            fps, ffprobe, cancel);
}

}  // namespace vml
