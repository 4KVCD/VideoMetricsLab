#include "vml/file_fingerprint.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

#include "vml/ffprobe.h"
#include "vml/metric_cache.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/sha256.h"

namespace vml {

namespace {

// What sampled_file_hash reads: the first and last megabyte, and 64 KiB at
// each of 14 points between. A version in the digest, so that another
// choice makes other hashes, never the same ones.
constexpr std::uintmax_t edge_bytes = 1 << 20;
constexpr std::uintmax_t sample_bytes = 1 << 16;
constexpr int middle_samples = 14;

void add_text(Sha256& hash, const std::string& text) {
    hash.update(text.data(), text.size());
    hash.update("\n", 1);
}

}  // namespace

std::optional<std::string> sampled_file_hash(const std::filesystem::path& file) {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file, error);
    if (error) {
        return std::nullopt;
    }
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    Sha256 hash;
    add_text(hash, "vml-sampled-1");
    add_text(hash, std::to_string(size));
    std::vector<char> buffer;
    const auto read_at = [&](std::uintmax_t offset, std::uintmax_t count) {
        count = std::min(count, size - std::min(offset, size));
        buffer.resize(static_cast<std::size_t>(count));
        in.clear();
        in.seekg(static_cast<std::streamoff>(offset));
        in.read(buffer.data(), static_cast<std::streamsize>(count));
        if (in.gcount() != static_cast<std::streamsize>(count)) {
            return false;
        }
        hash.update(buffer.data(), buffer.size());
        return true;
    };
    if (size <= 2 * edge_bytes + middle_samples * sample_bytes) {
        // A small file: all of it.
        if (!read_at(0, size)) {
            return std::nullopt;
        }
        return hash.hexdigest();
    }
    if (!read_at(0, edge_bytes)) {
        return std::nullopt;
    }
    const std::uintmax_t middle = size - 2 * edge_bytes;
    for (int i = 1; i <= middle_samples; ++i) {
        const std::uintmax_t offset = edge_bytes + middle / (middle_samples + 1) * static_cast<std::uintmax_t>(i);
        if (!read_at(offset, sample_bytes)) {
            return std::nullopt;
        }
    }
    if (!read_at(size - edge_bytes, edge_bytes)) {
        return std::nullopt;
    }
    return hash.hexdigest();
}

std::vector<double> fingerprint_times(double duration, double fps) {
    std::vector<double> times;
    if (!(fps > 0.0) || !(duration > 0.0)) {
        return times;
    }
    const double whole = std::max(1.0, std::round(duration));
    // The rate to a hundredth: an MP4's average rate is its duration's,
    // which a remux rounds (24.003 for an MKV's 24).
    fps = std::round(fps * 100.0) / 100.0;
    for (int k = 1; k <= 5; ++k) {
        const double frame = std::floor(whole * k / 6.0 * fps);
        times.push_back((frame + 0.5) / fps);
    }
    return times;
}

namespace {

// The video's colour as the app reads it (colour_tags_of): its transfer,
// matrix, range, primaries and chroma siting. A range not stated is limited
// ("tv") for YUV, as it is read: an MKV states the range its stream leaves
// out, and a remux to MP4 can drop it.
void add_colour(Sha256& hash, const VideoInfo& info) {
    ColourTags colour = colour_tags_of(info);
    if ((colour.color_range.empty() || colour.color_range == "unknown") && !info.pix_fmt.starts_with("yuvj")) {
        colour.color_range = "tv";
    }
    for (const std::string* tag : {&colour.color_transfer, &colour.color_space, &colour.color_range,
                                   &colour.color_primaries, &colour.chroma_location}) {
        add_text(hash, *tag);
    }
}

}  // namespace

std::optional<std::string> video_fingerprint(const std::filesystem::path& file, const VideoInfo& info,
                                             const std::string& ffmpeg, const CancelToken& cancel) {
    const std::vector<double> times = fingerprint_times(info.duration, info.fps);
    if (times.empty() || ffmpeg.empty()) {
        return std::nullopt;
    }
    Sha256 hash;
    add_text(hash, "vml-video-1");
    add_text(hash, info.codec_name);
    add_text(hash, std::to_string(info.width) + "x" + std::to_string(info.height));
    add_text(hash, info.pix_fmt);
    add_text(hash, python_repr(std::round(info.fps * 100.0) / 100.0));
    add_text(hash, std::to_string(static_cast<long long>(std::round(info.duration))));
    add_colour(hash, info);
    for (const double time : times) {
        // One picture, decoded as stored (no scaling or conversion), from
        // the first video stream; -ss before -i seeks and then decodes up
        // to that time exactly.
        const std::vector<std::string> command{ffmpeg,   "-nostdin", "-v",      "error", "-ss",
                                               python_repr(time), "-i", platform::path_text(file),
                                               "-map",   "0:v:0",    "-frames:v", "1",   "-f",
                                               "rawvideo", "-"};
        platform::ProcessOutput out;
        try {
            out = platform::run_program(command, std::chrono::minutes(2), {},
                                        [&cancel] { return cancel.cancelled(); });
        } catch (const std::exception&) {
            return std::nullopt;
        }
        if (out.exit_code != 0 || out.timed_out || out.stopped || out.standard_output.empty()) {
            return std::nullopt;
        }
        Sha256 picture;
        picture.update(out.standard_output.data(), out.standard_output.size());
        add_text(hash, picture.hexdigest());
    }
    return hash.hexdigest();
}

namespace {

// How each codec's stream is made the same whatever held it: to Annex B
// (from MKV's and MP4's length-prefixed units), then only the units that
// carry pictures.
struct StreamFilter {
    const char* codec;
    const char* filters;
};
constexpr StreamFilter stream_filters[] = {
    {"h264", "h264_mp4toannexb,filter_units=pass_types=1-5"},
    {"hevc", "hevc_mp4toannexb,filter_units=pass_types=0-31"},
    {"vvc", "vvc_mp4toannexb,filter_units=pass_types=0-11"},
    // OBU_FRAME_HEADER, OBU_TILE_GROUP, OBU_FRAME: not the temporal
    // delimiters a raw .obu stream has and a container drops.
    {"av1", "filter_units=pass_types=3|4|6"},
};

// The packets an MP4's edit list leaves out, before the first picture
// shown: FFmpeg's MP4 reader gives them times before 0 (and marks them to be
// discarded), and copies them; MKV and raw streams have none. A packet with
// no time (a raw stream's) is kept.
constexpr const char* not_shown_dropped = "noise=drop=lt(pts\\,0)*gt(pts\\,nopts)";

}  // namespace

std::vector<std::string> stream_fingerprint_command(const std::filesystem::path& file, const std::string& codec,
                                                    const std::string& ffmpeg, int packets) {
    const std::string name = ascii_lower(codec);
    std::vector<std::string> command{ffmpeg, "-nostdin", "-v", "error", "-i", platform::path_text(file),
                                     "-map", "0:v:0", "-c", "copy"};
    std::string filters = not_shown_dropped;
    for (const StreamFilter& filter : stream_filters) {
        if (name == filter.codec) {
            filters += std::string(",") + filter.filters;
        }
    }
    command.insert(command.end(), {"-bsf:v", filters});
    if (packets > 0) {
        command.insert(command.end(), {"-frames:v", std::to_string(packets)});
    }
    command.insert(command.end(), {"-f", "hash", "-hash", "sha256", "-"});
    return command;
}

std::optional<std::string> stream_fingerprint(const std::filesystem::path& file, const VideoInfo& info,
                                              const std::string& ffmpeg, int packets, const CancelToken& cancel) {
    const std::string& codec = info.codec_name;
    if (ffmpeg.empty() || codec.empty()) {
        return std::nullopt;
    }
    const std::string name = ascii_lower(codec);
    if (name == "rawvideo") {
        return std::nullopt;  // raw pictures (a .yuv file), whose scores are not saved: nothing to read
    }
    const std::vector<std::string> command = stream_fingerprint_command(file, codec, ffmpeg, packets);
    platform::ProcessOutput out;
    try {
        // A whole feature-length stream read over a network share takes
        // minutes: a limit only against a program that never ends.
        out = platform::run_program(command, std::chrono::hours(6), {}, [&cancel] { return cancel.cancelled(); });
    } catch (const std::exception&) {
        return std::nullopt;
    }
    // "SHA256=<hex>": the hash muxer's one line.
    const std::string text = out.standard_output;
    const std::size_t at = text.find("SHA256=");
    if (out.exit_code != 0 || out.timed_out || out.stopped || at == std::string::npos) {
        return std::nullopt;
    }
    const std::string digest = text.substr(at + 7, 64);
    if (digest.size() != 64) {
        return std::nullopt;
    }
    return stream_fingerprint_of(digest, info, packets);
}

std::string stream_fingerprint_of(const std::string& digest, const VideoInfo& info, int packets) {
    Sha256 hash;
    add_text(hash, "vml-stream-" + std::to_string(stream_fingerprint_format));
    add_text(hash, ascii_lower(info.codec_name));
    add_text(hash, std::to_string(packets));
    add_text(hash, digest);
    // The colour it is read with: a stream re-tagged (h264_metadata, or
    // muxed into a container that states other tags) is read with another
    // YUV->RGB matrix, which SSIMULACRA2, Butteraugli and CVVDP score.
    add_colour(hash, info);
    // The picture as read: its parameter sets are not hashed (only the coded
    // pictures are), and a remux that changes their crop (h264_metadata's
    // crop_right) or its pixel aspect ratio is decoded to another size, or
    // compared with another shape: another video, scored otherwise.
    add_text(hash, std::to_string(info.width) + "x" + std::to_string(info.height));
    const bool square = info.sar.empty() || info.sar == "1:1" || info.sar == "0:1" || info.sar == "N/A";
    add_text(hash, square ? std::string("1:1") : info.sar);
    return hash.hexdigest();
}

std::function<std::optional<std::string>(const std::filesystem::path&, int, const CancelToken&)>
stream_fingerprinter_using(std::string ffmpeg, std::string ffprobe) {
    return [ffmpeg = std::move(ffmpeg), ffprobe = std::move(ffprobe)](
               const std::filesystem::path& file, int packets,
               const CancelToken& cancel) -> std::optional<std::string> {
        if (ffmpeg.empty() || ffprobe.empty()) {
            return std::nullopt;
        }
        try {
            return stream_fingerprint(file, probe_video(file, ffprobe, cancel), ffmpeg, packets, cancel);
        } catch (const std::exception&) {
            return std::nullopt;  // not read: the test video is only recognised by its path and sampled hash
        }
    };
}

std::function<std::optional<std::string>(const std::filesystem::path&)> fingerprinter_using(std::string ffmpeg,
                                                                                         std::string ffprobe) {
    return [ffmpeg = std::move(ffmpeg), ffprobe = std::move(ffprobe)](
               const std::filesystem::path& file) -> std::optional<std::string> {
        if (ffmpeg.empty() || ffprobe.empty()) {
            return std::nullopt;
        }
        try {
            return video_fingerprint(file, probe_video(file, ffprobe), ffmpeg);
        } catch (const std::exception&) {
            return std::nullopt;  // not read: the file is only recognised by its path and sampled hash
        }
    };
}

}  // namespace vml
