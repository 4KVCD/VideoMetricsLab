#include "frame_types.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#ifdef VML_NATIVE_PLAYBACK
#include "ffmpeg_api.h"
#include "native_device.h"
#include "vml/app_log.h"
#include "vml/platform/system.h"
#endif

namespace vml::app {

std::vector<std::uint8_t> types_matching_frames(std::vector<ReadFrame> frames, bool keep_read_order,
                                                std::span<const long long> sizes) {
    if (frames.size() != sizes.size()) {
        return {};
    }
    if (!keep_read_order) {
        std::stable_sort(frames.begin(), frames.end(), [](const ReadFrame& a, const ReadFrame& b) {
            return a.time != b.time ? a.time < b.time : a.position < b.position;
        });
    }
    std::vector<std::uint8_t> types;
    types.reserve(frames.size());
    for (std::size_t index = 0; index < frames.size(); ++index) {
        if (frames[index].size != sizes[index]) {
            return {};
        }
        types.push_back(frames[index].type);
    }
    return types;
}

#ifdef VML_NATIVE_PLAYBACK

namespace {

struct FormatCloser {
    void operator()(AVFormatContext* format) const { avformat_close_input(&format); }
};
struct CodecFreer {
    void operator()(AVCodecContext* codec) const { avcodec_free_context(&codec); }
};
struct ParserCloser {
    void operator()(AVCodecParserContext* parser) const { av_parser_close(parser); }
};
struct PacketFreer {
    void operator()(AVPacket* packet) const { av_packet_free(&packet); }
};

// The video's file, read until `reader` is stopping. Not probed further
// (avformat_find_stream_info): that decodes, and FFmpeg's own VVC decoder can
// crash doing so (native_probe.h).
std::unique_ptr<AVFormatContext, FormatCloser> open_file(const VideoInfo& info, const FrameTypeReader& reader) {
    AVFormatContext* format = avformat_alloc_context();
    if (format == nullptr) {
        return nullptr;
    }
    format->interrupt_callback.callback = [](void* opaque) {
        return static_cast<const FrameTypeReader*>(opaque)->stopping() ? 1 : 0;
    };
    format->interrupt_callback.opaque = const_cast<FrameTypeReader*>(&reader);
    // It frees the context when it fails.
    if (avformat_open_input(&format, platform::path_text(info.path).c_str(), nullptr, nullptr) < 0) {
        return nullptr;
    }
    return std::unique_ptr<AVFormatContext, FormatCloser>(format);
}

// The first video stream that is no attached picture, as ffprobe's -select_streams
// V:0 (bitrate.cpp) picks it; every other stream is skipped.
AVStream* only_the_video(AVFormatContext& format) {
    AVStream* video = nullptr;
    for (unsigned index = 0; index < format.nb_streams; ++index) {
        AVStream* stream = format.streams[index];
        const bool picture = (stream->disposition & AV_DISPOSITION_ATTACHED_PIC) != 0;
        if (video == nullptr && stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && !picture) {
            video = stream;
        } else {
            stream->discard = AVDISCARD_ALL;
        }
    }
    return video;
}

// The type a packet is by its picture type, or by being a keyframe where the
// parser gave none.
std::uint8_t frame_type_of(int picture_type, bool keyframe) {
    switch (picture_type) {
    case AV_PICTURE_TYPE_I:
    case AV_PICTURE_TYPE_SI:
        return frame_type_i;
    case AV_PICTURE_TYPE_P:
    case AV_PICTURE_TYPE_S:
    case AV_PICTURE_TYPE_SP:
        return frame_type_p;
    case AV_PICTURE_TYPE_B:
    case AV_PICTURE_TYPE_BI:
        return frame_type_b;
    default:
        return keyframe ? frame_type_i : frame_type_unknown;
    }
}

// FFmpeg's parser for the stream's codec, where the bundled libraries have one
// (not MPEG-4 Part 2 or ProRes): without it, a packet's picture type is not known.
class PictureTypeParser {
public:
    explicit PictureTypeParser(const AVStream& video)
        : parser_(av_parser_init(video.codecpar->codec_id)), codec_(avcodec_alloc_context3(nullptr)) {
        if (!parser_ || !codec_ || avcodec_parameters_to_context(codec_.get(), video.codecpar) < 0) {
            parser_.reset();
            return;
        }
        parser_->flags |= PARSER_FLAG_COMPLETE_FRAMES;  // a packet is a frame: nothing is held back
    }

    int picture_type(const AVPacket& packet) {
        if (!parser_) {
            return AV_PICTURE_TYPE_NONE;
        }
        std::uint8_t* output = nullptr;
        int output_size = 0;
        parser_->pict_type = AV_PICTURE_TYPE_NONE;
        av_parser_parse2(parser_.get(), codec_.get(), &output, &output_size, packet.data, packet.size, packet.pts,
                         packet.dts, packet.pos);
        return parser_->pict_type;
    }

private:
    std::unique_ptr<AVCodecParserContext, ParserCloser> parser_;
    std::unique_ptr<AVCodecContext, CodecFreer> codec_;
};

// A packet's time as PacketList::finish takes ffprobe's: the decode time for a
// packet without a presentation time, and `next_time`, after the packets before
// it, for one with neither.
double packet_time(const AVPacket& packet, double seconds_per_tick, double next_time) {
    if (packet.pts != AV_NOPTS_VALUE) {
        return static_cast<double>(packet.pts) * seconds_per_tick;
    }
    return packet.dts != AV_NOPTS_VALUE ? static_cast<double>(packet.dts) * seconds_per_tick : next_time;
}

// Every packet of the video, with its type.
std::vector<ReadFrame> read_frames(AVFormatContext& format) {
    const std::unique_ptr<AVPacket, PacketFreer> packet(av_packet_alloc());
    if (!packet) {
        return {};
    }
    std::vector<ReadFrame> frames;
    const AVStream* video = nullptr;
    std::optional<PictureTypeParser> parser;
    double next_time = 0;
    while (av_read_frame(&format, packet.get()) >= 0) {
        if (video == nullptr) {
            // Here, not on opening: an MPEG program stream makes its streams
            // as their first packets come.
            video = only_the_video(format);
            if (video != nullptr) {
                parser.emplace(*video);
            }
        }
        if (video != nullptr && packet->stream_index == video->index) {
            const double seconds_per_tick = av_q2d(video->time_base);
            const double time = packet_time(*packet, seconds_per_tick, next_time);
            next_time = std::max(next_time, time + static_cast<double>(packet->duration) * seconds_per_tick);
            frames.push_back({time, packet->pos, packet->size,
                              frame_type_of(parser->picture_type(*packet), (packet->flags & AV_PKT_FLAG_KEY) != 0)});
        }
        av_packet_unref(packet.get());
    }
    return frames;
}

}  // namespace

FrameTypeReader::FrameTypeReader(VideoInfo info, const CancelToken& cancel)
    : info_(std::move(info)), cancel_(cancel), thread_([this] {
          try {
              if (!load_ffmpeg_libraries().has_value()) {
                  if (const auto format = open_file(info_, *this)) {
                      frames_ = read_frames(*format);
                  }
              }
          } catch (const std::exception& error) {  // out of memory: the graph goes without the types
              frames_.clear();
              app_log::write(app_log::Level::warning, "vml.bitrate", error.what());
          }
      }) {}

std::vector<std::uint8_t> FrameTypeReader::types_of(const BitrateData& data) {
    thread_.join();
    if (cancel_.cancelled()) {
        return {};
    }
    std::vector<std::uint8_t> types = types_matching_frames(std::move(frames_), info_.is_raw_stream(), data.sizes);
    if (types.empty()) {
        app_log::write(app_log::Level::info, "vml.bitrate",
                       platform::path_text(info_.path.filename()) +
                           ": frame types not shown (its packets read differently from ffprobe's)");
    }
    return types;
}

#else

FrameTypeReader::FrameTypeReader(VideoInfo info, const CancelToken& cancel) : info_(std::move(info)), cancel_(cancel) {}

std::vector<std::uint8_t> FrameTypeReader::types_of(const BitrateData& /*data*/) { return {}; }

#endif

FrameTypeReader::~FrameTypeReader() {
    stop_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
}

}  // namespace vml::app
