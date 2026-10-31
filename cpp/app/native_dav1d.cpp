#include "native_dav1d.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "dav1d_api.h"
#include "dav2d/dav2d_api.h"
#include "dav2d/display_order.h"
#include "ffmpeg_api.h"

#include "vml/gpu_frames.h"
#include "vml/platform/library.h"

namespace vml::app {

namespace {

// dav1d's or dav2d's functions, from its library, loaded once for the life
// of the process; or why they could not be. One API under two names.
template <typename SettingsT, typename DataT, typename PictureT>
struct DavLibrary {
    using Settings = SettingsT;
    using Data = DataT;
    using Picture = PictureT;
    std::optional<platform::DynamicLibrary> library;
    QString problem;
    unsigned (*version_api)() = nullptr;
    void (*default_settings)(Settings*) = nullptr;
    int (*open)(void**, const Settings*) = nullptr;
    int (*send_data)(void*, Data*) = nullptr;
    int (*get_picture)(void*, Picture*) = nullptr;
    void (*flush)(void*) = nullptr;
    void (*close)(void**) = nullptr;
    std::uint8_t* (*data_create)(Data*, std::size_t) = nullptr;
    void (*data_unref)(Data*) = nullptr;
    void (*picture_unref)(Picture*) = nullptr;

    // Loads the library at `path` and binds its functions by `prefix`
    // ("dav1d", "dav2d"), checking its API's major version.
    void load(const std::filesystem::path& path, const std::string& prefix, unsigned api_major) {
        const QString name = QString::fromStdString(prefix);
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(path, ignored)) {
            problem = QStringLiteral("%1 is not bundled").arg(name);
            return;
        }
        try {
            const platform::DynamicLibrary& loaded = library.emplace(path);
            const auto bind = [&](auto& target, const char* function) {
                target = loaded.function<std::remove_reference_t<decltype(target)>>((prefix + function).c_str());
            };
            bind(version_api, "_version_api");
            bind(default_settings, "_default_settings");
            bind(open, "_open");
            bind(send_data, "_send_data");
            bind(get_picture, "_get_picture");
            bind(flush, "_flush");
            bind(close, "_close");
            bind(data_create, "_data_create");
            bind(data_unref, "_data_unref");
            bind(picture_unref, "_picture_unref");
        } catch (const platform::LibraryError& error) {
            problem = QStringLiteral("%1 could not be loaded: %2").arg(name, QString::fromUtf8(error.what()));
            library.reset();
            return;
        }
        if (version_api() >> 16 != api_major) {
            problem = QStringLiteral("%1 is not the version the app was made for").arg(name);
            library.reset();
        }
    }
};
using Dav1dLibrary = DavLibrary<Dav1dSettings, Dav1dData, Dav1dPicture>;
using Dav2dLibrary = DavLibrary<Dav2dSettings, Dav2dData, Dav2dPicture>;

const Dav1dLibrary& dav1d() {
    static std::once_flag loaded;
    static Dav1dLibrary functions;
    std::call_once(loaded, [] { functions.load(dav1d_library_path(), "dav1d", kDav1dApiMajor); });
    return functions;
}

const Dav2dLibrary& dav2d() {
    static std::once_flag loaded;
    static Dav2dLibrary functions;
    std::call_once(loaded, [] { functions.load(dav2d_library_path(), "dav2d", kDav2dApiMajor); });
    return functions;
}

// FFmpeg's pixel format for pictures of `layout` (I400 to I444, the same
// values in both) and `depth`.
AVPixelFormat pixel_format(int layout, int depth) {
    const int index = depth == 8 ? 0 : depth == 10 ? 1 : depth == 12 ? 2 : -1;
    if (index < 0) {
        return AV_PIX_FMT_NONE;
    }
    static constexpr AVPixelFormat formats[4][3] = {
        {AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY10LE, AV_PIX_FMT_GRAY12LE},
        {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P12LE},
        {AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV422P12LE},
        {AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P12LE},
    };
    return layout >= kDav1dI400 && layout <= kDav1dI444 ? formats[layout][index] : AV_PIX_FMT_NONE;
}

// The colours a picture says (AV2's content interpretation), where it says
// them; dav1d's pictures say none the stream did not (theirs are given).
void take_picture_colour(const Dav1dPicture&, AVFrame*) {}
void take_picture_colour(const Dav2dPicture& picture, AVFrame* frame) {
    if (picture.content == nullptr || !picture.content->color_description_present) {
        return;
    }
    int codes[3];
    dav2d_colour_codes(*picture.content, codes);
    frame->color_primaries = static_cast<AVColorPrimaries>(codes[0]);  // H.273's codes are FFmpeg's
    frame->color_trc = static_cast<AVColorTransferCharacteristic>(codes[1]);
    frame->colorspace = static_cast<AVColorSpace>(codes[2]);
    frame->color_range = picture.content->color.range ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
}

// When a picture is shown: dav1d's carries its own time; dav2d's, in display
// order, are timed by the packets given (dav2d/display_order.h).
std::optional<long long> picture_time(const Dav1dPicture& picture, Dav2dDisplayOrder&) {
    return picture.m.timestamp;
}
std::optional<long long> picture_time(const Dav2dPicture&, Dav2dDisplayOrder& order) { return order.shown(); }

// One decoder of either library, as avcodec is asked.
template <typename Library>
struct DavDecoder {
    static constexpr bool av2 = std::is_same_v<Library, Dav2dLibrary>;
    const Library& api;
    void* context = nullptr;
    // AV1's sequence header, given before the first packet and again after
    // each seek, in case the packets after it do not repeat it (none for AV2).
    std::vector<std::uint8_t> sequence_header;
    bool header_due = true;
    // The packet last sent, as much of it as the decoder has not taken yet.
    typename Library::Data pending{};
    // Whether the decoder has been asked for a picture since it was last
    // given data (see receive()).
    bool asked_since_data = false;
    // The picture last received, held until release().
    typename Library::Picture picture{};
    // AV2: the times of the packets given, to time its pictures by.
    Dav2dDisplayOrder display_order;
    // Whether the end of the stream was sent, and (AV2) passed on to dav2d.
    bool finished = false;
    bool end_given = false;
    AVColorSpace matrix = AVCOL_SPC_UNSPECIFIED;
    AVColorRange range = AVCOL_RANGE_UNSPECIFIED;
    AVColorPrimaries primaries = AVCOL_PRI_UNSPECIFIED;
    AVColorTransferCharacteristic transfer = AVCOL_TRC_UNSPECIFIED;

    DavDecoder(const Library& library, const char* name, bool apply_grain) : api(library) {
        if (!api.library) {
            throw std::runtime_error(api.problem.toStdString());
        }
        typename Library::Settings settings{};
        api.default_settings(&settings);
        // As FFmpeg's libdav1d decoder sets it, and the software decoder: the
        // operating point's highest spatial layer only, film grain applied as
        // the video is scored; its own threads, as many as the CPU has. No
        // log: there is no console.
        settings.all_layers = 0;
        settings.n_threads = 0;
        settings.max_frame_delay = 0;
        settings.apply_grain = apply_grain ? 1 : 0;
        settings.logger_callback = nullptr;
        if (api.open(&context, &settings) < 0) {
            throw std::runtime_error(std::string(name) + " could not start");
        }
    }

    ~DavDecoder() {
        release();
        if (pending.sz > 0) {
            api.data_unref(&pending);
        }
        api.close(&context);
    }

    int send(const AVPacket* packet) {
        if (packet == nullptr) {
            finished = true;
            return 0;
        }
        if (pending.sz > 0) {
            // Never so once receive() has said EAGAIN: it says so only when the
            // decoder has taken all of the packet before.
            return AVERROR(EAGAIN);
        }
        const std::size_t header = header_due ? sequence_header.size() : 0;
        std::uint8_t* buffer = api.data_create(&pending, header + static_cast<std::size_t>(packet->size));
        if (buffer == nullptr) {
            return AVERROR(ENOMEM);
        }
        if (header > 0) {
            std::memcpy(buffer, sequence_header.data(), header);
        }
        std::memcpy(buffer + header, packet->data, static_cast<std::size_t>(packet->size));
        header_due = false;
        pending.m.timestamp = packet->pts != AV_NOPTS_VALUE ? packet->pts : packet->dts;
        return feed();
    }

    // Gives the decoder what it has not taken of the packet last sent.
    int feed() {
        if (pending.sz == 0) {
            return 0;
        }
        // EAGAIN: it still holds some of the packet before, and takes this one
        // once a picture has been taken.
        const long long stamp = pending.m.timestamp;
        const int result = api.send_data(context, &pending);
        if (result < 0 && result != kDav1dAgain) {
            api.data_unref(&pending);
            return result;
        }
        if constexpr (av2) {
            if (result == 0) {
                display_order.fed(stamp);  // taken: its picture will come
            }
        }
        asked_since_data = false;
        return 0;
    }

    int receive(AVFrame* frame) {
        release();
        while (true) {
            if (const int fed = feed(); fed < 0) {
                return fed;
            }
            // Asked for a picture once per packet: asked again straight away,
            // the decoder drains, waiting for each frame its threads are still
            // decoding. Only at the end is that wanted.
            if (asked_since_data && !finished) {
                return AVERROR(EAGAIN);
            }
            const bool draining = asked_since_data;
            const int result = api.get_picture(context, &picture);
            asked_since_data = true;
            if (result == kDav2dEof) {
                // dav2d has given every picture it held: one for each unit, or
                // the pictures shown were timed wrongly.
                return display_order.empty() ? AVERROR_EOF : AVERROR_INVALIDDATA;
            }
            if (result == kDav1dAgain) {
                if (pending.sz > 0) {
                    continue;  // it has read what it held, and takes the rest of the packet now
                }
                if (!finished) {
                    return AVERROR(EAGAIN);
                }
                if constexpr (av2) {
                    // dav2d gives the pictures it holds at the end once told it
                    // is the end, and only once it has parsed every unit (which
                    // it has when it wants more): told before, it would give
                    // them before the last unit's picture.
                    if (!end_given) {
                        api.send_data(context, nullptr);
                        end_given = true;
                        continue;
                    }
                }
                if (draining) {
                    return AVERROR_EOF;
                }
                continue;  // at the end: asked again, it drains
            }
            if (result < 0) {
                return result;
            }
            const AVPixelFormat format = pixel_format(picture.p.layout, picture.p.bpc);
            const std::optional<long long> time = picture_time(picture, display_order);
            if (format == AV_PIX_FMT_NONE || !time) {
                release();
                return AVERROR_INVALIDDATA;
            }
            frame->format = format;
            frame->width = picture.p.w;
            frame->height = picture.p.h;
            const int planes = picture.p.layout == kDav1dI400 ? 1 : 3;
            for (int plane = 0; plane < planes; ++plane) {
                frame->data[plane] = static_cast<std::uint8_t*>(picture.data[plane]);
                frame->linesize[plane] = static_cast<int>(picture.stride[plane == 0 ? 0 : 1]);
            }
            frame->pts = *time;
            frame->best_effort_timestamp = *time;
            frame->colorspace = matrix;
            frame->color_range = range;
            frame->color_primaries = primaries;
            frame->color_trc = transfer;
            take_picture_colour(picture, frame);
            return 0;
        }
    }

    void release() {
        if (picture.ref != nullptr || picture.data[0] != nullptr) {
            api.picture_unref(&picture);
        }
        picture = typename Library::Picture{};
    }

    void flush() {
        release();
        if (pending.sz > 0) {
            api.data_unref(&pending);
        }
        pending = typename Library::Data{};
        api.flush(context);
        display_order.clear();
        header_due = true;
        asked_since_data = false;
        finished = false;
        end_given = false;
    }
};

}  // namespace

struct Dav1dVideo::State : DavDecoder<Dav1dLibrary> {
    using DavDecoder::DavDecoder;
};

std::optional<QString> Dav1dVideo::unavailable() {
    return dav1d().library ? std::nullopt : std::optional(dav1d().problem);
}

Dav1dVideo::Dav1dVideo(const AVCodecParameters& stream, bool apply_grain)
    : state_(std::make_unique<State>(dav1d(), "dav1d", apply_grain)) {
    state_->matrix = stream.color_space;
    state_->range = stream.color_range;
    state_->primaries = stream.color_primaries;
    state_->transfer = stream.color_trc;
    // The configuration (an av1C record) holds the sequence header after its
    // own 4 bytes, as the GPU metrics' decoders are given it.
    if (stream.extradata_size > 4 && (stream.extradata[0] & 0x80) != 0) {
        state_->sequence_header.assign(stream.extradata + 4, stream.extradata + stream.extradata_size);
    }
}

Dav1dVideo::~Dav1dVideo() = default;
int Dav1dVideo::send(const AVPacket* packet) { return state_->send(packet); }
int Dav1dVideo::receive(AVFrame* frame) { return state_->receive(frame); }
void Dav1dVideo::release() { state_->release(); }
void Dav1dVideo::flush() { state_->flush(); }

struct Dav2dVideo::State : DavDecoder<Dav2dLibrary> {
    using DavDecoder::DavDecoder;
};

std::optional<QString> Dav2dVideo::unavailable() {
    return dav2d().library ? std::nullopt : std::optional(dav2d().problem);
}

Dav2dVideo::Dav2dVideo(bool apply_grain) : state_(std::make_unique<State>(dav2d(), "dav2d", apply_grain)) {}

Dav2dVideo::~Dav2dVideo() = default;
int Dav2dVideo::send(const AVPacket* packet) { return state_->send(packet); }
int Dav2dVideo::receive(AVFrame* frame) { return state_->receive(frame); }
void Dav2dVideo::release() { state_->release(); }
void Dav2dVideo::flush() { state_->flush(); }

}  // namespace vml::app
