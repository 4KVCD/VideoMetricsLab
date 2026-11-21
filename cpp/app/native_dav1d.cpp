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

#include "av2/display_order.h"
#include "dav1d_api.h"
#include "ffmpeg_api.h"

#include "vml/gpu_frames.h"
#include "vml/platform/library.h"
#include "vml/splot_library.h"

namespace vml::app {

namespace {

// dav1d's functions, from its library, loaded once for the life of the
// process; or why they could not be.
struct Dav1dLibrary {
    std::optional<platform::DynamicLibrary> library;
    QString problem;
    unsigned (*version_api)() = nullptr;
    void (*default_settings)(Dav1dSettings*) = nullptr;
    int (*open)(void**, const Dav1dSettings*) = nullptr;
    int (*send_data)(void*, Dav1dData*) = nullptr;
    int (*get_picture)(void*, Dav1dPicture*) = nullptr;
    void (*flush)(void*) = nullptr;
    void (*close)(void**) = nullptr;
    std::uint8_t* (*data_create)(Dav1dData*, std::size_t) = nullptr;
    void (*data_unref)(Dav1dData*) = nullptr;
    void (*picture_unref)(Dav1dPicture*) = nullptr;

    // Loads the library at `path` and binds its functions, checking its API's
    // major version.
    void load(const std::filesystem::path& path) {
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(path, ignored)) {
            problem = QStringLiteral("dav1d is not bundled");
            return;
        }
        try {
            const platform::DynamicLibrary& loaded = library.emplace(path);
            const auto bind = [&](auto& target, const char* function) {
                target = loaded.function<std::remove_reference_t<decltype(target)>>(function);
            };
            bind(version_api, "dav1d_version_api");
            bind(default_settings, "dav1d_default_settings");
            bind(open, "dav1d_open");
            bind(send_data, "dav1d_send_data");
            bind(get_picture, "dav1d_get_picture");
            bind(flush, "dav1d_flush");
            bind(close, "dav1d_close");
            bind(data_create, "dav1d_data_create");
            bind(data_unref, "dav1d_data_unref");
            bind(picture_unref, "dav1d_picture_unref");
        } catch (const platform::LibraryError& error) {
            problem = QStringLiteral("dav1d could not be loaded: %1").arg(QString::fromUtf8(error.what()));
            library.reset();
            return;
        }
        if (version_api() >> 16 != kDav1dApiMajor) {
            problem = QStringLiteral("dav1d is not the version the app was made for");
            library.reset();
        }
    }
};

const Dav1dLibrary& dav1d() {
    static std::once_flag loaded;
    static Dav1dLibrary functions;
    std::call_once(loaded, [] { functions.load(dav1d_library_path()); });
    return functions;
}

// FFmpeg's pixel format for pictures of `layout` (I400 to I444, the same
// values in dav1d and splot) and `depth`.
AVPixelFormat pixel_format(int layout, int depth) {
    static_assert(kDav1dI400 == kSplotI400 && kDav1dI444 == kSplotI444, "dav1d's and splot's layouts");
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

// A dav1d decoder, as avcodec is asked.
struct Dav1dDecoder {
    const Dav1dLibrary& api;
    void* context = nullptr;
    // The sequence header, given before the first packet and again after
    // each seek, in case the packets after it do not repeat it.
    std::vector<std::uint8_t> sequence_header;
    bool header_due = true;
    // The packet last sent, as much of it as the decoder has not taken yet.
    Dav1dData pending{};
    // Whether the decoder has been asked for a picture since it was last
    // given data (see receive()).
    bool asked_since_data = false;
    // The picture last received, held until release().
    Dav1dPicture picture{};
    // Whether the end of the stream was sent.
    bool finished = false;
    AVColorSpace matrix = AVCOL_SPC_UNSPECIFIED;
    AVColorRange range = AVCOL_RANGE_UNSPECIFIED;
    AVColorPrimaries primaries = AVCOL_PRI_UNSPECIFIED;
    AVColorTransferCharacteristic transfer = AVCOL_TRC_UNSPECIFIED;

    Dav1dDecoder(const Dav1dLibrary& library, bool apply_grain) : api(library) {
        if (!api.library) {
            throw std::runtime_error(api.problem.toStdString());
        }
        Dav1dSettings settings{};
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
            throw std::runtime_error("dav1d could not start");
        }
    }

    ~Dav1dDecoder() {
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
        const int result = api.send_data(context, &pending);
        if (result < 0 && result != kDav1dAgain) {
            api.data_unref(&pending);
            return result;
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
            if (result == kDav1dAgain) {
                if (pending.sz > 0) {
                    continue;  // it has read what it held, and takes the rest of the packet now
                }
                if (!finished) {
                    return AVERROR(EAGAIN);
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
            if (format == AV_PIX_FMT_NONE) {
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
            frame->pts = picture.m.timestamp;
            frame->best_effort_timestamp = picture.m.timestamp;
            frame->colorspace = matrix;
            frame->color_range = range;
            frame->color_primaries = primaries;
            frame->color_trc = transfer;
            return 0;
        }
    }

    void release() {
        if (picture.ref != nullptr || picture.data[0] != nullptr) {
            api.picture_unref(&picture);
        }
        picture = Dav1dPicture{};
    }

    void flush() {
        release();
        if (pending.sz > 0) {
            api.data_unref(&pending);
        }
        pending = Dav1dData{};
        api.flush(context);
        header_due = true;
        asked_since_data = false;
        finished = false;
    }
};

// splot's error as FFmpeg's: a stream it cannot decode, or a wrong call.
int splot_error(int result) {
    return result == kSplotDecodeError ? AVERROR_INVALIDDATA : AVERROR(EINVAL);
}

// A splot decoder, as avcodec is asked. splot copies each temporal unit it
// is given into its own queue and gives its pictures in display order, timed
// here by the units fed (av2/display_order.h). Asked for a picture, it says
// EAGAIN at once while it wants units; with its queue full, or drained, it
// waits until it has decoded one.
struct SplotDecoder {
    const SplotCalls& api;
    const SplotSettings settings;
    void* context = nullptr;
    // The unit last sent while splot's queue was full: fed once a picture has
    // been taken.
    std::vector<std::uint8_t> pending;
    long long pending_stamp = 0;
    bool has_pending = false;
    // The picture last received, held until release().
    SplotPicture picture{};
    bool holding = false;
    // The stamps of the units splot took, to time its pictures by.
    Av2DisplayOrder display_order;
    // Whether the end of the stream was sent, and passed on to splot.
    bool finished = false;
    bool drained = false;

    SplotDecoder(const SplotLibrary& library, bool apply_grain)
        : api(library.calls), settings(library.library ? splot_settings(library.calls, apply_grain) : SplotSettings{}) {
        if (!library.library) {
            throw std::runtime_error(library.problem);
        }
        if (api.open(&context, &settings) != kSplotOk) {
            throw std::runtime_error("splot could not start");
        }
    }

    ~SplotDecoder() {
        release();
        api.close(&context);
    }

    int send(const AVPacket* packet) {
        if (packet == nullptr) {
            finished = true;
            return 0;
        }
        if (has_pending) {
            // Never so once receive() has said EAGAIN: it says so only once
            // splot has taken the unit before.
            return AVERROR(EAGAIN);
        }
        pending.assign(packet->data, packet->data + packet->size);
        pending_stamp = packet->pts != AV_NOPTS_VALUE ? packet->pts : packet->dts;
        has_pending = true;
        return feed();
    }

    // Gives splot the unit last sent, if it has not taken it yet.
    int feed() {
        if (!has_pending || context == nullptr) {
            return 0;
        }
        const SplotDataProps props{pending_stamp, 0, 0};
        const int result = api.send_data(context, pending.data(), pending.size(), &props);
        if (result == kSplotAgain) {
            return 0;  // its queue is full: taken once a picture has been
        }
        has_pending = false;
        if (result != kSplotOk) {
            return splot_error(result);
        }
        display_order.fed(pending_stamp);  // taken: its picture will come
        return 0;
    }

    int receive(AVFrame* frame) {
        release();
        if (context == nullptr) {
            return AVERROR(EINVAL);  // it did not start again after a seek
        }
        while (true) {
            if (const int fed = feed(); fed < 0) {
                return fed;
            }
            if (finished && !has_pending && !drained) {
                api.drain(context);  // the end: it gives every picture it holds
                drained = true;
            }
            const int result = api.get_picture(context, &picture);
            if (result == kSplotAgain) {
                if (has_pending) {
                    continue;  // room again in its queue: the unit goes in now
                }
                return AVERROR(EAGAIN);
            }
            if (result == kSplotEof) {
                // splot has given every picture: one for each unit, or the
                // pictures shown were timed wrongly.
                return display_order.empty() ? AVERROR_EOF : AVERROR_INVALIDDATA;
            }
            if (result != kSplotOk) {
                return splot_error(result);
            }
            holding = true;
            const AVPixelFormat format = pixel_format(picture.layout, static_cast<int>(picture.bit_depth));
            const std::optional<long long> time = display_order.shown();
            if (format == AV_PIX_FMT_NONE || !time) {
                release();
                return AVERROR_INVALIDDATA;
            }
            frame->format = format;
            frame->width = static_cast<int>(picture.width);
            frame->height = static_cast<int>(picture.height);
            const int planes = picture.layout == kSplotI400 ? 1 : 3;
            for (int plane = 0; plane < planes; ++plane) {
                frame->data[plane] = const_cast<std::uint8_t*>(splot_visible_plane(picture, plane));
                frame->linesize[plane] = static_cast<int>(picture.stride[plane]);
            }
            frame->pts = *time;
            frame->best_effort_timestamp = *time;
            frame->colorspace = AVCOL_SPC_UNSPECIFIED;
            frame->color_range = AVCOL_RANGE_UNSPECIFIED;
            frame->color_primaries = AVCOL_PRI_UNSPECIFIED;
            frame->color_trc = AVCOL_TRC_UNSPECIFIED;
            if (splot_says_colour(picture.color)) {
                frame->color_primaries = static_cast<AVColorPrimaries>(picture.color.primaries);  // H.273's codes are FFmpeg's
                frame->color_trc = static_cast<AVColorTransferCharacteristic>(picture.color.transfer);
                frame->colorspace = static_cast<AVColorSpace>(picture.color.matrix);
                frame->color_range = picture.color.full_range ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
            }
            return 0;
        }
    }

    void release() {
        if (holding) {
            api.picture_unref(&picture);
            holding = false;
        }
        picture = SplotPicture{};
    }

    // splot has no flush: a seek starts it again, with the same settings.
    // Where it does not start, receive() says so.
    void flush() {
        release();
        api.close(&context);
        if (api.open(&context, &settings) != kSplotOk) {
            context = nullptr;
        }
        pending.clear();
        has_pending = false;
        display_order.clear();
        finished = false;
        drained = false;
    }
};

}  // namespace

struct Dav1dVideo::State : Dav1dDecoder {
    using Dav1dDecoder::Dav1dDecoder;
};

std::optional<QString> Dav1dVideo::unavailable() {
    return dav1d().library ? std::nullopt : std::optional(dav1d().problem);
}

Dav1dVideo::Dav1dVideo(const AVCodecParameters& stream, bool apply_grain)
    : state_(std::make_unique<State>(dav1d(), apply_grain)) {
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

struct SplotVideo::State : SplotDecoder {
    using SplotDecoder::SplotDecoder;
};

std::optional<QString> SplotVideo::unavailable() {
    const SplotLibrary& splot = splot_library();
    return splot.library ? std::nullopt : std::optional(QString::fromStdString(splot.problem));
}

SplotVideo::SplotVideo(bool apply_grain) : state_(std::make_unique<State>(splot_library(), apply_grain)) {}

SplotVideo::~SplotVideo() = default;
int SplotVideo::send(const AVPacket* packet) { return state_->send(packet); }
int SplotVideo::receive(AVFrame* frame) { return state_->receive(frame); }
void SplotVideo::release() { state_->release(); }
void SplotVideo::flush() { state_->flush(); }

}  // namespace vml::app
