#include "native_dav1d.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "dav1d_api.h"
#include "ffmpeg_api.h"

#include "vml/gpu_frames.h"
#include "vml/platform/library.h"

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
};

const Dav1dLibrary& dav1d() {
    static std::once_flag loaded;
    static Dav1dLibrary functions;
    std::call_once(loaded, [] {
        const std::filesystem::path path = dav1d_library_path();
        std::error_code ignored;
        if (!std::filesystem::is_regular_file(path, ignored)) {
            functions.problem = QStringLiteral("dav1d is not bundled");
            return;
        }
        try {
            const platform::DynamicLibrary& library = functions.library.emplace(path);
            functions.version_api = library.function<decltype(functions.version_api)>("dav1d_version_api");
            functions.default_settings =
                library.function<decltype(functions.default_settings)>("dav1d_default_settings");
            functions.open = library.function<decltype(functions.open)>("dav1d_open");
            functions.send_data = library.function<decltype(functions.send_data)>("dav1d_send_data");
            functions.get_picture = library.function<decltype(functions.get_picture)>("dav1d_get_picture");
            functions.flush = library.function<decltype(functions.flush)>("dav1d_flush");
            functions.close = library.function<decltype(functions.close)>("dav1d_close");
            functions.data_create = library.function<decltype(functions.data_create)>("dav1d_data_create");
            functions.data_unref = library.function<decltype(functions.data_unref)>("dav1d_data_unref");
            functions.picture_unref = library.function<decltype(functions.picture_unref)>("dav1d_picture_unref");
        } catch (const platform::LibraryError& error) {
            functions.problem = QStringLiteral("dav1d could not be loaded: %1").arg(QString::fromUtf8(error.what()));
            functions.library.reset();
            return;
        }
        if (functions.version_api() >> 16 != kDav1dApiMajor) {
            functions.problem = QStringLiteral("dav1d is not the version the app was made for");
            functions.library.reset();
        }
    });
    return functions;
}

// FFmpeg's pixel format for dav1d's pictures of `layout` and `depth`.
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

}  // namespace

struct Dav1dVideo::State {
    void* context = nullptr;
    // The stream's sequence header, given before the first packet and again
    // after each seek, in case the packets after it do not repeat it.
    std::vector<std::uint8_t> sequence_header;
    bool header_due = true;
    // The packet last sent, as much of it as dav1d has not taken yet.
    Dav1dData pending{};
    // Whether dav1d has been asked for a picture since it was last given
    // data (see receive()).
    bool asked_since_data = false;
    // The picture last received, held until release().
    Dav1dPicture picture{};
    // Whether the end of the stream was sent.
    bool finished = false;
    AVColorSpace matrix = AVCOL_SPC_UNSPECIFIED;
    AVColorRange range = AVCOL_RANGE_UNSPECIFIED;
    AVColorPrimaries primaries = AVCOL_PRI_UNSPECIFIED;
    AVColorTransferCharacteristic transfer = AVCOL_TRC_UNSPECIFIED;

    // Gives dav1d what it has not taken of the packet last sent.
    int feed();
};

std::optional<QString> Dav1dVideo::unavailable() {
    const Dav1dLibrary& functions = dav1d();
    if (!functions.library) {
        return functions.problem;
    }
    return std::nullopt;
}

Dav1dVideo::Dav1dVideo(const AVCodecParameters& stream) : state_(std::make_unique<State>()) {
    if (const std::optional<QString> problem = unavailable()) {
        throw std::runtime_error(problem->toStdString());
    }
    state_->matrix = stream.color_space;
    state_->range = stream.color_range;
    state_->primaries = stream.color_primaries;
    state_->transfer = stream.color_trc;
    // The configuration (an av1C record) holds the sequence header after its
    // own 4 bytes, as the GPU metrics' decoders are given it.
    if (stream.extradata_size > 4 && (stream.extradata[0] & 0x80) != 0) {
        state_->sequence_header.assign(stream.extradata + 4, stream.extradata + stream.extradata_size);
    }
    Dav1dSettings settings{};
    dav1d().default_settings(&settings);
    // As FFmpeg's libdav1d decoder sets it, and the software decoder: the
    // operating point's highest spatial layer only, film grain applied; its
    // own threads, as many as the CPU has. No log: there is no console.
    settings.all_layers = 0;
    settings.n_threads = 0;
    settings.max_frame_delay = 0;
    settings.logger_callback = nullptr;
    if (dav1d().open(&state_->context, &settings) < 0) {
        throw std::runtime_error("dav1d could not start");
    }
}

Dav1dVideo::~Dav1dVideo() {
    release();
    if (state_->pending.sz > 0) {
        dav1d().data_unref(&state_->pending);
    }
    dav1d().close(&state_->context);
}

int Dav1dVideo::send(const AVPacket* packet) {
    State& s = *state_;
    if (packet == nullptr) {
        s.finished = true;
        return 0;
    }
    if (s.pending.sz > 0) {
        // Never so once receive() has said EAGAIN: it says so only when dav1d
        // has taken all of the packet before.
        return AVERROR(EAGAIN);
    }
    const std::size_t header = s.header_due ? s.sequence_header.size() : 0;
    std::uint8_t* buffer = dav1d().data_create(&s.pending, header + static_cast<std::size_t>(packet->size));
    if (buffer == nullptr) {
        return AVERROR(ENOMEM);
    }
    if (header > 0) {
        std::memcpy(buffer, s.sequence_header.data(), header);
    }
    std::memcpy(buffer + header, packet->data, static_cast<std::size_t>(packet->size));
    s.header_due = false;
    s.pending.m.timestamp = packet->pts != AV_NOPTS_VALUE ? packet->pts : packet->dts;
    return s.feed();
}

int Dav1dVideo::State::feed() {
    if (pending.sz == 0) {
        return 0;
    }
    // EAGAIN: dav1d still holds some of the packet before, and takes this one
    // once a picture has been taken.
    const int result = dav1d().send_data(context, &pending);
    if (result < 0 && result != kDav1dAgain) {
        dav1d().data_unref(&pending);
        return result;
    }
    asked_since_data = false;
    return 0;
}

int Dav1dVideo::receive(AVFrame* frame) {
    release();
    State& s = *state_;
    while (true) {
        if (const int fed = s.feed(); fed < 0) {
            return fed;
        }
        // dav1d is asked for a picture once per packet: asked again straight
        // away, it drains, waiting for each frame its threads are still
        // decoding. Only at the end is that wanted.
        if (s.asked_since_data && !s.finished) {
            return AVERROR(EAGAIN);
        }
        const bool draining = s.asked_since_data;
        const int result = dav1d().get_picture(s.context, &s.picture);
        s.asked_since_data = true;
        if (result == kDav1dAgain) {
            if (s.pending.sz > 0) {
                continue;  // it has read what it held, and takes the rest of the packet now
            }
            if (!s.finished) {
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
        const Dav1dPicture& picture = s.picture;
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
        frame->colorspace = s.matrix;
        frame->color_range = s.range;
        frame->color_primaries = s.primaries;
        frame->color_trc = s.transfer;
        return 0;
    }
}

void Dav1dVideo::release() {
    if (state_->picture.ref != nullptr || state_->picture.data[0] != nullptr) {
        dav1d().picture_unref(&state_->picture);
    }
    state_->picture = Dav1dPicture{};
}

void Dav1dVideo::flush() {
    release();
    State& s = *state_;
    if (s.pending.sz > 0) {
        dav1d().data_unref(&s.pending);
    }
    s.pending = Dav1dData{};
    dav1d().flush(s.context);
    s.header_due = true;
    s.asked_since_data = false;
    s.finished = false;
}

}  // namespace vml::app
