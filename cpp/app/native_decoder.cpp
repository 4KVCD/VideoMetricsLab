#include "native_decoder.h"

#include <d3d11.h>
#include <wrl/client.h>

#include <QStringList>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "ffmpeg_api.h"
#include "native_dav1d.h"
#include "native_device.h"
#include "vml/platform/system.h"

using Microsoft::WRL::ComPtr;

namespace vml::app {

namespace {

// Frames decoded and not yet taken: GStreamer's appsink held three.
constexpr std::size_t queued_frames = 3;
// Decoder surfaces beyond the ones the decoder itself refers to: the frames
// queued here and in the pool (five per video), the pair shown and the
// presenter's two waiting.
constexpr int held_surfaces = 12;

QString error_text(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return QString::fromUtf8(text.data());
}

// FFmpeg's choice of pictures: Direct3D 11's, or none (the decoder then
// fails, and the video is decoded in software).
AVPixelFormat direct3d_format(AVCodecContext* /*context*/, const AVPixelFormat* formats) {
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_D3D11) {
            return *format;
        }
    }
    return AV_PIX_FMT_NONE;
}

// Whether FFmpeg decodes `codec` with Direct3D 11's video decoding.
bool decodes_on_gpu(const AVCodec* codec) {
    for (int index = 0;; ++index) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(codec, index);
        if (config == nullptr) {
            return false;
        }
        if (config->device_type == AV_HWDEVICE_TYPE_D3D11VA && config->pix_fmt == AV_PIX_FMT_D3D11 &&
            (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0) {
            return true;
        }
    }
}

// Textures for pictures decoded in software, reused once let go.
class UploadTextures : public std::enable_shared_from_this<UploadTextures> {
public:
    UploadTextures(ComPtr<ID3D11Device> device, UINT bind = D3D11_BIND_SHADER_RESOURCE)
        : device_(std::move(device)), bind_(bind) {}

    // A texture of `width` x `height` in `format`, held until the returned
    // pointer goes.
    std::shared_ptr<ID3D11Texture2D> take(int width, int height, DXGI_FORMAT format) {
        ComPtr<ID3D11Texture2D> texture;
        {
            const std::lock_guard lock(mutex_);
            if (width != width_ || height != height_ || format != format_) {
                spare_.clear();
                width_ = width;
                height_ = height;
                format_ = format;
            }
            if (!spare_.empty()) {
                texture = std::move(spare_.back());
                spare_.pop_back();
            }
        }
        if (!texture) {
            D3D11_TEXTURE2D_DESC description{};
            description.Width = static_cast<UINT>(width);
            description.Height = static_cast<UINT>(height);
            description.MipLevels = 1;
            description.ArraySize = 1;
            description.Format = format;
            description.SampleDesc.Count = 1;
            description.Usage = D3D11_USAGE_DEFAULT;
            description.BindFlags = bind_;
            if (FAILED(device_->CreateTexture2D(&description, nullptr, &texture))) {
                return nullptr;
            }
        }
        ID3D11Texture2D* raw = texture.Detach();
        return {raw, [owner = shared_from_this(), width, height, format](ID3D11Texture2D* given) {
                    owner->give_back(given, width, height, format);
                }};
    }

private:
    void give_back(ID3D11Texture2D* texture, int width, int height, DXGI_FORMAT format) {
        ComPtr<ID3D11Texture2D> owned;
        owned.Attach(texture);
        const std::lock_guard lock(mutex_);
        if (width == width_ && height == height_ && format == format_ && spare_.size() < 4) {
            spare_.push_back(std::move(owned));
        }
    }

    ComPtr<ID3D11Device> device_;
    UINT bind_;
    std::mutex mutex_;
    std::vector<ComPtr<ID3D11Texture2D>> spare_;
    int width_ = 0;
    int height_ = 0;
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN;
};

// Scales pictures down with Direct3D 11's video processor, cropped first,
// keeping their format and colours, as GStreamer's converter scaled the
// source down to the test video's size.
class Downscaler {
public:
    // Pictures `crop` of an `input_width` x `input_height` texture, to
    // `output` in `format`; throws std::runtime_error when the GPU cannot.
    // Its colour space is left as it is, the same in and out, so that only
    // the size changes: told HDR10's on both sides, NVIDIA's processor
    // (driver 616.92) wrote no picture at all.
    Downscaler(ID3D11Device* device, ID3D11DeviceContext* context, int input_width, int input_height, RECT crop,
               Size output, DXGI_FORMAT format)
        : crop_(crop), output_(output) {
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&video_device_))) ||
            FAILED(context->QueryInterface(IID_PPV_ARGS(&video_context_)))) {
            throw std::runtime_error("no Direct3D 11 video processing");
        }
        D3D11_VIDEO_PROCESSOR_CONTENT_DESC content{};
        content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        content.InputWidth = static_cast<UINT>(input_width);
        content.InputHeight = static_cast<UINT>(input_height);
        content.OutputWidth = static_cast<UINT>(output.first);
        content.OutputHeight = static_cast<UINT>(output.second);
        content.InputFrameRate = {1, 1};
        content.OutputFrameRate = {1, 1};
        content.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
        if (FAILED(video_device_->CreateVideoProcessorEnumerator(&content, &enumerator_)) ||
            FAILED(video_device_->CreateVideoProcessor(enumerator_.Get(), 0, &processor_))) {
            throw std::runtime_error("no Direct3D 11 video processor for this scaling");
        }
        outputs_ = std::make_shared<UploadTextures>(ComPtr<ID3D11Device>(device),
                                                    D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
        format_ = format;
    }

    // `texture`'s picture `subresource`, scaled; with the device's lock held.
    std::shared_ptr<ID3D11Texture2D> scale(ID3D11Texture2D* texture, unsigned subresource) {
        std::shared_ptr<ID3D11Texture2D> output = outputs_->take(output_.first, output_.second, format_);
        if (!output) {
            return nullptr;
        }
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view{};
        input_view.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        input_view.Texture2D.ArraySlice = subresource;
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view{};
        output_view.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        ComPtr<ID3D11VideoProcessorInputView> input;
        ComPtr<ID3D11VideoProcessorOutputView> target;
        if (FAILED(video_device_->CreateVideoProcessorInputView(texture, enumerator_.Get(), &input_view, &input)) ||
            FAILED(video_device_->CreateVideoProcessorOutputView(output.get(), enumerator_.Get(), &output_view,
                                                                  &target))) {
            return nullptr;
        }
        const RECT whole{0, 0, output_.first, output_.second};
        video_context_->VideoProcessorSetStreamFrameFormat(processor_.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
        video_context_->VideoProcessorSetStreamAutoProcessingMode(processor_.Get(), 0, FALSE);
        video_context_->VideoProcessorSetStreamSourceRect(processor_.Get(), 0, TRUE, &crop_);
        video_context_->VideoProcessorSetStreamDestRect(processor_.Get(), 0, TRUE, &whole);
        video_context_->VideoProcessorSetOutputTargetRect(processor_.Get(), TRUE, &whole);
        D3D11_VIDEO_PROCESSOR_STREAM stream{};
        stream.Enable = TRUE;
        stream.pInputSurface = input.Get();
        if (FAILED(video_context_->VideoProcessorBlt(processor_.Get(), target.Get(), 0, 1, &stream))) {
            return nullptr;
        }
        return output;
    }

private:
    ComPtr<ID3D11VideoDevice> video_device_;
    ComPtr<ID3D11VideoContext> video_context_;
    ComPtr<ID3D11VideoProcessorEnumerator> enumerator_;
    ComPtr<ID3D11VideoProcessor> processor_;
    std::shared_ptr<UploadTextures> outputs_;
    RECT crop_;
    Size output_;
    DXGI_FORMAT format_ = DXGI_FORMAT_NV12;
};

}  // namespace

// What one opening of the video holds: the file read, the decoder, and for
// software decoding the upload's staging texture and converter.
struct FfmpegNativeVideo::Decoding {
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    // AV1's decoder in software, in place of `codec`.
    std::unique_ptr<Dav1dVideo> dav1d;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* converter = nullptr;
    int stream = -1;
    AVRational time_base{};
    // One frame's time (ns), from the stream's frame rate.
    std::int64_t frame_ns = 0;
    bool hardware = false;
    std::string codec_name;
    ComPtr<ID3D11Texture2D> staging;
    std::shared_ptr<UploadTextures> uploads;
    std::unique_ptr<Downscaler> downscaler;
    // Whether the source could not be scaled down (it is then shown at its
    // own size).
    bool downscale_failed = false;
    // Whether the packets ran out, so that the decoder is being emptied.
    bool draining = false;

    Decoding() = default;
    Decoding(const Decoding&) = delete;
    Decoding& operator=(const Decoding&) = delete;
    ~Decoding() {
        sws_freeContext(converter);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec);
        avformat_close_input(&format);
    }

    // As avcodec_send_packet, avcodec_receive_frame and avcodec_flush_buffers,
    // with whichever decoder decodes.
    int send(const AVPacket* given) { return dav1d ? dav1d->send(given) : avcodec_send_packet(codec, given); }
    int receive(AVFrame* into) { return dav1d ? dav1d->receive(into) : avcodec_receive_frame(codec, into); }
    void flush() {
        if (dav1d) {
            dav1d->flush();
        } else {
            avcodec_flush_buffers(codec);
        }
    }
    // Lets the frame last received go.
    void unref_frame() {
        av_frame_unref(frame);
        if (dav1d) {
            dav1d->release();
        }
    }
};

FfmpegNativeVideo::FfmpegNativeVideo(std::shared_ptr<NativeDevice> device, FrameComparison comparison, FrameSide side,
                                     const PreviewColorSettings& colors, int position_ms, QObject* parent)
    : NativeVideo(parent), device_(std::move(device)), comparison_(std::move(comparison)), side_(side) {
    const VideoInfo& info = frame_video_info(comparison_, side_);
    shaded_ = native_shading_mode(info, colors);
    if (shaded_ != NativeShading::none) {
        shading_ = shading_parameters(*hdr_kind(info), info.color_primaries, shaded_ == NativeShading::hdr);
    }
    // The source played scaled down (source_playback_comparison): its crop
    // made the comparison's smaller size.
    if (side_ == FrameSide::source) {
        const Size size = comparison_dimensions(comparison_);
        const std::optional<CropBox>& crop = comparison_.source_crop;
        const int width = crop ? crop->w : info.width;
        const int height = crop ? crop->h : info.height;
        if (size.first > 0 && size.second > 0 && (size.first < width || size.second < height)) {
            scale_to_ = size;
        }
    }
    // Opened at the video's first frame, which says where frames are counted
    // from; then sought, as GStreamer prerolled and then sought.
    if (position_ms > 0) {
        wanted_ms_ = position_ms;
        seeks_asked_ = 1;
    }
}

FfmpegNativeVideo::~FfmpegNativeVideo() {
    cancel();
    wait();
}

void FfmpegNativeVideo::seek(int position_ms) {
    {
        const std::lock_guard lock(mutex_);
        wanted_ms_ = std::max(0, position_ms);
        ++seeks_asked_;
    }
    changed_.notify_all();
}

bool FfmpegNativeVideo::seeking() const {
    const std::lock_guard lock(mutex_);
    return seeks_made_ < seeks_asked_;
}

NativeFramePtr FfmpegNativeVideo::try_pull() {
    NativeFramePtr frame;
    {
        const std::lock_guard lock(mutex_);
        if (frames_.empty()) {
            return nullptr;
        }
        frame = std::move(frames_.front());
        frames_.pop_front();
    }
    changed_.notify_all();
    return frame;
}

NativeUpdate FfmpegNativeVideo::poll() {
    const std::lock_guard lock(mutex_);
    NativeUpdate update;
    update.status = std::exchange(status_, std::nullopt);
    update.error = std::exchange(error_, std::nullopt);
    update.ended = std::exchange(ended_, false);
    return update;
}

std::optional<int> FfmpegNativeVideo::first_frame_ms() const {
    const std::lock_guard lock(mutex_);
    return first_frame_ms_;
}

void FfmpegNativeVideo::cancel() {
    {
        const std::lock_guard lock(mutex_);
        cancelled_ = true;
    }
    changed_.notify_all();
}

bool FfmpegNativeVideo::wait_for_room() {
    std::unique_lock lock(mutex_);
    // A seek is made once the first frame says where the video starts:
    // until then the first frames are decoded as usual.
    const auto seek_due = [this] { return wanted_ms_.has_value() && first_frame_ms_.has_value(); };
    changed_.wait(lock, [&] { return cancelled_ || seek_due() || frames_.size() < queued_frames; });
    return !cancelled_ && !seek_due();
}

bool FfmpegNativeVideo::cancelled() const {
    const std::lock_guard lock(mutex_);
    return cancelled_;
}

int FfmpegNativeVideo::interrupted(void* self) { return static_cast<FfmpegNativeVideo*>(self)->cancelled() ? 1 : 0; }

void FfmpegNativeVideo::deliver(NativeFramePtr frame) {
    const std::lock_guard lock(mutex_);
    frames_.push_back(std::move(frame));
}

void FfmpegNativeVideo::fail(const QString& error) {
    const std::lock_guard lock(mutex_);
    error_ = error;
}

namespace {

// Opens `path` and its video's decoder: with Direct3D 11's video decoding on
// `device` (`hardware`), or FFmpeg's software decoder; its reading stops when
// `interrupt` says so. Returns nothing, with `error` set, when it cannot.
std::unique_ptr<FfmpegNativeVideo::Decoding> open_video(const std::filesystem::path& path, NativeDevice& device,
                                                         bool hardware, const AVIOInterruptCB& interrupt,
                                                         QString& error) {
    auto decoding = std::make_unique<FfmpegNativeVideo::Decoding>();
    decoding->format = avformat_alloc_context();
    if (decoding->format == nullptr) {
        error = QStringLiteral("Out of memory");
        return nullptr;
    }
    decoding->format->interrupt_callback = interrupt;
    const std::string name = platform::path_text(path);
    int result = avformat_open_input(&decoding->format, name.c_str(), nullptr, nullptr);
    if (result < 0) {
        error = QStringLiteral("Could not open the video: %1").arg(error_text(result));
        return nullptr;
    }
    result = avformat_find_stream_info(decoding->format, nullptr);
    if (result < 0) {
        error = QStringLiteral("Could not read the video: %1").arg(error_text(result));
        return nullptr;
    }
    const AVCodec* codec = nullptr;
    decoding->stream = av_find_best_stream(decoding->format, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (decoding->stream < 0 || codec == nullptr) {
        error = QStringLiteral("The bundled FFmpeg cannot decode this video");
        return nullptr;
    }
    // Only the video is read: the soundtrack plays on its own.
    for (unsigned index = 0; index < decoding->format->nb_streams; ++index) {
        if (static_cast<int>(index) != decoding->stream) {
            decoding->format->streams[index]->discard = AVDISCARD_ALL;
        }
    }
    const AVStream* stream = decoding->format->streams[decoding->stream];
    decoding->time_base = stream->time_base;
    const AVRational rate = stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate;
    decoding->frame_ns = rate.num > 0 ? av_rescale(1'000'000'000, rate.den, rate.num) : 41'666'667;
    decoding->codec_name = codec->name;
    if (hardware && !decodes_on_gpu(codec)) {
        error = QStringLiteral("no Direct3D 11 decoder for %1").arg(QString::fromUtf8(codec->name));
        return nullptr;
    }
    decoding->packet = av_packet_alloc();
    decoding->frame = av_frame_alloc();
    if (decoding->packet == nullptr || decoding->frame == nullptr) {
        error = QStringLiteral("Out of memory");
        return nullptr;
    }
    if (!hardware && codec->id == AV_CODEC_ID_AV1) {
        // FFmpeg's AV1 decoder only drives a GPU's: dav1d decodes it, as
        // GStreamer's dav1ddec did.
        try {
            decoding->dav1d = std::make_unique<Dav1dVideo>(*stream->codecpar);
        } catch (const std::exception& problem) {
            error = QStringLiteral("AV1 cannot be decoded in software: %1").arg(QString::fromUtf8(problem.what()));
            return nullptr;
        }
        decoding->codec_name = "libdav1d";
        return decoding;
    }
    decoding->codec = avcodec_alloc_context3(codec);
    if (decoding->codec == nullptr) {
        error = QStringLiteral("Out of memory");
        return nullptr;
    }
    avcodec_parameters_to_context(decoding->codec, stream->codecpar);
    decoding->codec->pkt_timebase = stream->time_base;
    if (hardware) {
        decoding->codec->hw_device_ctx = av_buffer_ref(device.hardware());
        decoding->codec->get_format = direct3d_format;
        decoding->codec->extra_hw_frames = held_surfaces;
    } else {
        decoding->codec->thread_count = 0;  // as many as the CPU has
    }
    result = avcodec_open2(decoding->codec, codec, nullptr);
    if (result < 0) {
        error = QStringLiteral("Could not start the decoder: %1").arg(error_text(result));
        return nullptr;
    }
    decoding->hardware = hardware;
    return decoding;
}

// A picture's time on the stream's timeline (ns), or nothing.
std::optional<std::int64_t> picture_time(const FfmpegNativeVideo::Decoding& decoding, const AVFrame* frame) {
    std::int64_t stamp = frame->best_effort_timestamp;
    if (stamp == AV_NOPTS_VALUE) {
        stamp = frame->pts;
    }
    if (stamp == AV_NOPTS_VALUE) {
        return std::nullopt;
    }
    return av_rescale_q(stamp, decoding.time_base, AVRational{1, 1'000'000'000});
}

// Copies rows of `bytes` each.
void copy_rows(std::uint8_t* target, std::size_t target_pitch, const std::uint8_t* source, std::ptrdiff_t source_pitch,
               std::size_t bytes, int rows) {
    for (int row = 0; row < rows; ++row) {
        std::memcpy(target + target_pitch * static_cast<std::size_t>(row), source + source_pitch * row, bytes);
    }
}

// Writes `frame`'s picture, in 4:2:0 with three planes or two, into NV12's
// or P010's two planes at `luma` and `chroma`.
bool write_planes(const AVFrame* frame, std::uint8_t* luma, std::uint8_t* chroma, std::size_t pitch) {
    const int width = frame->width;
    const int height = frame->height;
    const int chroma_width = (width + 1) / 2;
    const int chroma_height = (height + 1) / 2;
    const auto format = static_cast<AVPixelFormat>(frame->format);
    if (format == AV_PIX_FMT_NV12 || format == AV_PIX_FMT_P010LE) {
        const std::size_t sample = format == AV_PIX_FMT_NV12 ? 1 : 2;
        copy_rows(luma, pitch, frame->data[0], frame->linesize[0], sample * width, height);
        copy_rows(chroma, pitch, frame->data[1], frame->linesize[1], sample * 2 * chroma_width, chroma_height);
        return true;
    }
    if (format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUVJ420P) {
        copy_rows(luma, pitch, frame->data[0], frame->linesize[0], width, height);
        for (int row = 0; row < chroma_height; ++row) {
            std::uint8_t* out = chroma + pitch * static_cast<std::size_t>(row);
            const std::uint8_t* u = frame->data[1] + static_cast<std::ptrdiff_t>(frame->linesize[1]) * row;
            const std::uint8_t* v = frame->data[2] + static_cast<std::ptrdiff_t>(frame->linesize[2]) * row;
            for (int x = 0; x < chroma_width; ++x) {
                out[2 * x] = u[x];
                out[2 * x + 1] = v[x];
            }
        }
        return true;
    }
    if (format == AV_PIX_FMT_YUV420P10LE) {
        // P010 keeps the 10 bits at the top of each 16.
        for (int row = 0; row < height; ++row) {
            auto* out = reinterpret_cast<std::uint16_t*>(luma + pitch * static_cast<std::size_t>(row));
            const auto* in =
                reinterpret_cast<const std::uint16_t*>(frame->data[0] + static_cast<std::ptrdiff_t>(frame->linesize[0]) * row);
            for (int x = 0; x < width; ++x) {
                out[x] = static_cast<std::uint16_t>(in[x] << 6);
            }
        }
        for (int row = 0; row < chroma_height; ++row) {
            auto* out = reinterpret_cast<std::uint16_t*>(chroma + pitch * static_cast<std::size_t>(row));
            const auto* u =
                reinterpret_cast<const std::uint16_t*>(frame->data[1] + static_cast<std::ptrdiff_t>(frame->linesize[1]) * row);
            const auto* v =
                reinterpret_cast<const std::uint16_t*>(frame->data[2] + static_cast<std::ptrdiff_t>(frame->linesize[2]) * row);
            for (int x = 0; x < chroma_width; ++x) {
                out[2 * x] = static_cast<std::uint16_t>(u[x] << 6);
                out[2 * x + 1] = static_cast<std::uint16_t>(v[x] << 6);
            }
        }
        return true;
    }
    return false;
}

}  // namespace

void FfmpegNativeVideo::run() {
    const std::filesystem::path path = frame_input_path(comparison_, side_);
    const AVIOInterruptCB interrupt{&FfmpegNativeVideo::interrupted, this};
    QString why;
    // Direct3D 11's video decoding where the GPU decodes the video, else
    // software (FFmpeg's decoders, dav1d for AV1); the GPU's, if it fails,
    // gives way to software at the frame after the last given.
    std::unique_ptr<Decoding> decoding;
    if (gpu_decoding_) {
        decoding = open_video(path, *device_, true, interrupt, why);
    }
    if (!decoding) {
        decoding = open_video(path, *device_, false, interrupt, why);
        if (!decoding) {
            if (!cancelled()) {
                fail(why);
            }
            return;
        }
    }
    const QString side_name = side_ == FrameSide::source ? QStringLiteral("source") : QStringLiteral("distorted");
    const bool hdr10 = shaded_ == NativeShading::none && hdr_kind(frame_video_info(comparison_, side_)) == "HDR10 / PQ";
    std::optional<std::int64_t> origin_ns;  // the first frame's time on the stream's timeline
    std::int64_t keep_from_ns = std::numeric_limits<std::int64_t>::min();
    std::int64_t last_given_ns = -1;
    bool reported = false;
    bool at_end = false;
    // A stream with no timestamps (a raw H.264, HEVC or VVC stream, as
    // VVenC writes them): its frames are counted from its start instead, at
    // the comparison's frame rate, as FFmpeg's command line counts them; this
    // is the number of the next frame. Nothing says where a frame of such a
    // stream is, so a seek decodes on to the frame, or from the stream's
    // start again to one already passed.
    std::optional<std::int64_t> counted;
    const auto counted_time_ns = [&](std::int64_t number) {
        return comparison_.fps > 0 ? std::llround(static_cast<double>(number) * 1e9 / comparison_.fps)
                                   : number * decoding->frame_ns;
    };

    // Makes the seek asked for, if any (once the first frame is known).
    const auto make_seek = [&]() {
        std::optional<int> wanted;
        int asked = 0;
        {
            const std::lock_guard lock(mutex_);
            if (!wanted_ms_ || !origin_ns) {
                return;
            }
            wanted = std::exchange(wanted_ms_, std::nullopt);
            asked = seeks_asked_;
            frames_.clear();  // a flushing seek: nothing from before it is given
        }
        const std::int64_t target_ns = static_cast<std::int64_t>(*wanted) * 1'000'000;
        bool sought = true;
        if (!counted) {
            const std::int64_t stamp =
                av_rescale_q(*origin_ns + target_ns, AVRational{1, 1'000'000'000}, decoding->time_base);
            sought = av_seek_frame(decoding->format, decoding->stream, stamp, AVSEEK_FLAG_BACKWARD) >= 0;
            decoding->flush();
            decoding->draining = false;
            at_end = false;
        } else if (target_ns - counted_time_ns(1) / 2 < counted_time_ns(*counted)) {
            std::unique_ptr<Decoding> again = open_video(path, *device_, decoding->hardware, interrupt, why);
            sought = again != nullptr;
            if (again) {
                decoding = std::move(again);
                counted = 0;
                at_end = false;
            }
        }
        // Accurate: the frames before it are decoded, not given, but the one
        // on screen at it is.
        keep_from_ns = target_ns - (counted ? counted_time_ns(1) : decoding->frame_ns) / 2;
        const std::lock_guard lock(mutex_);
        seeks_made_ = std::max(seeks_made_, asked);
        if (!sought) {
            error_ = QStringLiteral("FFmpeg could not seek to that frame.");
        }
        if (at_end) {
            ended_ = true;  // sought past the end of a counted stream already decoded to it
        }
    };

    // Gives `frame`: its texture held for as long as the frame is.
    const auto give = [&](const AVFrame* frame, std::int64_t time_ns) -> bool {
        auto made = std::make_shared<NativeFrame>();
        made->time_ns = time_ns;
        made->width = frame->width;
        made->height = frame->height;
        const char* space = av_color_space_name(frame->colorspace);
        made->matrix = std::string(yuv_matrix_name(space != nullptr ? space : "", frame->height));
        made->full_range = frame->color_range == AVCOL_RANGE_JPEG;
        made->hdr10 = hdr10;
        if (frame->format == AV_PIX_FMT_D3D11) {
            const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
            made->ten_bit = frames->sw_format == AV_PIX_FMT_P010LE;
            made->texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
            made->subresource = static_cast<unsigned>(reinterpret_cast<std::intptr_t>(frame->data[1]));
            AVFrame* kept = av_frame_clone(frame);
            if (kept == nullptr) {
                return false;
            }
            made->keep = std::shared_ptr<AVFrame>(kept, [](AVFrame* held) { av_frame_free(&held); });
        } else {
            const AVPixFmtDescriptor* description = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
            if (description == nullptr) {
                return false;
            }
            made->ten_bit = description->comp[0].depth > 8;
            const DXGI_FORMAT format = made->ten_bit ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
            // NV12 and P010 textures have even sizes; the picture is drawn
            // from its own size.
            const int width = (frame->width + 1) & ~1;
            const int height = (frame->height + 1) & ~1;
            // Pictures not already 4:2:0 are converted, as GStreamer's
            // converter gave NV12 or P010.
            const AVFrame* picture = frame;
            AVFrame* converted = nullptr;
            const auto source_format = static_cast<AVPixelFormat>(frame->format);
            const bool direct = source_format == AV_PIX_FMT_NV12 || source_format == AV_PIX_FMT_P010LE ||
                                source_format == AV_PIX_FMT_YUV420P || source_format == AV_PIX_FMT_YUVJ420P ||
                                source_format == AV_PIX_FMT_YUV420P10LE;
            if (!direct) {
                const AVPixelFormat target = made->ten_bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
                decoding->converter =
                    sws_getCachedContext(decoding->converter, frame->width, frame->height, source_format, frame->width,
                                         frame->height, target, SWS_BICUBIC, nullptr, nullptr, nullptr);
                converted = av_frame_alloc();
                if (decoding->converter == nullptr || converted == nullptr) {
                    av_frame_free(&converted);
                    return false;
                }
                converted->format = target;
                converted->width = frame->width;
                converted->height = frame->height;
                if (av_frame_get_buffer(converted, 0) < 0 ||
                    sws_scale(decoding->converter, frame->data, frame->linesize, 0, frame->height, converted->data,
                              converted->linesize) <= 0) {
                    av_frame_free(&converted);
                    return false;
                }
                picture = converted;
            }
            if (!decoding->uploads) {
                decoding->uploads = std::make_shared<UploadTextures>(device_->device());
            }
            D3D11_TEXTURE2D_DESC staged{};
            if (decoding->staging) {
                decoding->staging->GetDesc(&staged);
            }
            if (!decoding->staging || staged.Width != static_cast<UINT>(width) ||
                staged.Height != static_cast<UINT>(height) || staged.Format != format) {
                D3D11_TEXTURE2D_DESC description_staging{};
                description_staging.Width = static_cast<UINT>(width);
                description_staging.Height = static_cast<UINT>(height);
                description_staging.MipLevels = 1;
                description_staging.ArraySize = 1;
                description_staging.Format = format;
                description_staging.SampleDesc.Count = 1;
                description_staging.Usage = D3D11_USAGE_STAGING;
                description_staging.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                decoding->staging.Reset();
                if (FAILED(device_->device()->CreateTexture2D(&description_staging, nullptr, &decoding->staging))) {
                    av_frame_free(&converted);
                    return false;
                }
            }
            std::shared_ptr<ID3D11Texture2D> texture = decoding->uploads->take(width, height, format);
            bool written = false;
            if (texture) {
                const std::lock_guard lock(device_->lock());
                D3D11_MAPPED_SUBRESOURCE mapped{};
                if (SUCCEEDED(device_->context()->Map(decoding->staging.Get(), 0, D3D11_MAP_WRITE, 0, &mapped))) {
                    auto* luma = static_cast<std::uint8_t*>(mapped.pData);
                    // The second plane follows the first, at the same pitch.
                    written = write_planes(picture, luma,
                                           luma + static_cast<std::size_t>(mapped.RowPitch) * static_cast<std::size_t>(height),
                                           mapped.RowPitch);
                    device_->context()->Unmap(decoding->staging.Get(), 0);
                    if (written) {
                        device_->context()->CopyResource(texture.get(), decoding->staging.Get());
                    }
                }
            }
            av_frame_free(&converted);
            if (!written) {
                return false;
            }
            made->texture = texture.get();
            made->keep = std::move(texture);
        }
        if (scale_to_ && !decoding->downscale_failed) {
            const std::lock_guard lock(device_->lock());
            try {
                if (!decoding->downscaler) {
                    D3D11_TEXTURE2D_DESC description{};
                    made->texture->GetDesc(&description);
                    const std::optional<CropBox>& crop = comparison_.source_crop;
                    const RECT area = crop ? RECT{crop->x, crop->y, crop->x + crop->w, crop->y + crop->h}
                                           : RECT{0, 0, frame->width, frame->height};
                    decoding->downscaler = std::make_unique<Downscaler>(
                        device_->device(), device_->context(), static_cast<int>(description.Width),
                        static_cast<int>(description.Height), area, *scale_to_, description.Format);
                }
                if (std::shared_ptr<ID3D11Texture2D> scaled = decoding->downscaler->scale(made->texture, made->subresource)) {
                    made->texture = scaled.get();
                    made->subresource = 0;
                    made->width = scale_to_->first;
                    made->height = scale_to_->second;
                    made->prepared = true;
                    made->keep = std::move(scaled);
                }
            } catch (const std::exception&) {
                // Shown at its own size, as when played natively.
                decoding->downscale_failed = true;
            }
        }
        if (!reported) {
            reported = true;
            QStringList details{QString::fromUtf8("%1\xC3\x97%2 %3")
                                    .arg(made->width)
                                    .arg(made->height)
                                    .arg(made->ten_bit ? QStringLiteral("P010_10LE") : QStringLiteral("NV12")),
                                QStringLiteral("memory:D3D11Memory"),
                                decoding->hardware ? QStringLiteral("%1 GPU: d3d11va %2")
                                                         .arg(side_name, QString::fromStdString(decoding->codec_name))
                                                   : QStringLiteral("%1 software: %2")
                                                         .arg(side_name, QString::fromStdString(decoding->codec_name))};
            if (shaded_ == NativeShading::sdr) {
                details.append(QString::fromUtf8("GPU HDR\xE2\x86\x92SDR \xC2\xB7 fixed Reinhard 1000\xE2\x86\x92" "100 nit"));
            } else if (shaded_ == NativeShading::hdr) {
                details.append(QString::fromUtf8("GPU colours \xE2\x86\x92 PQ BT.2020, HDR kept"));
            }
            const std::lock_guard lock(mutex_);
            status_ = details.join(QString::fromUtf8(" \xC2\xB7 "));
        }
        if (!wait_for_room()) {
            return true;  // cancelled, or a seek asked for: this frame is not wanted
        }
        deliver(std::move(made));
        return true;
    };

    while (!cancelled()) {
        make_seek();
        if (at_end) {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [this] { return cancelled_ || wanted_ms_.has_value(); });
            continue;
        }
        if (!wait_for_room()) {
            continue;
        }
        int result = 0;
        if (!decoding->draining) {
            result = av_read_frame(decoding->format, decoding->packet);
            if (result == AVERROR_EOF) {
                decoding->draining = true;
                result = decoding->send(nullptr);
            } else if (result < 0) {
                if (!cancelled()) {
                    fail(QStringLiteral("Could not read the video: %1").arg(error_text(result)));
                }
                return;
            } else if (decoding->packet->stream_index != decoding->stream) {
                av_packet_unref(decoding->packet);
                continue;
            } else {
                result = decoding->send(decoding->packet);
                av_packet_unref(decoding->packet);
            }
        }
        // Why this way of decoding failed, if it did.
        QString failure;
        if (result < 0 && result != AVERROR(EAGAIN) && result != AVERROR_EOF) {
            failure = QStringLiteral("Decoding failed: %1").arg(error_text(result));
        }
        while (failure.isEmpty()) {
            result = decoding->receive(decoding->frame);
            if (result == AVERROR(EAGAIN)) {
                break;
            }
            if (result == AVERROR_EOF) {
                at_end = true;
                const std::lock_guard lock(mutex_);
                ended_ = true;
                break;
            }
            if (result < 0) {
                failure = QStringLiteral("Decoding failed: %1").arg(error_text(result));
                break;
            }
            std::optional<std::int64_t> time = picture_time(*decoding, decoding->frame);
            if (!time && (!origin_ns || counted)) {
                if (!counted) {
                    counted = 0;
                }
                time = counted_time_ns((*counted)++);
            } else if (!time) {
                fail(QStringLiteral("Video frame has no usable presentation timestamp"));
                return;
            }
            if (!origin_ns && counted) {
                origin_ns = 0;
                const std::lock_guard lock(mutex_);
                first_frame_ms_ = 0;
            } else if (!origin_ns) {
                origin_ns = *time;
                const AVFormatContext* format = decoding->format;
                const std::int64_t file_start =
                    format->start_time == AV_NOPTS_VALUE ? 0 : av_rescale(format->start_time, 1'000'000'000, AV_TIME_BASE);
                const std::lock_guard lock(mutex_);
                first_frame_ms_ = static_cast<int>(std::llround(static_cast<double>(*origin_ns - file_start) / 1e6));
            }
            if (*time - *origin_ns >= keep_from_ns) {
                if (!give(decoding->frame, *time - *origin_ns)) {
                    failure = QStringLiteral("A decoded picture could not be shown");
                } else {
                    last_given_ns = *time - *origin_ns;
                }
            }
            decoding->unref_frame();
        }
        if (failure.isEmpty()) {
            continue;
        }
        if (!decoding->hardware) {
            fail(failure);
            return;
        }
        // The GPU's decoder failed (a profile it does not take, a device
        // lost): FFmpeg's software decoder goes on from the frame after the
        // last given, or from the start.
        std::unique_ptr<Decoding> software = open_video(path, *device_, false, interrupt, why);
        if (!software) {
            if (!cancelled()) {
                fail(why);
            }
            return;
        }
        decoding = std::move(software);
        reported = false;
        if (counted) {
            // From the start again, counted, keeping the frames after the last given.
            counted = 0;
            if (last_given_ns >= 0) {
                keep_from_ns = last_given_ns + counted_time_ns(1) / 2;
            }
        } else if (origin_ns && last_given_ns >= 0) {
            const std::int64_t next_ns = last_given_ns + decoding->frame_ns / 2;
            av_seek_frame(decoding->format, decoding->stream,
                          av_rescale_q(*origin_ns + next_ns, AVRational{1, 1'000'000'000}, decoding->time_base),
                          AVSEEK_FLAG_BACKWARD);
            keep_from_ns = next_ns;
        }
    }
}

}  // namespace vml::app
