#include "native_decoder.h"

#include <d3d11.h>
#include <wrl/client.h>

#include <QStringList>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "ffmpeg_api.h"
#include "native_dav1d.h"
#include "native_device.h"
#include "native_grain.h"
#include "vml/app_log.h"
#include "vml/film_grain.h"
#include "vml/platform/system.h"

using Microsoft::WRL::ComPtr;

namespace vml::app {

namespace {

const app_log::Logger logger("vml.gstreamer_playback");

// The most frames of one video held outside its decoder: those ready to be
// taken, the playback pool's, and the presenter's, those waiting and the one
// being drawn, which after a seek are no longer among the pool's. While a
// video plays, the pool's and the ready frame are all that is held.
constexpr int held_frames = static_cast<int>(decoder_ready_frames + pool_kept_frames + presenter_waiting_frames + 1);

QString error_text(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return QString::fromUtf8(text.data());
}

// The most pictures an HEVC stream's decoder keeps at once (its decoded
// picture buffer, MaxDpbSize), from its level and picture size as H.265's
// Annex A.4.2 limits them; 16, the most the standard allows, where the
// level is not one it lists or allows no picture that large (the level is
// then wrong, and limits nothing).
int hevc_buffered_pictures(const AVCodecContext& context) {
    // The level's largest picture in luma samples (Table A.8), by
    // general_level_idc (30 times the level).
    static const std::map<int, long long> max_luma_samples{
        {30, 36864},     {60, 122880},    {63, 245760},    {90, 552960},    {93, 983040},
        {120, 2228224},  {123, 2228224},  {150, 8912896},  {153, 8912896},  {156, 8912896},
        {180, 35651584}, {183, 35651584}, {186, 35651584},
    };
    const auto level = max_luma_samples.find(context.level);
    const long long picture = static_cast<long long>(context.coded_width) * context.coded_height;
    if (level == max_luma_samples.end() || picture > level->second) {
        return 16;
    }
    const long long most = level->second;
    const int buffers = context.profile == AV_PROFILE_HEVC_SCC ? 7 : 6;  // maxDpbPicBuf
    if (picture <= most / 4) {
        return std::min(4 * buffers, 16);
    }
    if (picture <= most / 2) {
        return std::min(2 * buffers, 16);
    }
    if (picture <= 3 * most / 4) {
        return std::min(4 * buffers / 3, 16);
    }
    return buffers;
}

// The surfaces an HEVC or AV1 decoder uses at most: the pictures it refers
// to and the one being decoded, which FFmpeg makes before it lets go of the
// pictures no longer referred to. An HEVC stream's level limits the pictures
// it keeps, the one being decoded among them (a 4K one at level 5.1, 6); AV1
// refers to 8. Nothing for other codecs: FFmpeg's H.264 decoder can hold
// back more frames than the level allows when a stream does not say how far
// its frames are reordered, and its VP9 decoder keeps earlier frames beside
// the ones referred to.
std::optional<int> decoder_surfaces(const AVCodecContext& context) {
    switch (context.codec_id) {
    case AV_CODEC_ID_HEVC:
        return 1 + hevc_buffered_pictures(context);
    case AV_CODEC_ID_AV1:
        return 1 + 8;
    default:
        return std::nullopt;
    }
}

// Gives `context` a pool of `surfaces` Direct3D 11 surfaces, set up as
// FFmpeg sets up its own. Where it cannot, FFmpeg makes its own, larger.
void make_pool(AVCodecContext* context, int surfaces) {
    AVBufferRef* frames = nullptr;
    if (avcodec_get_hw_frames_parameters(context, context->hw_device_ctx, AV_PIX_FMT_D3D11, &frames) < 0) {
        return;
    }
    reinterpret_cast<AVHWFramesContext*>(frames->data)->initial_pool_size = surfaces;
    if (av_hwframe_ctx_init(frames) < 0) {
        av_buffer_unref(&frames);
        return;
    }
    av_buffer_unref(&context->hw_frames_ctx);
    context->hw_frames_ctx = frames;
}

// FFmpeg's choice of pictures: Direct3D 11's, or none (the decoder then
// fails, and the video is decoded in software). The stream is known by now
// (its level and size), and FFmpeg makes its surfaces after this returns:
// as many as the decoder uses and held_frames more. FFmpeg's own pool,
// which other codecs get, has four work surfaces, of which decoding needs
// one, and one for each picture a stream can refer to, 16 for H.264 and
// HEVC and 8 for AV1 and VP9 (decode.c, ff_decode_get_hw_frames_ctx;
// dxva2.c, ff_dxva2_common_frame_params), and held_frames more
// (extra_hw_frames).
AVPixelFormat direct3d_format(AVCodecContext* context, const AVPixelFormat* formats) {
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_D3D11) {
            if (const std::optional<int> used = decoder_surfaces(*context)) {
                make_pool(context, *used + held_frames);
            }
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

// Textures for frames (a picture decoded in software, its film grain drawn,
// a picture scaled down), each reused once let go. All those made are kept
// until the size or format asked for changes: they number at most the
// frames held at once (held_frames).
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
        if (width == width_ && height == height_ && format == format_) {
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
    // A VVC video's film grain, drawn on its frames when FFmpeg gives each
    // frame's grain message rather than drawing it; the planes a frame with
    // grain is uploaded to, and the textures it is shown from.
    std::unique_ptr<NativeGrain> grain;
    std::array<ComPtr<ID3D11Texture2D>, 3> grain_planes;
    std::shared_ptr<UploadTextures> grain_targets;
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
    changed_.wait(lock, [&] { return cancelled_ || seek_due() || frames_.size() < decoder_ready_frames; });
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
// `interrupt` says so. A VVC video's film grain is drawn by the grain library
// where it can be (`library_grain`), else by FFmpeg. Returns nothing, with
// `error` set, when it cannot.
std::unique_ptr<FfmpegNativeVideo::Decoding> open_video(const std::filesystem::path& path, NativeDevice& device,
                                                         bool hardware, const AVIOInterruptCB& interrupt,
                                                         QString& error, bool library_grain = true) {
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
    if (!hardware && library_grain && codec->id == AV_CODEC_ID_VVC && !NativeGrain::unavailable()) {
        // Each frame's film grain message given, not drawn: FFmpeg draws
        // H.274 grain on 8-bit video only, the grain library on every frame,
        // as the Python app draws it. (AV1's grain, AFGS1, the library does
        // not draw: a video with it is opened again with FFmpeg drawing it.)
        try {
            decoding->grain = std::make_unique<NativeGrain>(device.device());
            decoding->codec->export_side_data |= AV_CODEC_EXPORT_DATA_FILM_GRAIN;
        } catch (const std::exception& error) {
            logger.warning(std::string("Film grain could not be drawn, and is left to FFmpeg: ") + error.what());
        }
    }
    if (hardware) {
        decoding->codec->hw_device_ctx = av_buffer_ref(device.hardware());
        decoding->codec->get_format = direct3d_format;
        decoding->codec->extra_hw_frames = held_frames;
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

// Frees an AVFrame it owns.
struct FrameFree {
    void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};
using OwnedFrame = std::unique_ptr<AVFrame, FrameFree>;

// Whether pictures in `format` are uploaded as they are: 4:2:0, as NV12 or
// P010 hold them.
bool is_420(AVPixelFormat format) {
    return format == AV_PIX_FMT_NV12 || format == AV_PIX_FMT_P010LE || format == AV_PIX_FMT_YUV420P ||
           format == AV_PIX_FMT_YUVJ420P || format == AV_PIX_FMT_YUV420P10LE;
}

// `frame` converted to P010 (`ten_bit`) or NV12 with `converter`, made or
// remade for it; nothing when it cannot be.
OwnedFrame converted_to_420(SwsContext*& converter, const AVFrame* frame, bool ten_bit) {
    const AVPixelFormat target = ten_bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    converter = sws_getCachedContext(converter, frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                                     frame->width, frame->height, target, SWS_BICUBIC, nullptr, nullptr, nullptr);
    OwnedFrame converted(av_frame_alloc());
    if (converter == nullptr || converted == nullptr) {
        return nullptr;
    }
    converted->format = target;
    converted->width = frame->width;
    converted->height = frame->height;
    if (av_frame_get_buffer(converted.get(), 0) < 0 ||
        sws_scale(converter, frame->data, frame->linesize, 0, frame->height, converted->data, converted->linesize) <=
            0) {
        return nullptr;
    }
    return converted;
}

// A film grain message as FFmpeg gives it (H.274's), as the grain library
// takes it.
FilmGrainMessage grain_message(const AVFilmGrainH274Params& h274) {
    FilmGrainMessage message;
    message.model_id = h274.model_id;
    message.log2_scale_factor = h274.log2_scale_factor;
    for (std::size_t c = 0; c < 3; ++c) {
        if (h274.component_model_present[c] == 0) {
            continue;
        }
        for (int i = 0; i < h274.num_intensity_intervals[c]; ++i) {
            FilmGrainInterval interval{h274.intensity_interval_lower_bound[c][i],
                                       h274.intensity_interval_upper_bound[c][i], {}};
            for (int v = 0; v < h274.num_model_values[c]; ++v) {
                interval.values.push_back(h274.comp_model_value[c][i][v]);
            }
            message.components[c].push_back(std::move(interval));
        }
    }
    return message;
}

// Makes sure `texture` is a `width` x `height` texture in `format`, for
// `usage` and `bind` (with `cpu_access`), making it again if not; false
// when it cannot be made.
bool ensure_texture(ID3D11Device* device, ComPtr<ID3D11Texture2D>& texture, UINT width, UINT height,
                    DXGI_FORMAT format, D3D11_USAGE usage, UINT bind, UINT cpu_access = 0) {
    if (texture) {
        D3D11_TEXTURE2D_DESC existing{};
        texture->GetDesc(&existing);
        if (existing.Width == width && existing.Height == height && existing.Format == format) {
            return true;
        }
    }
    D3D11_TEXTURE2D_DESC description{};
    description.Width = width;
    description.Height = height;
    description.MipLevels = 1;
    description.ArraySize = 1;
    description.Format = format;
    description.SampleDesc.Count = 1;
    description.Usage = usage;
    description.BindFlags = bind;
    description.CPUAccessFlags = cpu_access;
    texture.Reset();
    return SUCCEEDED(device->CreateTexture2D(&description, nullptr, &texture));
}

}  // namespace

// One playing of the video, on its decoder's thread: the opening, the
// frames' timeline, seeks, film grain, the frames given, and the switch to
// software decoding when the GPU's decoder fails.
class FfmpegNativeVideo::Session {
public:
    explicit Session(FfmpegNativeVideo& video)
        : video_(video),
          path_(frame_input_path(video.comparison_, video.side_)),
          interrupt_{&FfmpegNativeVideo::interrupted, &video},
          side_name_(video.side_ == FrameSide::source ? QStringLiteral("source") : QStringLiteral("distorted")),
          hdr10_(video.shaded_ == NativeShading::none &&
                 hdr_kind(frame_video_info(video.comparison_, video.side_)) == "HDR10 / PQ") {}

    // Decodes until cancelled, or until decoding fails (fail() says why).
    void run();

private:
    bool open();
    void make_seek();
    std::int64_t counted_time_ns(std::int64_t number) const;
    bool take_frames(QString& failure);
    std::optional<std::int64_t> frame_time();
    bool grain_left_to_ffmpeg();
    bool decode_on_in_software();
    bool give(const AVFrame* frame, std::int64_t time_ns);
    std::shared_ptr<ID3D11Texture2D> grained(const AVFrame* frame, const AVFilmGrainParams& given);
    bool upload(const AVFrame* frame, NativeFrame& made);
    void scale_down(const AVFrame* frame, NativeFrame& made);
    void report(const NativeFrame& made);

    FfmpegNativeVideo& video_;
    const std::filesystem::path path_;
    const AVIOInterruptCB interrupt_;
    // Why the video could not be opened, if it could not.
    QString why_;
    // Direct3D 11's video decoding where the GPU decodes the video, else
    // software (FFmpeg's decoders, dav1d for AV1).
    std::unique_ptr<Decoding> decoding_;
    const QString side_name_;
    // PQ video shown as it comes (NativeFrame::hdr10).
    const bool hdr10_;

    // The first frame's time on the stream's timeline.
    std::optional<std::int64_t> origin_ns_;
    // Frames before this time (from the first frame) are decoded, not given.
    std::int64_t keep_from_ns_ = std::numeric_limits<std::int64_t>::min();
    std::int64_t last_given_ns_ = -1;
    bool at_end_ = false;
    // A stream with no timestamps (a raw H.264, HEVC or VVC stream, as
    // VVenC writes them): its frames are counted from its start instead, at
    // the comparison's frame rate, as FFmpeg's command line counts them; this
    // is the number of the next frame. Nothing says where a frame of such a
    // stream is, so a seek decodes on to the frame, or from the stream's
    // start again to one already passed.
    std::optional<std::int64_t> counted_;

    // Whether FFmpeg draws the video's grain: AV1's (AFGS1), which the grain
    // library does not draw.
    bool grain_by_ffmpeg_ = false;
    // What the status line says about the film grain: drawn, or turned off
    // by an error (then not tried again for this video).
    std::optional<QString> grain_note_;
    bool grain_off_ = false;
    // The status line given, and the grain note it had.
    bool reported_ = false;
    std::optional<QString> reported_grain_note_;
};

void FfmpegNativeVideo::run() { Session(*this).run(); }

void FfmpegNativeVideo::Session::run() {
    if (!open()) {
        return;
    }
    while (!video_.cancelled()) {
        make_seek();
        if (at_end_) {
            std::unique_lock lock(video_.mutex_);
            video_.changed_.wait(lock, [this] { return video_.cancelled_ || video_.wanted_ms_.has_value(); });
            continue;
        }
        if (!video_.wait_for_room()) {
            continue;
        }
        int result = 0;
        if (!decoding_->draining) {
            result = av_read_frame(decoding_->format, decoding_->packet);
            if (result == AVERROR_EOF) {
                decoding_->draining = true;
                result = decoding_->send(nullptr);
            } else if (result < 0) {
                if (!video_.cancelled()) {
                    video_.fail(QStringLiteral("Could not read the video: %1").arg(error_text(result)));
                }
                return;
            } else if (decoding_->packet->stream_index != decoding_->stream) {
                av_packet_unref(decoding_->packet);
                continue;
            } else {
                result = decoding_->send(decoding_->packet);
                av_packet_unref(decoding_->packet);
            }
        }
        // Why this way of decoding failed, if it did.
        QString failure;
        if (result < 0 && result != AVERROR(EAGAIN) && result != AVERROR_EOF) {
            failure = QStringLiteral("Decoding failed: %1").arg(error_text(result));
        } else if (!take_frames(failure)) {
            return;
        }
        // Whether FFmpeg is to draw the grain the grain library was to draw.
        const bool regrain = grain_by_ffmpeg_ && decoding_->grain;
        if (failure.isEmpty() && !regrain) {
            continue;
        }
        if (!decoding_->hardware && !regrain) {
            video_.fail(failure);
            return;
        }
        if (!decode_on_in_software()) {
            return;
        }
    }
}

// Opens the video: with the GPU's decoder where it decodes it, else in
// software. False when neither can (fail() then says why).
bool FfmpegNativeVideo::Session::open() {
    if (video_.gpu_decoding_) {
        decoding_ = open_video(path_, *video_.device_, true, interrupt_, why_);
    }
    if (!decoding_) {
        decoding_ = open_video(path_, *video_.device_, false, interrupt_, why_);
    }
    if (!decoding_ && !video_.cancelled()) {
        video_.fail(why_);
    }
    return decoding_ != nullptr;
}

// Makes the seek asked for, if any (once the first frame is known).
void FfmpegNativeVideo::Session::make_seek() {
    std::optional<int> wanted;
    int asked = 0;
    {
        const std::lock_guard lock(video_.mutex_);
        if (!video_.wanted_ms_ || !origin_ns_) {
            return;
        }
        wanted = std::exchange(video_.wanted_ms_, std::nullopt);
        asked = video_.seeks_asked_;
        video_.frames_.clear();  // a flushing seek: nothing from before it is given
    }
    const std::int64_t target_ns = static_cast<std::int64_t>(*wanted) * 1'000'000;
    bool sought = true;
    if (!counted_) {
        const std::int64_t stamp =
            av_rescale_q(*origin_ns_ + target_ns, AVRational{1, 1'000'000'000}, decoding_->time_base);
        sought = av_seek_frame(decoding_->format, decoding_->stream, stamp, AVSEEK_FLAG_BACKWARD) >= 0;
        decoding_->flush();
        decoding_->draining = false;
        at_end_ = false;
    } else if (target_ns - counted_time_ns(1) / 2 < counted_time_ns(*counted_)) {
        std::unique_ptr<Decoding> again =
            open_video(path_, *video_.device_, decoding_->hardware, interrupt_, why_, !grain_by_ffmpeg_);
        sought = again != nullptr;
        if (again) {
            decoding_ = std::move(again);
            counted_ = 0;
            at_end_ = false;
        }
    }
    // Accurate: the frames before it are decoded, not given, but the one on
    // screen at it is.
    keep_from_ns_ = target_ns - (counted_ ? counted_time_ns(1) : decoding_->frame_ns) / 2;
    const std::lock_guard lock(video_.mutex_);
    video_.seeks_made_ = std::max(video_.seeks_made_, asked);
    if (!sought) {
        video_.error_ = QStringLiteral("FFmpeg could not seek to that frame.");
    }
    if (at_end_) {
        video_.ended_ = true;  // sought past the end of a counted stream already decoded to it
    }
}

// The time of a counted stream's frame `number` (ns).
std::int64_t FfmpegNativeVideo::Session::counted_time_ns(std::int64_t number) const {
    return video_.comparison_.fps > 0 ? std::llround(static_cast<double>(number) * 1e9 / video_.comparison_.fps)
                                      : number * decoding_->frame_ns;
}

// Takes the frames the decoder has made, as there is room for them, and
// gives those from keep_from_ns_ on. `failure` says why, when the decoder
// failed or a frame could not be given; false when a frame has no usable
// time (fail() then says so).
bool FfmpegNativeVideo::Session::take_frames(QString& failure) {
    while (true) {
        // A frame is taken only when there is room for it: one waiting here
        // would keep a surface the decoder's pool has none for.
        if (!video_.wait_for_room()) {
            return true;  // cancelled, or a seek asked for: the frames still in the decoder are not wanted
        }
        const int result = decoding_->receive(decoding_->frame);
        if (result == AVERROR(EAGAIN)) {
            return true;
        }
        if (result == AVERROR_EOF) {
            at_end_ = true;
            const std::lock_guard lock(video_.mutex_);
            video_.ended_ = true;
            return true;
        }
        if (result < 0) {
            failure = QStringLiteral("Decoding failed: %1").arg(error_text(result));
            return true;
        }
        const std::optional<std::int64_t> time = frame_time();
        if (!time) {
            video_.fail(QStringLiteral("Video frame has no usable presentation timestamp"));
            return false;
        }
        if (grain_left_to_ffmpeg()) {
            decoding_->unref_frame();
            return true;
        }
        if (*time - *origin_ns_ >= keep_from_ns_) {
            if (!give(decoding_->frame, *time - *origin_ns_)) {
                failure = QStringLiteral("A decoded picture could not be shown");
            } else {
                last_given_ns_ = *time - *origin_ns_;
            }
        }
        decoding_->unref_frame();
        if (!failure.isEmpty()) {
            return true;
        }
    }
}

// The time of the frame just received, on the stream's timeline (ns):
// counted where the stream has none. The first frame's says where the video
// starts. Nothing when a frame has no time though the first had one.
std::optional<std::int64_t> FfmpegNativeVideo::Session::frame_time() {
    std::optional<std::int64_t> time = picture_time(*decoding_, decoding_->frame);
    if (!time && (!origin_ns_ || counted_)) {
        if (!counted_) {
            counted_ = 0;
        }
        time = counted_time_ns((*counted_)++);
    } else if (!time) {
        return std::nullopt;
    }
    if (!origin_ns_ && counted_) {
        origin_ns_ = 0;
        const std::lock_guard lock(video_.mutex_);
        video_.first_frame_ms_ = 0;
    } else if (!origin_ns_) {
        origin_ns_ = *time;
        const AVFormatContext* format = decoding_->format;
        const std::int64_t file_start =
            format->start_time == AV_NOPTS_VALUE ? 0 : av_rescale(format->start_time, 1'000'000'000, AV_TIME_BASE);
        const std::lock_guard lock(video_.mutex_);
        video_.first_frame_ms_ = static_cast<int>(std::llround(static_cast<double>(*origin_ns_ - file_start) / 1e6));
    }
    return time;
}

// Whether the frame just received has AV1's film grain (AFGS1) in a VVC
// video: FFmpeg draws it, not the grain library, so the video is to be
// decoded again from this frame with FFmpeg drawing it.
bool FfmpegNativeVideo::Session::grain_left_to_ffmpeg() {
    if (!decoding_->grain) {
        return false;
    }
    const AVFrameSideData* grain = av_frame_get_side_data(decoding_->frame, AV_FRAME_DATA_FILM_GRAIN_PARAMS);
    if (grain == nullptr || reinterpret_cast<const AVFilmGrainParams*>(grain->data)->type != AV_FILM_GRAIN_PARAMS_AV1) {
        return false;
    }
    logger.info("The video's film grain is AV1's (AFGS1): FFmpeg draws it");
    grain_by_ffmpeg_ = true;
    return true;
}

// The GPU's decoder failed (a profile it does not take, a device lost), or
// FFmpeg is to draw the grain: FFmpeg's software decoder goes on from the
// frame after the last given, or from the start. False when it cannot (fail()
// then says why).
bool FfmpegNativeVideo::Session::decode_on_in_software() {
    std::unique_ptr<Decoding> software = open_video(path_, *video_.device_, false, interrupt_, why_, !grain_by_ffmpeg_);
    if (!software) {
        if (!video_.cancelled()) {
            video_.fail(why_);
        }
        return false;
    }
    decoding_ = std::move(software);
    reported_ = false;
    if (counted_) {
        // From the start again, counted, keeping the frames after the last given.
        counted_ = 0;
        if (last_given_ns_ >= 0) {
            keep_from_ns_ = last_given_ns_ + counted_time_ns(1) / 2;
        }
    } else if (origin_ns_ && last_given_ns_ >= 0) {
        const std::int64_t next_ns = last_given_ns_ + decoding_->frame_ns / 2;
        av_seek_frame(decoding_->format, decoding_->stream,
                      av_rescale_q(*origin_ns_ + next_ns, AVRational{1, 1'000'000'000}, decoding_->time_base),
                      AVSEEK_FLAG_BACKWARD);
        keep_from_ns_ = next_ns;
    }
    return true;
}

// Gives `frame`, at `time_ns` from the first frame: its texture held for as
// long as the frame is. False when it cannot be shown.
bool FfmpegNativeVideo::Session::give(const AVFrame* frame, std::int64_t time_ns) {
    auto made = std::make_shared<NativeFrame>();
    made->time_ns = time_ns;
    made->width = frame->width;
    made->height = frame->height;
    const char* space = av_color_space_name(frame->colorspace);
    made->matrix = std::string(yuv_matrix_name(space != nullptr ? space : "", frame->height));
    made->full_range = frame->color_range == AVCOL_RANGE_JPEG;
    made->hdr10 = hdr10_;
    if (frame->format == AV_PIX_FMT_D3D11) {
        // A slice of the decoder's texture array, held with the frame.
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
        const AVFrameSideData* grain =
            decoding_->grain && !grain_off_ ? av_frame_get_side_data(frame, AV_FRAME_DATA_FILM_GRAIN_PARAMS) : nullptr;
        if (grain != nullptr) {
            if (std::shared_ptr<ID3D11Texture2D> texture =
                    grained(frame, *reinterpret_cast<const AVFilmGrainParams*>(grain->data))) {
                made->texture = texture.get();
                made->keep = std::move(texture);
            }
        }
        if (made->texture == nullptr && !upload(frame, *made)) {
            return false;
        }
    }
    if (video_.scale_to_ && !decoding_->downscale_failed) {
        scale_down(frame, *made);
    }
    report(*made);
    video_.deliver(std::move(made));
    return true;
}

// `frame`'s film grain (`given`, its message as FFmpeg gave it) drawn on its
// planes, uploaded, and packed into the texture it is shown from; nothing
// when it cannot be drawn so (it is then shown without it): only H.274's
// frequency model, on planar 4:2:0, 4:2:2, 4:4:4 or 4:0:0 video of 8 to 12
// bits, as the Python app draws it.
std::shared_ptr<ID3D11Texture2D> FfmpegNativeVideo::Session::grained(const AVFrame* frame,
                                                                     const AVFilmGrainParams& given) {
    const AVPixFmtDescriptor* layout = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
    if (given.type != AV_FILM_GRAIN_PARAMS_H274 || layout == nullptr ||
        (layout->flags & (AV_PIX_FMT_FLAG_PLANAR | AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_ALPHA | AV_PIX_FMT_FLAG_BE |
                          AV_PIX_FMT_FLAG_HWACCEL)) != AV_PIX_FMT_FLAG_PLANAR ||
        (layout->nb_components != 1 && layout->nb_components != 3) || layout->comp[0].depth < 8 ||
        layout->comp[0].depth > 12 || layout->log2_chroma_w > 1 || layout->log2_chroma_h > 1) {
        return nullptr;
    }
    const int depth = layout->comp[0].depth;
    const int count = layout->nb_components;
    const std::pair<int, int> chroma_shift =
        count == 3 ? std::pair<int, int>(layout->log2_chroma_w, layout->log2_chroma_h) : std::pair(0, 0);
    const std::vector<std::pair<int, int>> sizes = grain_plane_sizes(frame->width, frame->height, chroma_shift, count);
    // The seed is the frame's picture order count, which can be negative.
    const std::optional<GrainParams> params =
        grain_params(grain_message(given.codec.h274), static_cast<int>(given.seed), depth, sizes, chroma_shift);
    if (!params) {
        return nullptr;
    }
    ID3D11Device* gpu = video_.device_->device();
    for (std::size_t c = 0; c < static_cast<std::size_t>(count); ++c) {
        if (!ensure_texture(gpu, decoding_->grain_planes[c], static_cast<UINT>(sizes[c].first),
                            static_cast<UINT>(sizes[c].second), depth > 8 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM,
                            D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE)) {
            return nullptr;
        }
    }
    if (!decoding_->grain_targets) {
        decoding_->grain_targets =
            std::make_shared<UploadTextures>(gpu, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
    }
    std::shared_ptr<ID3D11Texture2D> target = decoding_->grain_targets->take(
        (frame->width + 1) & ~1, (frame->height + 1) & ~1, depth > 8 ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12);
    if (!target) {
        return nullptr;
    }
    std::optional<QString> failed;
    {
        const std::lock_guard lock(video_.device_->lock());
        for (std::size_t c = 0; c < static_cast<std::size_t>(count); ++c) {
            video_.device_->context()->UpdateSubresource(decoding_->grain_planes[c].Get(), 0, nullptr, frame->data[c],
                                                         static_cast<UINT>(frame->linesize[c]), 0);
        }
        ID3D11Texture2D* const planes[3] = {decoding_->grain_planes[0].Get(), decoding_->grain_planes[1].Get(),
                                            decoding_->grain_planes[2].Get()};
        failed = decoding_->grain->draw(planes, count, *params, chroma_shift, target.get());
    }
    if (failed) {
        logger.warning("Film grain could not be drawn, and is left off: " + failed->toStdString());
        grain_off_ = true;
        grain_note_ = QStringLiteral("film grain off (error)");
        return nullptr;
    }
    grain_note_ = QStringLiteral("film grain synthesized");
    return target;
}

// `frame`, decoded in software, uploaded into a texture of its own, NV12 or
// P010, which `made` then holds: converted first when it is not 4:2:0.
// False when it cannot be.
bool FfmpegNativeVideo::Session::upload(const AVFrame* frame, NativeFrame& made) {
    const DXGI_FORMAT format = made.ten_bit ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
    // NV12 and P010 textures have even sizes; the picture is drawn from its
    // own size.
    const int width = (frame->width + 1) & ~1;
    const int height = (frame->height + 1) & ~1;
    OwnedFrame converted;
    const AVFrame* picture = frame;
    if (!is_420(static_cast<AVPixelFormat>(frame->format))) {
        converted = converted_to_420(decoding_->converter, frame, made.ten_bit);
        if (!converted) {
            return false;
        }
        picture = converted.get();
    }
    ID3D11Device* gpu = video_.device_->device();
    if (!decoding_->uploads) {
        decoding_->uploads = std::make_shared<UploadTextures>(gpu);
    }
    if (!ensure_texture(gpu, decoding_->staging, static_cast<UINT>(width), static_cast<UINT>(height), format,
                        D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_WRITE)) {
        return false;
    }
    std::shared_ptr<ID3D11Texture2D> texture = decoding_->uploads->take(width, height, format);
    if (!texture) {
        return false;
    }
    {
        const std::lock_guard lock(video_.device_->lock());
        ID3D11DeviceContext* context = video_.device_->context();
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (FAILED(context->Map(decoding_->staging.Get(), 0, D3D11_MAP_WRITE, 0, &mapped))) {
            return false;
        }
        auto* luma = static_cast<std::uint8_t*>(mapped.pData);
        // The second plane follows the first, at the same pitch.
        const bool written = write_planes(
            picture, luma, luma + static_cast<std::size_t>(mapped.RowPitch) * static_cast<std::size_t>(height),
            mapped.RowPitch);
        context->Unmap(decoding_->staging.Get(), 0);
        if (!written) {
            return false;
        }
        context->CopyResource(texture.get(), decoding_->staging.Get());
    }
    made.texture = texture.get();
    made.keep = std::move(texture);
    return true;
}

// The source, `made` from `frame`, scaled down to the test video's size
// (cropped first), when it can be; else it is shown at its own size.
void FfmpegNativeVideo::Session::scale_down(const AVFrame* frame, NativeFrame& made) {
    const std::lock_guard lock(video_.device_->lock());
    try {
        if (!decoding_->downscaler) {
            D3D11_TEXTURE2D_DESC description{};
            made.texture->GetDesc(&description);
            const std::optional<CropBox>& crop = video_.comparison_.source_crop;
            const RECT area = crop ? RECT{crop->x, crop->y, crop->x + crop->w, crop->y + crop->h}
                                   : RECT{0, 0, frame->width, frame->height};
            decoding_->downscaler = std::make_unique<Downscaler>(
                video_.device_->device(), video_.device_->context(), static_cast<int>(description.Width),
                static_cast<int>(description.Height), area, *video_.scale_to_, description.Format);
        }
        if (std::shared_ptr<ID3D11Texture2D> scaled = decoding_->downscaler->scale(made.texture, made.subresource)) {
            made.texture = scaled.get();
            made.subresource = 0;
            made.width = video_.scale_to_->first;
            made.height = video_.scale_to_->second;
            made.prepared = true;
            made.keep = std::move(scaled);
        }
    } catch (const std::exception&) {
        decoding_->downscale_failed = true;
    }
}

// The status line for frames like `made`, given when it changes: the first
// frame of each way of decoding, and the film grain drawn or turned off.
void FfmpegNativeVideo::Session::report(const NativeFrame& made) {
    if (reported_ && grain_note_ == reported_grain_note_) {
        return;
    }
    reported_ = true;
    reported_grain_note_ = grain_note_;
    const QString decoder = QString::fromStdString(decoding_->codec_name);
    QStringList details{
        QString::fromUtf8("%1\xC3\x97%2 %3")
            .arg(made.width)
            .arg(made.height)
            .arg(made.ten_bit ? QStringLiteral("P010_10LE") : QStringLiteral("NV12")),
        QStringLiteral("memory:D3D11Memory"),
        decoding_->hardware ? QStringLiteral("%1 GPU: d3d11va %2").arg(side_name_, decoder)
                            : QStringLiteral("%1 software: %2").arg(side_name_, decoder)};
    if (grain_note_) {
        details.append(*grain_note_);
    }
    if (video_.shaded_ == NativeShading::sdr) {
        details.append(QString::fromUtf8("GPU HDR\xE2\x86\x92SDR \xC2\xB7 fixed Reinhard 1000\xE2\x86\x92" "100 nit"));
    } else if (video_.shaded_ == NativeShading::hdr) {
        details.append(QString::fromUtf8("GPU colours \xE2\x86\x92 PQ BT.2020, HDR kept"));
    }
    const std::lock_guard lock(video_.mutex_);
    video_.status_ = details.join(QString::fromUtf8(" \xC2\xB7 "));
}

}  // namespace vml::app
