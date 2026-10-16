#include "native_soundtrack.h"

#include <audioclient.h>
#include <mmdeviceapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "ffmpeg_api.h"
#include "vml/platform/system.h"

using Microsoft::WRL::ComPtr;

namespace vml::app {

namespace {

// How much decoded sound is kept ahead of the device, and how much a seek
// waits for before the soundtrack counts as ready to play.
constexpr double queued_ms = 300;
constexpr double ready_ms = 100;
// The device's buffer: 100 ms (in 100 ns units).
constexpr REFERENCE_TIME buffer_hns = 1'000'000;

double seconds_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

QString error_text(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return QString::fromUtf8(text.data());
}

// What reads and decodes the soundtrack.
struct Reader {
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwrContext* resampler = nullptr;
    int stream = -1;
    AVRational time_base{};
    // The file's start on its timeline (ms): positions count from it.
    double file_start_ms = 0;
    bool draining = false;
    bool ended = false;

    Reader() = default;
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    ~Reader() {
        swr_free(&resampler);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec);
        avformat_close_input(&format);
    }
};

// Closes an event handle.
struct EventHandle {
    HANDLE handle = nullptr;
    EventHandle() : handle(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {}
    ~EventHandle() {
        if (handle != nullptr) {
            CloseHandle(handle);
        }
    }
    EventHandle(const EventHandle&) = delete;
    EventHandle& operator=(const EventHandle&) = delete;
};

}  // namespace

FfmpegSoundtrack::FfmpegSoundtrack(std::filesystem::path path, int start_ms, bool enabled) : path_(std::move(path)) {
    enabled_ = enabled;
    wanted_ms_ = std::max(0, start_ms);  // counted from the file's start until the offset is known
    thread_ = std::thread([this] { run(); });
}

FfmpegSoundtrack::~FfmpegSoundtrack() {
    {
        const std::lock_guard lock(mutex_);
        stop_ = true;
    }
    changed_.notify_all();
    thread_.join();
}

bool FfmpegSoundtrack::ready() const {
    const std::lock_guard lock(mutex_);
    return ready_;
}

std::optional<QString> FfmpegSoundtrack::failed() const {
    const std::lock_guard lock(mutex_);
    return failed_;
}

void FfmpegSoundtrack::set_failed(const QString& why) {
    {
        const std::lock_guard lock(mutex_);
        failed_ = why;
        want_playing_ = false;
        clock_ms_.reset();
    }
    changed_.notify_all();
    clock_.set_playing(false);
}

void FfmpegSoundtrack::set_offset(int first_ms, int position_ms) {
    clock_.set_offset(first_ms);
    if (!failed()) {
        seek(position_ms);
    }
}

void FfmpegSoundtrack::seek(int position_ms) {
    clock_.restart();
    {
        const std::lock_guard lock(mutex_);
        ready_ = false;
        clock_ms_.reset();
        wanted_ms_ = std::max(0, position_ms + clock_.offset_ms());
    }
    changed_.notify_all();
}

void FfmpegSoundtrack::set_playing(bool playing) {
    if (failed()) {
        return;
    }
    clock_.set_playing(playing);
    {
        const std::lock_guard lock(mutex_);
        want_playing_ = playing;
    }
    changed_.notify_all();
}

void FfmpegSoundtrack::set_enabled(bool enabled) {
    const std::lock_guard lock(mutex_);
    enabled_ = enabled;
}

std::optional<double> FfmpegSoundtrack::poll() {
    std::optional<double> reading;
    {
        const std::lock_guard lock(mutex_);
        reading = clock_ms_;
    }
    return clock_.poll(reading, seconds_now());
}

bool FfmpegSoundtrack::stopping() const {
    const std::lock_guard lock(mutex_);
    return stop_;
}

int FfmpegSoundtrack::interrupted(void* self) { return static_cast<FfmpegSoundtrack*>(self)->stopping() ? 1 : 0; }

// The soundtrack played, on its thread: the file read and decoded, the sound
// device fed, seeks made, and the device's clock read.
class FfmpegSoundtrack::Player {
public:
    explicit Player(FfmpegSoundtrack& soundtrack) : soundtrack_(soundtrack) {}
    Player(const Player&) = delete;
    Player& operator=(const Player&) = delete;

    // Opens the soundtrack and the sound device; false when either cannot
    // be (fail() then says why).
    bool open() { return open_file() && open_device(); }
    // Plays until stopped, or until the device or the file fails.
    void play();

private:
    bool open_file();
    bool open_device();
    void fail(const QString& why);
    std::size_t queued_frames() const { return (queue_.size() - queue_start_) / frame_bytes_; }
    double queued_sound_ms() const { return static_cast<double>(queued_frames()) * 1000.0 / rate_; }
    void make_seek(double position_ms);
    bool fill_queue();
    bool decode_some();
    void take_sound();
    bool feed_device(bool enabled);
    void read_clock();

    FfmpegSoundtrack& soundtrack_;
    Reader reader_;

    // The sound device: Windows' default, shared, in its own format.
    ComPtr<IMMDeviceEnumerator> devices_;
    ComPtr<IMMDevice> device_;
    ComPtr<IAudioClient> client_;
    int channels_ = 0;
    int rate_ = 0;
    std::size_t frame_bytes_ = 0;
    EventHandle ready_event_;
    UINT32 buffer_frames_ = 0;
    ComPtr<IAudioRenderClient> render_;
    ComPtr<IAudioClock> clock_;
    UINT64 frequency_ = 0;

    // The sound decoded, in the device's format, not yet given to it.
    std::vector<std::uint8_t> queue_;
    std::size_t queue_start_ = 0;  // bytes already given
    // Where the first sample after the last seek is on the file's timeline
    // (ms); where the next sample decoded is; and the samples before the
    // seek's target, which are left out.
    double base_ms_ = 0;
    std::optional<double> next_ms_;
    double target_ms_ = 0;
    bool started_ = false;
    bool ready_ = false;
    std::vector<std::uint8_t> converted_;
};

void FfmpegSoundtrack::run() {
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    {
        // Freed before COM goes.
        Player player(*this);
        if (player.open()) {
            player.play();
        }
    }
    if (SUCCEEDED(com)) {
        CoUninitialize();
    }
}

// It cannot play: its clock goes too, so that the frames follow the wall
// clock rather than a clock that stopped.
void FfmpegSoundtrack::Player::fail(const QString& why) {
    const std::lock_guard lock(soundtrack_.mutex_);
    soundtrack_.failed_ = why;
    soundtrack_.ready_ = false;
    soundtrack_.clock_ms_.reset();
}

bool FfmpegSoundtrack::Player::open_file() {
    reader_.format = avformat_alloc_context();
    if (reader_.format == nullptr) {
        fail(QStringLiteral("Out of memory"));
        return false;
    }
    reader_.format->interrupt_callback = AVIOInterruptCB{&FfmpegSoundtrack::interrupted, &soundtrack_};
    const std::string name = platform::path_text(soundtrack_.path_);
    int result = avformat_open_input(&reader_.format, name.c_str(), nullptr, nullptr);
    if (result >= 0) {
        result = avformat_find_stream_info(reader_.format, nullptr);
    }
    if (result < 0) {
        if (!soundtrack_.stopping()) {
            fail(QStringLiteral("Could not open the soundtrack: %1").arg(error_text(result)));
        }
        return false;
    }
    const AVCodec* decoder = nullptr;
    reader_.stream = av_find_best_stream(reader_.format, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
    if (reader_.stream < 0 || decoder == nullptr) {
        fail(QStringLiteral("The source has no soundtrack the app plays"));
        return false;
    }
    for (unsigned index = 0; index < reader_.format->nb_streams; ++index) {
        if (static_cast<int>(index) != reader_.stream) {
            reader_.format->streams[index]->discard = AVDISCARD_ALL;
        }
    }
    const AVStream* stream = reader_.format->streams[reader_.stream];
    reader_.time_base = stream->time_base;
    reader_.file_start_ms =
        reader_.format->start_time == AV_NOPTS_VALUE ? 0 : static_cast<double>(reader_.format->start_time) / 1000;
    reader_.codec = avcodec_alloc_context3(decoder);
    reader_.packet = av_packet_alloc();
    reader_.frame = av_frame_alloc();
    if (reader_.codec == nullptr || reader_.packet == nullptr || reader_.frame == nullptr ||
        avcodec_parameters_to_context(reader_.codec, stream->codecpar) < 0) {
        fail(QStringLiteral("Out of memory"));
        return false;
    }
    reader_.codec->pkt_timebase = stream->time_base;
    result = avcodec_open2(reader_.codec, decoder, nullptr);
    if (result < 0) {
        fail(QStringLiteral("Could not decode the soundtrack: %1").arg(error_text(result)));
        return false;
    }
    return true;
}

// The sound device, and the resampler that turns the soundtrack into its
// format.
bool FfmpegSoundtrack::Player::open_device() {
    WAVEFORMATEX* mix = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&devices_))) ||
        FAILED(devices_->GetDefaultAudioEndpoint(eRender, eConsole, &device_)) ||
        FAILED(device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, &client_)) ||
        FAILED(client_->GetMixFormat(&mix))) {
        fail(QStringLiteral("No sound device"));
        return false;
    }
    channels_ = mix->nChannels;
    rate_ = static_cast<int>(mix->nSamplesPerSec);
    bool floats = mix->wFormatTag == WAVE_FORMAT_IEEE_FLOAT;
    if (mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        floats = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(mix)->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
    }
    const bool shorts = !floats && mix->wBitsPerSample == 16;
    const HRESULT made =
        floats || shorts ? client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                               AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_NOPERSIST,
                                               buffer_hns, 0, mix, nullptr)
                         : E_FAIL;
    CoTaskMemFree(mix);
    if (FAILED(made) || ready_event_.handle == nullptr || FAILED(client_->SetEventHandle(ready_event_.handle)) ||
        FAILED(client_->GetBufferSize(&buffer_frames_)) || FAILED(client_->GetService(IID_PPV_ARGS(&render_))) ||
        FAILED(client_->GetService(IID_PPV_ARGS(&clock_))) || FAILED(clock_->GetFrequency(&frequency_)) ||
        frequency_ == 0) {
        fail(QStringLiteral("The sound device cannot play"));
        return false;
    }
    const AVSampleFormat sample_format = floats ? AV_SAMPLE_FMT_FLT : AV_SAMPLE_FMT_S16;
    frame_bytes_ = static_cast<std::size_t>(channels_) * (floats ? 4 : 2);
    AVChannelLayout out_layout{};
    av_channel_layout_default(&out_layout, channels_);
    AVChannelLayout in_layout{};
    if (reader_.codec->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
        av_channel_layout_default(&in_layout, reader_.codec->ch_layout.nb_channels);
    } else {
        av_channel_layout_copy(&in_layout, &reader_.codec->ch_layout);
    }
    const int result = swr_alloc_set_opts2(&reader_.resampler, &out_layout, sample_format, rate_, &in_layout,
                                           reader_.codec->sample_fmt, reader_.codec->sample_rate, 0, nullptr);
    av_channel_layout_uninit(&in_layout);
    av_channel_layout_uninit(&out_layout);
    if (result < 0 || swr_init(reader_.resampler) < 0) {
        fail(QStringLiteral("The soundtrack cannot be converted for the sound device"));
        return false;
    }
    return true;
}

void FfmpegSoundtrack::Player::play() {
    while (!soundtrack_.stopping()) {
        std::optional<double> seek;
        bool want_playing = false;
        bool enabled = true;
        {
            const std::lock_guard lock(soundtrack_.mutex_);
            seek = std::exchange(soundtrack_.wanted_ms_, std::nullopt);
            want_playing = soundtrack_.want_playing_ && !soundtrack_.failed_;
            enabled = soundtrack_.enabled_;
        }
        if (seek) {
            make_seek(*seek);
        }
        if (!fill_queue()) {
            return;
        }
        if (!ready_ && (reader_.ended || queued_sound_ms() >= ready_ms)) {
            ready_ = true;
            const std::lock_guard lock(soundtrack_.mutex_);
            if (!soundtrack_.wanted_ms_) {
                soundtrack_.ready_ = true;
                soundtrack_.clock_ms_ = base_ms_;
            }
        }
        if (want_playing && ready_ && !started_) {
            if (FAILED(client_->Start())) {
                fail(QStringLiteral("The sound device cannot play"));
                return;
            }
            started_ = true;
        } else if (!want_playing && started_) {
            client_->Stop();
            started_ = false;
        }
        if (!started_) {
            std::unique_lock lock(soundtrack_.mutex_);
            soundtrack_.changed_.wait_for(lock, std::chrono::milliseconds(10));
            continue;
        }
        WaitForSingleObject(ready_event_.handle, 20);
        if (!feed_device(enabled)) {
            return;
        }
        read_clock();
    }
    if (started_) {
        client_->Stop();
    }
}

// Seeks to `position_ms` on the file's timeline: the device and the queue
// emptied, the decoder started again from the key frame before it.
void FfmpegSoundtrack::Player::make_seek(double position_ms) {
    if (started_) {
        client_->Stop();
        started_ = false;
    }
    client_->Reset();
    queue_.clear();
    queue_start_ = 0;
    reader_.draining = false;
    reader_.ended = false;
    avcodec_flush_buffers(reader_.codec);
    swr_close(reader_.resampler);
    swr_init(reader_.resampler);
    target_ms_ = position_ms;
    base_ms_ = position_ms;
    next_ms_.reset();
    ready_ = false;
    const auto stamp = static_cast<std::int64_t>((position_ms + reader_.file_start_ms) / 1000 / av_q2d(reader_.time_base));
    av_seek_frame(reader_.format, reader_.stream, stamp, AVSEEK_FLAG_BACKWARD);
}

// Decodes until queued_ms of sound is queued, the soundtrack ends or another
// seek is asked for. False when playing is to end (stopped, or the file
// could not be read).
bool FfmpegSoundtrack::Player::fill_queue() {
    while (!reader_.ended && queued_sound_ms() < queued_ms) {
        if (!decode_some() || soundtrack_.stopping()) {
            return false;
        }
        const std::lock_guard lock(soundtrack_.mutex_);
        if (soundtrack_.wanted_ms_) {
            break;  // another seek: made first
        }
    }
    return true;
}

// Decodes some sound into the queue; at the soundtrack's end, marks it.
// False when the file could not be read (fail() then says why).
bool FfmpegSoundtrack::Player::decode_some() {
    if (!reader_.draining) {
        const int result = av_read_frame(reader_.format, reader_.packet);
        if (result == AVERROR_EOF) {
            reader_.draining = true;
            avcodec_send_packet(reader_.codec, nullptr);
        } else if (result < 0) {
            if (!soundtrack_.stopping()) {
                fail(QStringLiteral("Could not read the soundtrack: %1").arg(error_text(result)));
            }
            return false;
        } else {
            if (reader_.packet->stream_index == reader_.stream) {
                avcodec_send_packet(reader_.codec, reader_.packet);
            }
            av_packet_unref(reader_.packet);
        }
    }
    while (true) {
        const int result = avcodec_receive_frame(reader_.codec, reader_.frame);
        if (result == AVERROR(EAGAIN)) {
            return true;
        }
        if (result < 0) {  // the end (or a broken stream: the sound ends there)
            reader_.ended = true;
            return true;
        }
        take_sound();
    }
}

// Queues the sound decoded into reader_.frame, converted to the device's
// format, from the seek's target on: silence first where the sound starts
// after it.
void FfmpegSoundtrack::Player::take_sound() {
    std::int64_t stamp = reader_.frame->best_effort_timestamp;
    if (stamp == AV_NOPTS_VALUE) {
        stamp = reader_.frame->pts;
    }
    if (!next_ms_ && stamp != AV_NOPTS_VALUE) {
        next_ms_ = av_q2d(reader_.time_base) * static_cast<double>(stamp) * 1000 - reader_.file_start_ms;
        if (*next_ms_ > target_ms_) {
            const auto gap =
                static_cast<std::size_t>(std::llround((*next_ms_ - target_ms_) * rate_ / 1000)) * frame_bytes_;
            queue_.insert(queue_.end(), gap, 0);
        }
    }
    if (!next_ms_) {
        next_ms_ = target_ms_;
    }
    const int most = swr_get_out_samples(reader_.resampler, reader_.frame->nb_samples);
    converted_.resize(static_cast<std::size_t>(std::max(most, 0)) * frame_bytes_);
    std::uint8_t* out = converted_.data();
    const int made_frames =
        swr_convert(reader_.resampler, &out, most, reader_.frame->extended_data, reader_.frame->nb_samples);
    av_frame_unref(reader_.frame);
    if (made_frames <= 0) {
        return;
    }
    // Left out up to the target.
    std::size_t skip = 0;
    if (*next_ms_ < target_ms_) {
        skip = std::min(static_cast<std::size_t>(made_frames),
                        static_cast<std::size_t>(std::llround((target_ms_ - *next_ms_) * rate_ / 1000)));
    }
    queue_.insert(queue_.end(), converted_.begin() + static_cast<std::ptrdiff_t>(skip * frame_bytes_),
                  converted_.begin() + static_cast<std::ptrdiff_t>(made_frames) * static_cast<std::ptrdiff_t>(frame_bytes_));
    *next_ms_ += made_frames * 1000.0 / rate_;
}

// Gives the device as much sound as it has room for: as far as decoded,
// then, past the soundtrack's end, silence, so that its clock goes on timing
// the frames; silence too while muted (not `enabled`). False when the device
// went (fail() then says so).
bool FfmpegSoundtrack::Player::feed_device(bool enabled) {
    UINT32 padding = 0;
    if (FAILED(client_->GetCurrentPadding(&padding))) {
        fail(QStringLiteral("The sound device went"));
        return false;
    }
    const UINT32 room = buffer_frames_ - padding;
    const auto given = static_cast<UINT32>(std::min<std::size_t>(room, reader_.ended ? room : queued_frames()));
    if (given == 0) {
        return true;
    }
    BYTE* data = nullptr;
    if (FAILED(render_->GetBuffer(given, &data))) {
        fail(QStringLiteral("The sound device went"));
        return false;
    }
    const std::size_t from_queue = std::min<std::size_t>(given, queued_frames());
    if (enabled) {
        std::memcpy(data, queue_.data() + queue_start_, from_queue * frame_bytes_);
        std::memset(data + from_queue * frame_bytes_, 0, (given - from_queue) * frame_bytes_);
    } else {
        std::memset(data, 0, static_cast<std::size_t>(given) * frame_bytes_);
    }
    render_->ReleaseBuffer(given, 0);
    queue_start_ += from_queue * frame_bytes_;
    if (queue_start_ > queue_.size() / 2) {
        queue_.erase(queue_.begin(), queue_.begin() + static_cast<std::ptrdiff_t>(queue_start_));
        queue_start_ = 0;
    }
    return true;
}

// The clock: where the device has played to, on the file's timeline (ms),
// unless a seek waits.
void FfmpegSoundtrack::Player::read_clock() {
    UINT64 position = 0;
    UINT64 counter = 0;
    if (SUCCEEDED(clock_->GetPosition(&position, &counter))) {
        const std::lock_guard lock(soundtrack_.mutex_);
        if (!soundtrack_.wanted_ms_) {
            soundtrack_.clock_ms_ = base_ms_ + static_cast<double>(position) * 1000 / static_cast<double>(frequency_);
        }
    }
}

}  // namespace vml::app
