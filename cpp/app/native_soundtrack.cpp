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

void FfmpegSoundtrack::run() {
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    // It cannot play: its clock goes too, so that the frames follow the
    // wall clock rather than a clock that stopped.
    const auto fail = [this](const QString& why) {
        const std::lock_guard lock(mutex_);
        failed_ = why;
        ready_ = false;
        clock_ms_.reset();
    };
    // Everything below is freed before COM goes.
    [&] {
        Reader reader;
        reader.format = avformat_alloc_context();
        if (reader.format == nullptr) {
            fail(QStringLiteral("Out of memory"));
            return;
        }
        reader.format->interrupt_callback = AVIOInterruptCB{&FfmpegSoundtrack::interrupted, this};
        const std::string name = platform::path_text(path_);
        int result = avformat_open_input(&reader.format, name.c_str(), nullptr, nullptr);
        if (result >= 0) {
            result = avformat_find_stream_info(reader.format, nullptr);
        }
        if (result < 0) {
            if (!stopping()) {
                fail(QStringLiteral("Could not open the soundtrack: %1").arg(error_text(result)));
            }
            return;
        }
        const AVCodec* decoder = nullptr;
        reader.stream = av_find_best_stream(reader.format, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
        if (reader.stream < 0 || decoder == nullptr) {
            fail(QStringLiteral("The source has no soundtrack the app plays"));
            return;
        }
        for (unsigned index = 0; index < reader.format->nb_streams; ++index) {
            if (static_cast<int>(index) != reader.stream) {
                reader.format->streams[index]->discard = AVDISCARD_ALL;
            }
        }
        const AVStream* stream = reader.format->streams[reader.stream];
        reader.time_base = stream->time_base;
        reader.file_start_ms =
            reader.format->start_time == AV_NOPTS_VALUE ? 0 : static_cast<double>(reader.format->start_time) / 1000;
        reader.codec = avcodec_alloc_context3(decoder);
        reader.packet = av_packet_alloc();
        reader.frame = av_frame_alloc();
        if (reader.codec == nullptr || reader.packet == nullptr || reader.frame == nullptr ||
            avcodec_parameters_to_context(reader.codec, stream->codecpar) < 0) {
            fail(QStringLiteral("Out of memory"));
            return;
        }
        reader.codec->pkt_timebase = stream->time_base;
        result = avcodec_open2(reader.codec, decoder, nullptr);
        if (result < 0) {
            fail(QStringLiteral("Could not decode the soundtrack: %1").arg(error_text(result)));
            return;
        }

        // The sound device: Windows' default, shared, in its own format.
        ComPtr<IMMDeviceEnumerator> devices;
        ComPtr<IMMDevice> device;
        ComPtr<IAudioClient> client;
        WAVEFORMATEX* mix = nullptr;
        if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&devices))) ||
            FAILED(devices->GetDefaultAudioEndpoint(eRender, eConsole, &device)) ||
            FAILED(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, &client)) ||
            FAILED(client->GetMixFormat(&mix))) {
            fail(QStringLiteral("No sound device"));
            return;
        }
        const int channels = mix->nChannels;
        const int rate = static_cast<int>(mix->nSamplesPerSec);
        bool floats = mix->wFormatTag == WAVE_FORMAT_IEEE_FLOAT;
        if (mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
            floats = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(mix)->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
        }
        const bool shorts = !floats && mix->wBitsPerSample == 16;
        EventHandle ready_event;
        HRESULT made = floats || shorts ? client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                                             AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
                                                                 AUDCLNT_STREAMFLAGS_NOPERSIST,
                                                             buffer_hns, 0, mix, nullptr)
                                        : E_FAIL;
        CoTaskMemFree(mix);
        UINT32 buffer_frames = 0;
        ComPtr<IAudioRenderClient> render;
        ComPtr<IAudioClock> clock;
        UINT64 frequency = 0;
        if (FAILED(made) || ready_event.handle == nullptr || FAILED(client->SetEventHandle(ready_event.handle)) ||
            FAILED(client->GetBufferSize(&buffer_frames)) || FAILED(client->GetService(IID_PPV_ARGS(&render))) ||
            FAILED(client->GetService(IID_PPV_ARGS(&clock))) || FAILED(clock->GetFrequency(&frequency)) ||
            frequency == 0) {
            fail(QStringLiteral("The sound device cannot play"));
            return;
        }
        const AVSampleFormat sample_format = floats ? AV_SAMPLE_FMT_FLT : AV_SAMPLE_FMT_S16;
        const std::size_t frame_bytes = static_cast<std::size_t>(channels) * (floats ? 4 : 2);
        AVChannelLayout out_layout{};
        av_channel_layout_default(&out_layout, channels);
        AVChannelLayout in_layout{};
        if (reader.codec->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
            av_channel_layout_default(&in_layout, reader.codec->ch_layout.nb_channels);
        } else {
            av_channel_layout_copy(&in_layout, &reader.codec->ch_layout);
        }
        result = swr_alloc_set_opts2(&reader.resampler, &out_layout, sample_format, rate, &in_layout,
                                     reader.codec->sample_fmt, reader.codec->sample_rate, 0, nullptr);
        av_channel_layout_uninit(&in_layout);
        av_channel_layout_uninit(&out_layout);
        if (result < 0 || swr_init(reader.resampler) < 0) {
            fail(QStringLiteral("The soundtrack cannot be converted for the sound device"));
            return;
        }

        // The sound decoded, in the device's format, not yet given to it.
        std::vector<std::uint8_t> queue;
        std::size_t queue_start = 0;  // bytes already given
        const auto queued_frames = [&] { return (queue.size() - queue_start) / frame_bytes; };
        // Where the first sample after the last seek is on the file's
        // timeline (ms); where the next sample decoded is; and the samples
        // before the seek's target, which are left out.
        double base_ms = 0;
        std::optional<double> next_ms;
        double target_ms = 0;
        bool started = false;
        bool ready = false;
        std::vector<std::uint8_t> converted;

        // Decodes some sound into the queue; at the soundtrack's end, marks it.
        const auto decode_some = [&]() -> bool {
            if (!reader.draining) {
                result = av_read_frame(reader.format, reader.packet);
                if (result == AVERROR_EOF) {
                    reader.draining = true;
                    avcodec_send_packet(reader.codec, nullptr);
                } else if (result < 0) {
                    if (!stopping()) {
                        fail(QStringLiteral("Could not read the soundtrack: %1").arg(error_text(result)));
                    }
                    return false;
                } else {
                    if (reader.packet->stream_index == reader.stream) {
                        avcodec_send_packet(reader.codec, reader.packet);
                    }
                    av_packet_unref(reader.packet);
                }
            }
            while (true) {
                result = avcodec_receive_frame(reader.codec, reader.frame);
                if (result == AVERROR(EAGAIN)) {
                    return true;
                }
                if (result < 0) {  // the end (or a broken stream: the sound ends there)
                    reader.ended = true;
                    return true;
                }
                std::int64_t stamp = reader.frame->best_effort_timestamp;
                if (stamp == AV_NOPTS_VALUE) {
                    stamp = reader.frame->pts;
                }
                if (!next_ms && stamp != AV_NOPTS_VALUE) {
                    next_ms = av_q2d(reader.time_base) * static_cast<double>(stamp) * 1000 - reader.file_start_ms;
                    if (*next_ms > target_ms) {
                        // The sound starts after the target: silence first.
                        const auto gap =
                            static_cast<std::size_t>(std::llround((*next_ms - target_ms) * rate / 1000)) * frame_bytes;
                        queue.insert(queue.end(), gap, 0);
                    }
                }
                if (!next_ms) {
                    next_ms = target_ms;
                }
                const int most = swr_get_out_samples(reader.resampler, reader.frame->nb_samples);
                converted.resize(static_cast<std::size_t>(std::max(most, 0)) * frame_bytes);
                std::uint8_t* out = converted.data();
                const int made_frames =
                    swr_convert(reader.resampler, &out, most, reader.frame->extended_data, reader.frame->nb_samples);
                av_frame_unref(reader.frame);
                if (made_frames <= 0) {
                    continue;
                }
                // Left out up to the target.
                std::size_t skip = 0;
                if (*next_ms < target_ms) {
                    skip = std::min(static_cast<std::size_t>(made_frames),
                                    static_cast<std::size_t>(std::llround((target_ms - *next_ms) * rate / 1000)));
                }
                queue.insert(queue.end(), converted.begin() + static_cast<std::ptrdiff_t>(skip * frame_bytes),
                             converted.begin() + static_cast<std::ptrdiff_t>(made_frames) * static_cast<std::ptrdiff_t>(frame_bytes));
                *next_ms += made_frames * 1000.0 / rate;
            }
        };

        while (!stopping()) {
            std::optional<double> seek;
            bool want_playing = false;
            bool enabled = true;
            {
                const std::lock_guard lock(mutex_);
                seek = std::exchange(wanted_ms_, std::nullopt);
                want_playing = want_playing_ && !failed_;
                enabled = enabled_;
            }
            if (seek) {
                if (started) {
                    client->Stop();
                    started = false;
                }
                client->Reset();
                queue.clear();
                queue_start = 0;
                reader.draining = false;
                reader.ended = false;
                avcodec_flush_buffers(reader.codec);
                swr_close(reader.resampler);
                swr_init(reader.resampler);
                target_ms = *seek;
                base_ms = *seek;
                next_ms.reset();
                ready = false;
                const auto stamp = static_cast<std::int64_t>((*seek + reader.file_start_ms) / 1000 / av_q2d(reader.time_base));
                av_seek_frame(reader.format, reader.stream, stamp, AVSEEK_FLAG_BACKWARD);
            }
            while (!reader.ended && queued_frames() * 1000.0 / rate < queued_ms) {
                if (!decode_some()) {
                    return;
                }
                if (stopping()) {
                    return;
                }
                const std::lock_guard lock(mutex_);
                if (wanted_ms_) {
                    break;  // another seek: made first
                }
            }
            if (!ready && (reader.ended || queued_frames() * 1000.0 / rate >= ready_ms)) {
                ready = true;
                const std::lock_guard lock(mutex_);
                if (!wanted_ms_) {
                    ready_ = true;
                    clock_ms_ = base_ms;
                }
            }
            if (want_playing && ready && !started) {
                if (FAILED(client->Start())) {
                    fail(QStringLiteral("The sound device cannot play"));
                    return;
                }
                started = true;
            } else if (!want_playing && started) {
                client->Stop();
                started = false;
            }
            if (!started) {
                std::unique_lock lock(mutex_);
                changed_.wait_for(lock, std::chrono::milliseconds(10));
                continue;
            }
            WaitForSingleObject(ready_event.handle, 20);
            UINT32 padding = 0;
            if (FAILED(client->GetCurrentPadding(&padding))) {
                fail(QStringLiteral("The sound device went"));
                return;
            }
            const UINT32 room = buffer_frames - padding;
            // Sound as far as decoded; past the soundtrack's end, silence, so
            // that its clock goes on timing the frames.
            const auto given = static_cast<UINT32>(
                std::min<std::size_t>(room, reader.ended ? room : queued_frames()));
            if (given > 0) {
                BYTE* data = nullptr;
                if (FAILED(render->GetBuffer(given, &data))) {
                    fail(QStringLiteral("The sound device went"));
                    return;
                }
                const std::size_t from_queue = std::min<std::size_t>(given, queued_frames());
                if (enabled) {
                    std::memcpy(data, queue.data() + queue_start, from_queue * frame_bytes);
                    std::memset(data + from_queue * frame_bytes, 0, (given - from_queue) * frame_bytes);
                } else {  // muted: silence, the clock still running
                    std::memset(data, 0, static_cast<std::size_t>(given) * frame_bytes);
                }
                render->ReleaseBuffer(given, 0);
                queue_start += from_queue * frame_bytes;
                if (queue_start > queue.size() / 2) {
                    queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(queue_start));
                    queue_start = 0;
                }
            }
            UINT64 position = 0;
            UINT64 counter = 0;
            if (SUCCEEDED(clock->GetPosition(&position, &counter))) {
                const std::lock_guard lock(mutex_);
                if (!wanted_ms_) {
                    clock_ms_ = base_ms + static_cast<double>(position) * 1000 / static_cast<double>(frequency);
                }
            }
        }
        if (started) {
            client->Stop();
        }
    }();
    if (SUCCEEDED(com)) {
        CoUninitialize();
    }
}

}  // namespace vml::app
