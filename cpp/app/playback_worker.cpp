#include "playback_worker.h"

#include <deque>
#include <exception>
#include <string_view>
#include <utility>

#include "vml/platform/process.h"
#include "vml/python_text.h"

namespace vml::app {

namespace {

// Buffers of frames the view is done with, kept to read the next frames
// into: a worker's frames are at most three queued, three in the view's
// history and one being read, so a few spares keep them all in use.
constexpr std::size_t spare_frames = 4;

// FFmpeg's pipe to the worker, large enough for whole frames: through a
// small pipe, 4K RGBA came at 30 frames a second, through this, at 72 (the
// Python app's proc.FRAME_PIPE_BYTES).
constexpr std::size_t frame_pipe_bytes = 64 * 1024 * 1024;

// How each way of decoding is described once its first frame comes.
QString way_of(PlaybackProcessing processing) {
    switch (processing) {
        case PlaybackProcessing::interop: return QStringLiteral("GPU decode + GPU tone mapping/RGB");
        case PlaybackProcessing::transfer:
            return QStringLiteral("hardware decode + GPU tone mapping/RGB (host transfer)");
        case PlaybackProcessing::software: return QStringLiteral("software decode + GPU tone mapping/RGB");
        case PlaybackProcessing::cpu: return QStringLiteral("CPU fallback (GPU processing unavailable)");
    }
    return {};
}

// The last lines a process wrote to its error output, kept as its reader
// thread hands them over.
class LineTail {
public:
    explicit LineTail(std::size_t most) : most_(most) {}

    void add(std::string_view line) {
        const std::lock_guard lock(mutex_);
        lines_.emplace_back(line);
        if (lines_.size() > most_) {
            lines_.pop_front();
        }
    }
    // The lines, each ended by a newline.
    std::string text() const {
        const std::lock_guard lock(mutex_);
        std::string said;
        for (const std::string& line : lines_) {
            said += line + "\n";
        }
        return said;
    }

private:
    const std::size_t most_;
    mutable std::mutex mutex_;
    std::deque<std::string> lines_;
};

}  // namespace

StreamDecodeWorker::StreamDecodeWorker(std::string ffmpeg, FrameComparison comparison, FrameSide side,
                                       long long start_frame, PreviewColorSettings colors,
                                       std::optional<HardwareDecoding> plan, std::optional<Size> maximum,
                                       QObject* parent)
    : PlaybackDecoder(parent),
      ffmpeg_(std::move(ffmpeg)),
      comparison_(std::move(comparison)),
      side_(side),
      start_frame_(start_frame),
      colors_(colors),
      plan_(std::move(plan)),
      maximum_(maximum) {}

StreamDecodeWorker::~StreamDecodeWorker() {
    cancel();
    wait();
}

void StreamDecodeWorker::cancel() {
    {
        const std::lock_guard lock(mutex_);
        cancelled_ = true;
    }
    changed_.notify_all();
    cancel_.cancel();  // ends FFmpeg
}

std::vector<PlaybackFrame> StreamDecodeWorker::drain_through(long long frame) {
    std::vector<PlaybackFrame> due;
    {
        const std::lock_guard lock(mutex_);
        while (!frames_.empty() && frames_.front().number <= frame) {
            due.push_back(std::move(frames_.front()));
            frames_.pop_front();
        }
    }
    if (!due.empty()) {
        // Room for the decoder only now: woken at every look, a paused one
        // would wake 100 times a second for nothing.
        changed_.notify_all();
    }
    return due;
}

void StreamDecodeWorker::recycle(FramePixels pixels) {
    const std::lock_guard lock(mutex_);
    if (spare_.size() < spare_frames && pixels && pixels->size() == frame_bytes_) {
        spare_.push_back(std::move(pixels));
    }
}

std::optional<long long> StreamDecodeWorker::latest_frame_number() const {
    const std::lock_guard lock(mutex_);
    return frames_.empty() ? std::nullopt : std::optional(frames_.back().number);
}

bool StreamDecodeWorker::ended() const {
    const std::lock_guard lock(mutex_);
    return ended_;
}

QString StreamDecodeWorker::error() const {
    const std::lock_guard lock(mutex_);
    return error_;
}

std::vector<QString> StreamDecodeWorker::attempt_errors() const {
    const std::lock_guard lock(mutex_);
    return attempt_errors_;
}

FramePixels StreamDecodeWorker::buffer() {
    std::size_t bytes = 0;
    {
        const std::lock_guard lock(mutex_);
        if (!spare_.empty()) {
            FramePixels pixels = std::move(spare_.back());
            spare_.pop_back();
            return pixels;
        }
        bytes = frame_bytes_;
    }
    return std::make_shared<std::vector<unsigned char>>(bytes);
}

bool StreamDecodeWorker::put(long long frame, FramePixels pixels) {
    // Waits while three are queued; false once cancelled.
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [this] { return frames_.size() < 3 || cancelled_; });
    if (cancelled_) {
        return false;
    }
    frames_.push_back({frame, std::move(pixels)});
    return true;
}

void StreamDecodeWorker::run() {
    const Size size = playback_dimensions(comparison_, maximum_);
    {
        const std::lock_guard lock(mutex_);
        frame_bytes_ = static_cast<std::size_t>(size.first) * static_cast<std::size_t>(size.second) * 4;
    }
    const HardwareDecoding plan = plan_ ? *plan_
                                        : plan_hardware_decoding(DecodingCapabilities::remembered(ffmpeg_),
                                                                 GpuVendor::automatic, comparison_.source_info,
                                                                 comparison_.distorted_info);
    const std::optional<std::string>& accel = side_ == FrameSide::source ? plan.source : plan.distorted;
    std::vector<QString> errors;
    long long next_frame = start_frame_;
    for (const PlaybackProcessing processing : playback_modes(accel)) {
        if (cancelled()) {
            return;
        }
        const std::optional<Attempt> attempt = try_decoding(processing, plan, next_frame, errors);
        if (!attempt) {
            return;
        }
        if (attempt->gave_frames && attempt->error.isEmpty() && attempt->exit_code == 0) {
            const std::lock_guard lock(mutex_);
            ended_ = true;
            return;
        }
        // The frames queued are kept, and the next way starts at the first
        // frame not decoded: after a device lost mid-video too, without
        // replaying frames or retrying for ever.
        const QString detail =
            QString::fromStdString(std::string(python_strip(replace_invalid_utf8(attempt->error_output))));
        errors.push_back(!detail.isEmpty()          ? detail
                         : !attempt->error.isEmpty() ? attempt->error
                                                     : QStringLiteral("Decoder produced no frames."));
        const std::lock_guard lock(mutex_);
        attempt_errors_ = errors;
    }
    QString error = errors.empty() ? QStringLiteral("Decoder produced no frames.") : errors.back();
    {
        const std::lock_guard lock(mutex_);
        error_ = error;
    }
    emit failed(error.right(1500));
}

bool StreamDecodeWorker::cancelled() const {
    const std::lock_guard lock(mutex_);
    return cancelled_;
}

// Decodes with FFmpeg the way `processing` says (`plan` its decoders), from
// `next_frame` on, which it moves on with the frames queued. `errors` are
// the ways tried before. Nothing when the worker is to stop: cancelled, or
// its frames no longer wanted.
std::optional<StreamDecodeWorker::Attempt> StreamDecodeWorker::try_decoding(PlaybackProcessing processing,
                                                                            const HardwareDecoding& plan,
                                                                            long long& next_frame,
                                                                            const std::vector<QString>& errors) {
    // The end of FFmpeg's error output, its last 30 lines.
    const auto tail = std::make_shared<LineTail>(30);
    platform::ProcessOptions options;
    options.output_pipe_bytes = frame_pipe_bytes;
    options.on_error_line = [tail](std::string_view line) { tail->add(line); };
    Attempt attempt;
    try {
        platform::ChildProcess process = platform::ChildProcess::start(
            build_playback_command(ffmpeg_, comparison_, side_, next_frame, colors_, plan, maximum_, processing),
            options);
        const CancelToken::Attached attached = cancel_.attach(process);
        if (!read_frames(process, processing, next_frame, errors, attempt)) {
            return std::nullopt;
        }
        if (cancelled()) {
            return std::nullopt;
        }
        attempt.exit_code = process.wait();
    } catch (const std::exception& failure) {
        attempt.error = QString::fromUtf8(failure.what());
    }
    if (cancelled()) {
        return std::nullopt;
    }
    attempt.error_output = tail->text();
    return attempt;
}

// Reads `process`'s frames into the queue until it ends or the worker is
// cancelled, saying the way of decoding once the first comes (with the
// error of the way before, if any). False when the queue no longer wants
// them.
bool StreamDecodeWorker::read_frames(platform::ChildProcess& process, PlaybackProcessing processing,
                                     long long& next_frame, const std::vector<QString>& errors, Attempt& attempt) {
    while (!cancelled()) {
        // The large pipe holds a whole frame: a read or two each.
        FramePixels pixels = buffer();
        std::size_t read = 0;
        while (read < frame_bytes_) {
            const std::size_t got = process.read(pixels->data() + read, frame_bytes_ - read);
            if (got == 0) {
                break;
            }
            read += got;
        }
        if (read != frame_bytes_) {
            if (read > 0) {
                attempt.error = QStringLiteral("Decoder returned a truncated frame.");
            }
            return true;
        }
        if (!attempt.gave_frames) {
            attempt.gave_frames = true;
            QString detail = way_of(processing);
            if (!errors.empty()) {
                detail += QString::fromUtf8(" \xC2\xB7 retry: ") + errors.back().split('\n').front().left(180);
            }
            emit ready(detail);
        }
        if (!put(next_frame, std::move(pixels))) {
            return false;
        }
        ++next_frame;
    }
    return true;
}

}  // namespace vml::app
