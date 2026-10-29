// Video Compare's videos decoded by FFmpeg for playback (the Python app's
// vmaf_app/ui/playback_worker.py): one worker per video, the source, the
// test video shown and its neighbours, each a few frames ahead.
//
// The workers have no clocks of their own: the view keeps the one playback
// clock, and takes the frames due from each (drain_through). Picture bytes
// never go through signals.
#pragma once

#include <QString>
#include <QThread>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "vml/cancel.h"
#include "vml/frame_extract.h"
#include "vml/hardware_decoding.h"
#include "vml/video_playback.h"

namespace vml::platform {
class ChildProcess;
}

namespace vml::app {

// A frame's pixels: 8-bit RGBA, row after row.
using FramePixels = std::shared_ptr<std::vector<unsigned char>>;

// A decoded frame: its number, counted from the video's first frame.
struct PlaybackFrame {
    long long number = 0;
    FramePixels pixels;
};

// What the view needs of a video's decoder; a stand-in replaces it in the
// view's tests.
class PlaybackDecoder : public QThread {
    Q_OBJECT

public:
    using QThread::QThread;

    // Stops decoding, ending FFmpeg.
    virtual void cancel() = 0;
    // The frames decoded up to `frame`, handed over; later ones stay queued.
    virtual std::vector<PlaybackFrame> drain_through(long long frame) = 0;
    // A frame's pixels the view no longer shows, read into again.
    virtual void recycle(FramePixels pixels) = 0;
    // The newest frame decoded and queued, if any.
    virtual std::optional<long long> latest_frame_number() const = 0;

signals:
    // Its first frame came, decoded as `detail` says.
    void ready(const QString& detail);
    // Every way of decoding it failed; the last's error.
    void failed(const QString& error);
};

// Decodes one video of Video Compare with FFmpeg into frames for the view,
// on a thread of its own, at most three frames ahead. A way of decoding that
// fails gives way to the next (playback_modes), from the first frame not
// yet decoded: the frames queued are kept, none is decoded twice.
class StreamDecodeWorker : public PlaybackDecoder {
    Q_OBJECT

public:
    // `plan`: the hardware decoders to use; nothing to plan them here (on
    // this thread: asking FFmpeg which it has takes a process).
    StreamDecodeWorker(std::string ffmpeg, FrameComparison comparison, FrameSide side, long long start_frame,
                       PreviewColorSettings colors, std::optional<HardwareDecoding> plan,
                       std::optional<Size> maximum, QObject* parent = nullptr);
    ~StreamDecodeWorker() override;

    void cancel() override;
    std::vector<PlaybackFrame> drain_through(long long frame) override;
    void recycle(FramePixels pixels) override;
    std::optional<long long> latest_frame_number() const override;

    // Whether the video ended (FFmpeg finished cleanly after frames).
    bool ended() const;
    // Why it could not play, once failed().
    QString error() const;
    // Each failed way's error, in order.
    std::vector<QString> attempt_errors() const;

protected:
    void run() override;

private:
    // One way of decoding tried: whether it gave a frame, why it failed if
    // it did, FFmpeg's exit code, and the end of its error output.
    struct Attempt {
        bool gave_frames = false;
        QString error;
        std::optional<int> exit_code;
        std::string error_output;
    };

    FramePixels buffer();
    bool put(long long frame, FramePixels pixels);
    bool cancelled() const;
    std::optional<Attempt> try_decoding(PlaybackProcessing processing, const HardwareDecoding& plan,
                                        long long& next_frame, const std::vector<QString>& errors);
    bool read_frames(platform::ChildProcess& process, PlaybackProcessing processing, long long& next_frame,
                     const std::vector<QString>& errors, Attempt& attempt);

    std::string ffmpeg_;
    FrameComparison comparison_;
    FrameSide side_;
    long long start_frame_;
    PreviewColorSettings colors_;
    std::optional<HardwareDecoding> plan_;
    std::optional<Size> maximum_;
    CancelToken cancel_;
    std::atomic<bool> stopping_{false};  // set by cancel(), for the reads' gathers

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<PlaybackFrame> frames_;
    std::vector<FramePixels> spare_;
    std::size_t frame_bytes_ = 0;
    bool cancelled_ = false;
    bool ended_ = false;
    QString error_;
    std::vector<QString> attempt_errors_;
};

}  // namespace vml::app
