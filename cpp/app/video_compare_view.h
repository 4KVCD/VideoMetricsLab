// Video Compare's playback view (the Python app's
// vmaf_app/ui/video_compare_view.py): the source against the test video
// shown, frame by frame in step.
//
// Natively where it can (NativePool, Windows): FFmpeg's libraries decode on
// the GPU and the app's presenter draws the pair, HDR included. Otherwise
// FFmpeg itself decodes the source, the test video shown and its neighbours,
// each in a worker of its own a few frames ahead (StreamDecodeWorker), so
// that switching to a neighbour is instant; the view keeps the one playback
// clock and shows a frame only when the source and the test video both have
// that frame's number, the neighbours following the same clock without ever
// holding up the pair shown. The source's soundtrack plays with ffplay.
#pragma once

#include <QFutureWatcher>
#include <QImage>
#include <QPointF>
#include <QRect>
#include <QString>
#include <QWidget>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "playback_worker.h"
#ifdef VML_NATIVE_PLAYBACK
#include "native_pool.h"
#endif
#include "vml/frame_extract.h"
#include "vml/platform/process.h"
#include "zoom.h"

class QTimer;

namespace vml::app {

class VideoCompareView;

// Where FFmpeg's frames of one side are drawn: fitted to the view, or where
// the zoom puts them, with dark bars around. A zoomed frame is dragged here.
class StreamSurface : public QWidget {
    Q_OBJECT

public:
    explicit StreamSurface(VideoCompareView* view);

    // The frame it shows (`size` its pixels); painted when shown.
    void set_frame(FramePixels pixels, Size size);
    void clear_frame();
    bool shows(const FramePixels& pixels) const { return pixels && pixels == pixels_; }
    // Paints its new frame: only where the last was drawn, when drawn the
    // same way (the bars beside it stay as painted).
    void update_frame();
    void refresh_cursor();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    friend class PlaybackTest;  // the view's tests (tests/app)

    using DrawnKey = std::tuple<int, int, double, int, int>;
    DrawnKey drawn_key() const;
    QImage& frame_image();
    bool can_drag() const;

    VideoCompareView* view_;
    FramePixels pixels_;
    Size size_{0, 0};
    std::optional<QImage> image_;  // made from the pixels when first painted
    // Where the last paint drew a fitted frame, for what view and frame size.
    std::optional<std::pair<DrawnKey, QRect>> drawn_;
    std::optional<QPointF> drag_from_;
};

class VideoCompareView : public QWidget {
    Q_OBJECT

public:
    // Makes the decoder of one video: the view's tests make stand-ins.
    using DecoderFactory = std::function<PlaybackDecoder*(const FrameComparison& comparison, FrameSide side,
                                                          long long start_frame, const PreviewColorSettings& colors,
                                                          std::optional<Size> maximum, QObject* parent)>;

    // `ffmpeg`: the FFmpeg to decode with, asked for each start. `zoom`: the
    // tab's, shared with its still frames.
    VideoCompareView(std::function<std::string()> ffmpeg, Zoom& zoom, QWidget* parent = nullptr);
    ~VideoCompareView() override;

    // Whether a pair can be played, and if not, why (in English, as the
    // engine's messages are; tr_message translates it): a resolution test,
    // a file missing, or a raw stream whose guessed frame rate disagrees.
    static std::pair<bool, std::string> can_play(const FrameComparison& comparison);

    // Whether the black bars found for `pending` leave its pictures as they
    // are: none on either side, the size compared at unchanged, and neither
    // side boxed while they were looked for. Then the frames playing are
    // the ones the crops give.
    static bool crops_change_nothing(const FrameComparison& pending, const FrameComparison& cropped);

    // Shows `comparison` (one of `series`, the pairs that can play) at
    // `position_ms`. Another test video of the same series keeps the
    // decoders running; so does a series whose black bars, found since,
    // change nothing (crops_change_nothing). False when it cannot be played.
    bool load(const FrameComparison& comparison, int position_ms, bool playing, const PreviewColorSettings& colors,
              const std::vector<FrameComparison>& series, bool source_native);
    // Stops playback and shows nothing.
    void clear();
    void set_position(int position_ms);
    void set_playing(bool playing);
    // Shows the source's frame (true) or the test video's.
    void show_source(bool showing);
    void set_audio_enabled(bool enabled);
    // Decodes again if the colours shown change.
    void set_color_settings(const PreviewColorSettings& colors);
    // How many test videos are kept decoding: extra neighbours are stopped,
    // missing ones started; the pair shown is never touched.
    void set_decoded_videos(int count);
    // Shows the frame again after the zoom, or the part in view, changed.
    void zoom_changed();

    bool is_playing() const { return is_playing_; }
    // Whether FFmpeg itself plays the pair (an SDR preview), not native
    // playback.
    bool plays_with_ffmpeg() const { return pool_active_; }
    // Whether playing was asked for (it may still be buffering).
    bool playback_requested() const { return wanted_playing_; }
    // The position shown, in ms from the first frame.
    int position() const;
    // The decoders still running, those stopping included.
    std::vector<QThread*> live_workers() const;
    // Whether frames are fitted to the view, not zoomed.
    bool fitted() const { return !zoom_.factor; }
    // Where a `frame`-pixel picture of the pair is drawn: at the
    // comparison's scale, so that a source decoded at its own size beside a
    // smaller encode is drawn at the encode's size, and one of another shape
    // keeps its shape. Nothing when nothing is loaded.
    std::optional<Placement> zoom_placement(Size frame) const;
    // The zoom, the view's size, the frame's size and the ratio to use for a
    // `frame`-pixel picture (by default the comparison's own size).
    struct ZoomContext {
        Zoom::Extent view;
        Zoom::Extent frame;
        double ratio;
    };
    std::optional<ZoomContext> zoom_context(std::optional<Size> frame = std::nullopt) const;
    Zoom& zoom() { return zoom_; }

    DecoderFactory make_decoder;
    // The playback clock, in seconds (a stand-in in the tests).
    std::function<double()> clock;
#ifdef VML_NATIVE_PLAYBACK
    // What native playback plays with (decoders, presenter, soundtrack), or
    // why it cannot run here: the tests give stand-ins.
    std::function<std::optional<NativePool::Parts>(const PreviewColorSettings& colors, QString& why)> native_parts;
    // Where a native frame of `comparison`'s `side` is drawn: the part of it
    // shown and where in the view, in device pixels.
    NativeOutput::Placement native_placement(const NativeFrame& frame, const FrameComparison& comparison,
                                             FrameSide side) const;
#endif

signals:
    void position_changed(int position_ms);
    void playing_changed(bool playing);
    void status_changed(const QString& status);

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool event(QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private:
    friend class PlaybackTest;  // the view's tests (tests/app)
    friend class NativePool;    // native playback, which plays on the view's clock and limits
    friend class NativePoolTest;  // its tests (tests/app)
    friend class FramePanelTest;  // the panel's tests: which decoders play (tests/app)

    // What a decoder decodes: a pair (the source's recipe, or a test video's
    // comparison) and its side.
    struct Wanted {
        FrameComparison comparison;
        FrameSide side;
    };

    void wake();
    void pace(bool settled);
    int busy_tick_ms() const;
    std::optional<Size> wanted_maximum() const;
    void follow_decode_size();
    bool decode_size_changes(std::optional<Size> maximum) const;
    void restart_decoder(bool realtime);
    void start_ffmpeg(bool realtime, bool keep_frame = false, const QString& reason = {});
    // Native playback failed: FFmpeg plays on, saying why.
    void native_failed(const std::exception& error);
    // Whether native playback plays.
    bool native_active() const;
    // Advances native playback.
    void native_tick();
    void status(const QString& text);
    int decoder_limit() const { return 1 + decoded_videos_; }
    FrameComparison source_recipe() const;
    void sync_pool();
    void launch_missing();
    void stream_ready(const std::string& key, PlaybackDecoder* worker, const QString& detail);
    void stream_failed(const std::string& key, PlaybackDecoder* worker, const QString& error);
    void stream_finished(const std::string& key, PlaybackDecoder* worker);
    // Lets go of a decoder taken out of pool_ (cancelled first): deleted
    // once, now or when its thread's finished comes (ended_).
    void retire(PlaybackDecoder* worker);
    long long target_frame() const;
    void tick();
    void pool_status();
    void check_end();
    void start_audio(long long frame);
    void fell_behind();
    void stop_audio();
    void stop_decoder(bool keep_frame = false);
    std::string source_key() const;

    std::function<std::string()> ffmpeg_;
    Zoom& zoom_;
    std::optional<FrameComparison> comparison_;
    PreviewColorSettings colors_;
    std::vector<FrameComparison> series_;
    int selected_ = 0;
    bool source_native_ = true;
    long long frame_ = 0;
    bool wanted_playing_ = false;
    bool is_playing_ = false;
    bool showing_source_ = false;
    bool audio_enabled_ = true;
    std::optional<platform::ChildProcess> audio_;
    // The soundtrack is to start (again) with the frames, not before
    // audio_resume_at_ (playback fell behind until then).
    bool audio_due_ = false;
    double audio_resume_at_ = 0.0;
    // The decoders, by key: the source's ("s:" and its file, crop and size)
    // and each test video's ("d:" and its index in the series).
    std::map<std::string, PlaybackDecoder*> pool_;
    // Decoders being stopped, which still count against the limit.
    std::set<QThread*> retired_;
    // The decoders in pool_ whose thread's finished has been handled
    // (stream_finished kept them for their frames): one retired goes at
    // once, the others when their finished comes. isRunning() cannot tell
    // the two apart: it is false a moment before finished is sent, so a
    // decoder deleted on it would be deleted again by stream_finished.
    std::set<PlaybackDecoder*> ended_;
    // Native playback's presenter being ended, which counts only as running.
    std::set<QThread*> stopping_;
#ifdef VML_NATIVE_PLAYBACK
    std::unique_ptr<NativePool> native_pool_;
    // What native playback first needs being made (NativePool::load).
    QFutureWatcher<void>* native_loading_ = nullptr;
#endif
    // Why FFmpeg plays, not native playback: said in the status.
    QString pool_reason_;
    std::map<std::string, std::map<long long, FramePixels>> history_;
    std::map<std::string, QString> details_;
    std::map<std::string, QString> failures_;
    std::vector<std::pair<std::string, Wanted>> desired_;  // in order: the source, then the test videos
    long long clock_frame_ = 0;
    std::optional<double> clock_started_;
    long long presented_ = -1;
    bool buffering_ = true;
    bool pool_active_ = false;
    std::optional<Size> pool_maximum_;
    QString last_status_;
    int decoded_videos_;
    bool on_screen_ = false;
    QTimer* pool_timer_ = nullptr;
    QTimer* resize_timer_ = nullptr;
    StreamSurface* source_surface_ = nullptr;
    StreamSurface* distorted_surface_ = nullptr;
};

}  // namespace vml::app
