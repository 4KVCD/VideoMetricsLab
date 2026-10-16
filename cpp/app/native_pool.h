// Video Compare's native playback (the Python app's LockedNativePool): one
// decoder per video (the source, and the selected test video with its
// neighbours), each test frame paired with the source frame of the same
// number before the one presenter draws either, in time with the source's
// soundtrack (with the wall clock where it cannot play). The frames are
// always paired here, never shown by separate sinks, so the two videos
// cannot drift apart.
#pragma once

#include <QFuture>
#include <QString>
#include <QWidget>

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "native_decoder.h"
#include "native_presentation.h"
#include "native_soundtrack.h"
#include "vml/frame_extract.h"

namespace vml::app {

class NativeDevice;
class VideoCompareView;

// A native window the presenter's window sits in, painted dark by Qt, which
// shows until the presenter draws its first frame; a zoomed frame is dragged
// on it (the view's zoom).
class PairedFrameWidget : public QWidget {
    Q_OBJECT

public:
    explicit PairedFrameWidget(VideoCompareView* view);
    void refresh_cursor();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    bool can_drag() const;

    VideoCompareView* view_;
    std::optional<QPointF> drag_from_;
};

class NativePool {
public:
    // The decoder of one video, from `position_ms` on: a test seam.
    using VideoFactory = std::function<NativeVideo*(const FrameComparison& comparison, FrameSide side,
                                                    const PreviewColorSettings& colors, int position_ms,
                                                    QObject* owner)>;
    // What a pool is made of, a test seam: the decoders, the presenter and
    // the soundtrack. By default FFmpeg's decoders on the session's device,
    // its presenter over the view, and the source's sound played in the app.
    struct Parts {
        VideoFactory make_video;
        std::function<std::unique_ptr<NativeOutput>(PairedFrameWidget& surface)> make_presentation;
        std::function<std::unique_ptr<Soundtrack>(const FrameComparison& source, int position_ms, bool enabled)>
            make_soundtrack;
    };

    // Plays `series` from `position_ms`, the test video `selected` shown.
    // Throws std::runtime_error when it cannot (the view then plays with
    // FFmpeg, saying why).
    NativePool(VideoCompareView& view, std::vector<FrameComparison> series, int selected, int position_ms,
               const PreviewColorSettings& colors, bool playing, bool source_native, Parts parts);
    ~NativePool();
    NativePool(const NativePool&) = delete;
    NativePool& operator=(const NativePool&) = delete;

    // The parts a pool plays with on this PC, or why native playback cannot
    // run here at all.
    static Parts default_parts(const PreviewColorSettings& colors);
    static std::optional<QString> unavailable();
    // What native playback's first start needs, made off the window's
    // thread: FFmpeg's libraries, the GPU's device and the presenter's
    // shaders. Made on the window's thread, the device alone held it up for
    // 79 ms (the display driver loading) as Video Compare first opened.
    // `load` starts it once and gives the same future after; `loaded` is
    // true once it is done, and on the offscreen platform, which has no
    // native playback.
    static QFuture<void> load();
    static bool loaded();
    // Lets go of the GPU's device: it goes once the decoders and the
    // presenter still ending have, with the rest of the GPU memory and the
    // RAM the display driver held for it (about 300 MB after 4K playback);
    // the next start makes it again (load). False, and nothing let go of,
    // while a load is under way.
    static bool unload();

    // Selects a test video: keeps the decoders still needed, stops the
    // others and starts the missing ones.
    void sync(int selected);
    void seek(int position_ms);
    void set_playing(bool playing);
    void show_source(bool showing);
    void set_audio_enabled(bool enabled);
    // Draws the frame again after the zoom or the view's size changed.
    void place();
    void resize();
    // Advances playback; returns the position shown (ms). Throws
    // std::runtime_error when a decoder or the presenter failed.
    int poll();
    // How long to wait (ms) for the next tick while playing, or nothing while
    // that frame is due but not decoded yet.
    std::optional<int> next_tick_ms() const;
    QString description() const;
    // Stops every decoder, the sound and the presenter, each on a thread of
    // its own.
    void stop();

    long long frame() const { return frame_; }
    bool buffering() const { return buffering_; }
    bool ended() const { return ended_; }
    bool paired_on_selected() const { return pair_ && pair_index_ == selected_; }
    std::size_t decoders() const { return entries_.size(); }

private:
    friend class NativePoolTest;

    void launch_missing();
    // poll's steps.
    long long shown_reach() const;
    void take_frames(const std::string& key, NativeVideo& player, long long target, long long reach);
    void keep_up(bool matched, long long target);
    void check_ended();
    bool choose_pair(long long target);
    void present(bool source);
    // Stops a decoder (counted against the limit while stopping), or the
    // presenter or the sound, on a thread of its own.
    void retire_video(NativeVideo* video);
    std::string distorted_key(int index) const { return "d:" + std::to_string(index); }

    VideoCompareView& view_;
    std::vector<FrameComparison> series_;
    PreviewColorSettings colors_;
    bool source_native_ = true;
    Parts parts_;
    int selected_ = 0;
    double fps_ = 0;
    int position_ = 0;
    long long frame_ = 0;
    bool playing_ = false;
    bool ended_ = false;
    bool closed_ = false;
    bool buffering_ = true;
    // Whether the pair wanted is the one on screen.
    bool shown_ = false;
    std::optional<double> late_since_;
    // The soundtrack clock's reading (ms) the last poll chose the frame by,
    // and when it was read.
    std::optional<double> clock_ms_;
    double clock_at_ = 0;
    bool showing_source_ = false;
    std::string source_key_;
    FrameComparison source_recipe_;
    std::vector<std::pair<std::string, std::pair<FrameComparison, FrameSide>>> desired_;
    std::map<std::string, NativeVideo*> entries_;
    std::map<std::string, std::map<long long, NativeFramePtr>> frames_;
    std::set<std::string> eos_;
    // Each video's shading, kept when its decoder goes, so that a test video
    // chosen again before its decoder is back is not shown unshaded.
    std::map<std::string, std::optional<std::array<float, 16>>> shadings_;
    std::optional<std::pair<NativeFramePtr, NativeFramePtr>> pair_;
    std::optional<int> pair_index_;
    double anchor_ = 0;
    long long anchor_frame_ = 0;
    bool audio_running_ = false;
    double audio_deadline_ = 0;
    std::map<std::string, QString> status_details_;
    PairedFrameWidget* surface_ = nullptr;
    std::unique_ptr<NativeOutput> output_;
    std::unique_ptr<Soundtrack> audio_;
};

}  // namespace vml::app
