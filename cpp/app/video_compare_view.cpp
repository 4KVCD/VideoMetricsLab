#include "video_compare_view.h"

#include <QEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QTimer>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>

#include "vml/geometry.h"
#include "vml/platform/system.h"
#include "vml/video_playback.h"

namespace vml::app {

namespace {

// How often playback is looked after while frames are due (playing, a
// seek, another test video, decoders starting), and once paused on the
// frame wanted: at 10 ms throughout, a paused view would cost 2-4% of a core.
constexpr int busy_tick_ms_default = 10;
constexpr int settled_tick_ms = 100;
// How long playback must keep up before its soundtrack, stopped when the
// frames fell behind, starts again: restarted at the next frame, decoders
// slower than real time would restart it at every frame, breaking the sound
// into pieces.
constexpr double audio_resume_seconds = 1.0;
// How long a video file found on disk counts as there (can_play is asked at
// every frame played, which would ask the disk as often).
constexpr double found_for_seconds = 2.0;

const QColor background("#171717");

double seconds_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Whether `path` is a file: found less than found_for_seconds ago, or on disk
// now. A file found missing is looked for again at once.
bool found_on_disk(const std::filesystem::path& path) {
    static std::mutex mutex;
    static std::map<std::filesystem::path, double> found;
    const double now = seconds_now();
    {
        const std::lock_guard lock(mutex);
        if (const auto known = found.find(path); known != found.end() && now - known->second < found_for_seconds) {
            return true;
        }
    }
    std::error_code error;
    const bool there = std::filesystem::is_regular_file(path, error);
    const std::lock_guard lock(mutex);
    if (!there) {
        found.erase(path);
        return false;
    }
    if (found.size() > 256) {
        found.clear();
    }
    found[path] = now;
    return true;
}

std::string crop_text(const std::optional<CropBox>& crop) {
    return crop ? std::to_string(crop->w) + "," + std::to_string(crop->h) + "," + std::to_string(crop->x) + "," +
                      std::to_string(crop->y)
                : "none";
}

}  // namespace

// --------------------------------------------------------------- the surfaces

StreamSurface::StreamSurface(VideoCompareView* view) : QWidget(view), view_(view) {
    setFocusPolicy(Qt::StrongFocus);
    // It paints all of itself: nothing behind it is painted first.
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void StreamSurface::set_frame(FramePixels pixels, Size size) {
    // The view repaints only the surface on top (the other is covered), and
    // the image is made when it is painted, so the covered one makes none.
    pixels_ = std::move(pixels);
    size_ = size;
    image_.reset();
}

void StreamSurface::clear_frame() {
    pixels_.reset();
    size_ = {0, 0};
    image_ = QImage();
    update();
}

void StreamSurface::update_frame() {
    if (drawn_ && drawn_->first == drawn_key()) {
        update(drawn_->second);
    } else {
        update();
    }
}

StreamSurface::DrawnKey StreamSurface::drawn_key() const {
    return {width(), height(), devicePixelRatioF(), size_.first, size_.second};
}

QImage& StreamSurface::frame_image() {
    if (!image_) {
        // RGBX, not RGBA: FFmpeg's alpha is always opaque, and a frame that
        // may be translucent is blended into the window, not copied.
        image_ = pixels_ ? QImage(pixels_->data(), size_.first, size_.second, size_.first * 4, QImage::Format_RGBX8888)
                         : QImage();
    }
    return *image_;
}

void StreamSurface::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    QImage& image = frame_image();
    drawn_.reset();
    if (image.isNull() || !view_->fitted()) {
        painter.fillRect(rect(), background);
        if (!image.isNull()) {
            if (const std::optional<Placement> place = view_->zoom_placement(size_)) {
                image.setDevicePixelRatio(1.0);
                painter.drawImage(QRectF(place->x, place->y, place->width, place->height), image);
            }
        }
        return;
    }
    // Fitted, on whole device pixels. A frame decoded at the view's size is
    // copied as it is, not scaled by the pixel or two its even size leaves,
    // and only the bars beside it are filled. Each bar reaches a pixel into
    // the frame, which then covers it, so that rounding leaves no gap.
    const double ratio = devicePixelRatioF();
    const auto view_w = static_cast<int>(std::nearbyint(width() * ratio));
    const auto view_h = static_cast<int>(std::nearbyint(height() * ratio));
    const double scale = std::min(static_cast<double>(view_w) / size_.first, static_cast<double>(view_h) / size_.second);
    auto w = static_cast<int>(std::nearbyint(size_.first * scale));
    auto h = static_cast<int>(std::nearbyint(size_.second * scale));
    if (w - size_.first >= 0 && w - size_.first <= 2 && h - size_.second >= 0 && h - size_.second <= 2) {
        w = size_.first;
        h = size_.second;
    }
    const int x = (view_w - w) / 2;
    const int y = (view_h - h) / 2;
    drawn_ = std::pair(drawn_key(), QRectF(x / ratio, y / ratio, w / ratio, h / ratio).toAlignedRect());
    if (!drawn_->second.contains(event->rect())) {  // not just the frame (update_frame)
        std::vector<QRect> bars;
        if (x > 0) {
            bars.emplace_back(0, 0, x + 1, view_h);
        }
        if (x + w < view_w) {
            bars.emplace_back(x + w - 1, 0, view_w - x - w + 1, view_h);
        }
        if (y > 0) {
            bars.emplace_back(0, 0, view_w, y + 1);
        }
        if (y + h < view_h) {
            bars.emplace_back(0, y + h - 1, view_w, view_h - y - h + 1);
        }
        for (const QRect& bar : bars) {
            painter.fillRect(QRectF(bar.x() / ratio, bar.y() / ratio, bar.width() / ratio, bar.height() / ratio),
                             background);
        }
    }
    if (w == size_.first && h == size_.second) {
        image.setDevicePixelRatio(ratio);
        painter.drawImage(QPointF(x / ratio, y / ratio), image);
    } else {  // decoded for another size, until the view's settles
        image.setDevicePixelRatio(1.0);
        painter.drawImage(QRectF(x / ratio, y / ratio, w / ratio, h / ratio), image);
    }
}

bool StreamSurface::can_drag() const {
    const std::optional<VideoCompareView::ZoomContext> context = view_->zoom_context();
    return context && view_->zoom().factor && view_->zoom().can_drag(context->view, context->frame, context->ratio);
}

void StreamSurface::refresh_cursor() {
    // An open hand where the frame can be dragged; else the usual one.
    if (!drag_from_) {
        if (can_drag()) {
            setCursor(Qt::OpenHandCursor);
        } else {
            unsetCursor();
        }
    }
}

void StreamSurface::mousePressEvent(QMouseEvent* event) {
    setFocus(Qt::MouseFocusReason);
    if (event->button() == Qt::LeftButton && can_drag()) {
        drag_from_ = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void StreamSurface::mouseMoveEvent(QMouseEvent* event) {
    if (!drag_from_) {
        QWidget::mouseMoveEvent(event);
        return;
    }
    const QPointF moved = event->position() - *drag_from_;
    drag_from_ = event->position();
    if (const std::optional<VideoCompareView::ZoomContext> context = view_->zoom_context();
        context && view_->zoom().drag(moved.x(), moved.y(), context->view, context->frame, context->ratio)) {
        view_->zoom_changed();
    }
    event->accept();
}

void StreamSurface::mouseReleaseEvent(QMouseEvent* event) {
    if (drag_from_ && event->button() == Qt::LeftButton) {
        drag_from_.reset();
        refresh_cursor();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

// ------------------------------------------------------------------- the view

VideoCompareView::VideoCompareView(std::function<std::string()> ffmpeg, Zoom& zoom, QWidget* parent)
    : QWidget(parent), ffmpeg_(std::move(ffmpeg)), zoom_(zoom), decoded_videos_(default_compare_decoded_videos) {
    setFocusPolicy(Qt::StrongFocus);
    setAutoFillBackground(true);
    QPalette palette = this->palette();
    palette.setColor(QPalette::Window, background);
    setPalette(palette);
    make_decoder = [this](const FrameComparison& comparison, FrameSide side, long long start_frame,
                          const PreviewColorSettings& colors, std::optional<Size> maximum, QObject* owner) {
        return new StreamDecodeWorker(ffmpeg_(), comparison, side, start_frame, colors, std::nullopt, maximum, owner);
    };
    clock = seconds_now;
#ifdef VML_NATIVE_PLAYBACK
    native_parts = [](const PreviewColorSettings& colors, QString& why) -> std::optional<NativePool::Parts> {
        if (const std::optional<QString> problem = NativePool::unavailable()) {
            why = *problem;
            return std::nullopt;
        }
        try {
            return NativePool::default_parts(colors);
        } catch (const std::exception& error) {
            why = QString::fromUtf8(error.what());
            return std::nullopt;
        }
    };
#endif
    // Once a resize has settled, FFmpeg's frames are decoded at the new size.
    resize_timer_ = new QTimer(this);
    resize_timer_->setSingleShot(true);
    resize_timer_->setInterval(300);
    connect(resize_timer_, &QTimer::timeout, this, [this] { follow_decode_size(); });
    // Precise: while playing it ticks every half frame.
    pool_timer_ = new QTimer(this);
    pool_timer_->setTimerType(Qt::PreciseTimer);
    pool_timer_->setInterval(busy_tick_ms_default);
    connect(pool_timer_, &QTimer::timeout, this, [this] { tick(); });
    source_surface_ = new StreamSurface(this);
    distorted_surface_ = new StreamSurface(this);
    source_surface_->hide();
    distorted_surface_->hide();
}

VideoCompareView::~VideoCompareView() {
    stop_decoder();
    stop_audio();
    for (QThread* worker : retired_) {
        worker->wait();
    }
    for (QThread* worker : stopping_) {
        worker->wait();
    }
}

void VideoCompareView::resizeEvent(QResizeEvent* event) {
    for (StreamSurface* surface : {source_surface_, distorted_surface_}) {
        surface->setGeometry(rect());
        surface->refresh_cursor();
    }
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        native_pool_->resize();
    }
#endif
    if (pool_active_) {
        resize_timer_->start();
    }
    QWidget::resizeEvent(event);
}

bool VideoCompareView::event(QEvent* event) {
    // Moved to a screen of another scaling: other device pixels, as a resize.
    if (event->type() == QEvent::DevicePixelRatioChange && pool_active_) {
        resize_timer_->start();
    }
    return QWidget::event(event);
}

void VideoCompareView::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    on_screen_ = true;
    if (pool_active_ || native_active()) {
        wake();
    }
}

void VideoCompareView::hideEvent(QHideEvent* event) {
    // Off screen nothing is drawn, and the decoders wait with full queues:
    // ticking on would cost 2-4% of a core for the rest of the session.
    on_screen_ = false;
    pool_timer_->stop();
    QWidget::hideEvent(event);
}

void VideoCompareView::wake() {
    // Ticks often from now, as something is due; only on screen. Not
    // restarted when already ticking that fast: a slider dragged seeks more
    // often than it ticks, and each restart would put the next tick off.
    const int busy = busy_tick_ms();
    if (on_screen_ && !(pool_timer_->isActive() && pool_timer_->interval() == busy)) {
        pool_timer_->start(busy);
    }
}

void VideoCompareView::pace(bool settled) {
    const int interval = settled ? settled_tick_ms : busy_tick_ms();
    if (pool_timer_->interval() != interval) {
        pool_timer_->setInterval(interval);
    }
}

int VideoCompareView::busy_tick_ms() const {
    // Half a frame's time while playing (4 ms at 120 fps), 10 ms otherwise:
    // at 10 ms, a 120 fps video showed only 83 frames a second.
    const double fps = wanted_playing_ && !series_.empty() ? series_.front().fps : 0;
    return fps > 0 ? std::max(2, std::min(busy_tick_ms_default, static_cast<int>(500 / fps))) : busy_tick_ms_default;
}

// ------------------------------------------------------------------- the zoom

std::optional<VideoCompareView::ZoomContext> VideoCompareView::zoom_context(std::optional<Size> frame) const {
    if (!comparison_) {
        return std::nullopt;
    }
    const Size reference = comparison_dimensions(*comparison_);
    if (reference.first <= 0 || reference.second <= 0) {
        return std::nullopt;
    }
    const Size size = frame.value_or(reference);
    return ZoomContext{{width(), height()},
                       {size.first, size.second},
                       devicePixelRatioF() * size.first / reference.first};
}

std::optional<Placement> VideoCompareView::zoom_placement(Size frame) const {
    const std::optional<ZoomContext> context = zoom_context(frame);
    return context ? std::optional(zoom_.placement(context->view, context->frame, context->ratio)) : std::nullopt;
}

void VideoCompareView::zoom_changed() {
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        try {
            native_pool_->place();
        } catch (const std::exception& error) {
            native_failed(error);
        }
        return;
    }
#endif
    // When zoomed, FFmpeg's frames are decoded at the comparison's own size,
    // not the view's.
    follow_decode_size();
    for (StreamSurface* surface : {source_surface_, distorted_surface_}) {
        surface->refresh_cursor();
        surface->update();
    }
}

std::optional<Size> VideoCompareView::wanted_maximum() const {
    // When fitted, the view's size: no more of a frame is ever shown, and
    // FFmpeg scales better than painting does. Zoomed, the comparison's own.
    // Decoded at the screen's size, 4K HDR played at 22-26 frames a second.
    if (zoom_.factor) {
        return std::nullopt;
    }
    const double ratio = devicePixelRatioF();
    return Size{std::max(2, static_cast<int>(std::nearbyint(width() * ratio))),
                std::max(2, static_cast<int>(std::nearbyint(height() * ratio)))};
}

void VideoCompareView::follow_decode_size() {
    if (!pool_active_) {
        return;
    }
    const std::optional<Size> wanted = wanted_maximum();
    if (wanted == pool_maximum_) {
        return;
    }
    if (decode_size_changes(wanted)) {
        // The frame on screen stays until the new ones come (1.4-2 s on an
        // RTX 5090); cleared, the view would be dark that long.
        stop_decoder(true);
        stop_audio();
        start_ffmpeg(wanted_playing_, true);
    } else {
        pool_maximum_ = wanted;  // the same frames: they go on decoding
    }
}

bool VideoCompareView::decode_size_changes(std::optional<Size> maximum) const {
    // Whether any decoded video (the source, the test video shown, its
    // neighbours) would get another frame size.
    if (!comparison_ || maximum == pool_maximum_) {
        return false;
    }
    std::vector<FrameComparison> decoded;
    for (const auto& [key, wanted] : desired_) {
        decoded.push_back(wanted.comparison);
    }
    if (decoded.empty()) {
        decoded = {*comparison_, source_recipe()};
    }
    return std::any_of(decoded.begin(), decoded.end(), [&](const FrameComparison& comparison) {
        return playback_dimensions(comparison, maximum) != playback_dimensions(comparison, pool_maximum_);
    });
}

// --------------------------------------------------------------- playing

std::pair<bool, std::string> VideoCompareView::can_play(const FrameComparison& comparison) {
    if (comparison.resample_target) {
        return {false,
                "Video playback is unavailable for synthetic resolution tests; their processed frames are displayed "
                "automatically."};
    }
    for (const FrameSide side : {FrameSide::source, FrameSide::distorted}) {
        const std::filesystem::path& path = frame_input_path(comparison, side);
        if (!found_on_disk(path)) {
            return {false, "Video file is missing: " + platform::path_text(path)};
        }
    }
    if (const std::optional<std::string> problem =
            untimed_pair_problem(comparison.source_info, comparison.distorted_info)) {
        return {false, *problem};
    }
    return {true, ""};
}

bool VideoCompareView::crops_change_nothing(const FrameComparison& pending, const FrameComparison& cropped) {
    const Size size = comparison_dimensions(pending);
    if (size != comparison_dimensions(cropped)) {
        return false;
    }
    for (const auto& [info, crop] : {std::pair(&cropped.source_info, &cropped.source_crop),
                                     std::pair(&cropped.distorted_info, &cropped.distorted_crop)}) {
        if (*crop && !(*crop)->is_noop(info->width, info->height)) {
            return false;
        }
        if (would_distort(pending, *info, std::nullopt, size.first, size.second)) {
            return false;
        }
    }
    return true;
}

int VideoCompareView::position() const {
    if (!comparison_ || comparison_->fps <= 0) {
        return 0;
    }
    return static_cast<int>(std::nearbyint(static_cast<double>(frame_) / comparison_->fps * 1000));
}

std::vector<QThread*> VideoCompareView::live_workers() const {
    std::vector<QThread*> live;
    for (const std::set<QThread*>* stopping : {&retired_, &stopping_}) {
        for (QThread* worker : *stopping) {
            if (worker->isRunning()) {
                live.push_back(worker);
            }
        }
    }
    for (const auto& [key, worker] : pool_) {
        if (worker->isRunning() && std::find(live.begin(), live.end(), worker) == live.end()) {
            live.push_back(worker);
        }
    }
    return live;
}

bool VideoCompareView::load(const FrameComparison& comparison, int position_ms, bool playing,
                            const PreviewColorSettings& colors, const std::vector<FrameComparison>& series,
                            bool source_native) {
    if (const auto [available, reason] = can_play(comparison); !available) {
        clear();
        emit status_changed(QString::fromStdString(reason));
        return false;
    }
    std::vector<FrameComparison> played = series.empty() ? std::vector<FrameComparison>{comparison} : series;
    int selected = 0;
    if (const auto found = std::find(played.begin(), played.end(), comparison); found != played.end()) {
        selected = static_cast<int>(found - played.begin());
    } else {
        played = {comparison};
    }
    // The pairs playing, each as it is or with black bars found since that
    // change nothing: then they play as they are. Loaded afresh instead,
    // every decoder would start again (seconds at 4K), and the first switch
    // to another test video after a pair's bars were found took that long.
    const auto plays_as_is = [this, &played] {
        if (series_.size() != played.size()) {
            return false;
        }
        for (std::size_t index = 0; index < played.size(); ++index) {
            const FrameComparison& playing = series_[index];
            if (!(playing == played[index]) &&
                !(playing.auto_crop_pending && !played[index].auto_crop_pending &&
                  crops_change_nothing(playing, played[index]))) {
                return false;
            }
        }
        return true;
    };
    if (colors_ == colors && comparison_ && source_native_ == source_native && plays_as_is()) {
        // Another test video of the same series: the decoders go on.
        const int timestamp = position();
        const bool changed = selected_ != selected;
        selected_ = selected;
        comparison_ = series_[static_cast<std::size_t>(selected)];  // what plays, as it plays it
        frame_ = static_cast<long long>(std::nearbyint(timestamp / 1000.0 * comparison.fps));
#ifdef VML_NATIVE_PLAYBACK
        if (native_pool_) {
            if (changed) {
                try {
                    native_pool_->sync(selected);
                } catch (const std::exception& error) {
                    native_failed(error);
                    return true;
                }
            }
            set_playing(playing);
            return true;
        }
#endif
        if (pool_active_) {
            if (changed) {
                distorted_surface_->clear_frame();
                sync_pool();
                presented_ = -1;
                audio_resume_at_ = 0.0;  // a start: the soundtrack with the first frame
                tick();
                wake();
            }
            if (playing != wanted_playing_) {
                set_playing(playing);
            }
            return true;
        }
        if (changed) {
            restart_decoder(playing);
        }
        return true;
    }
    series_ = played;
    source_native_ = source_native;
    selected_ = selected;
    comparison_ = comparison;
    colors_ = colors;
    frame_ = static_cast<long long>(std::nearbyint(position_ms / 1000.0 * comparison.fps));
    wanted_playing_ = playing;
    restart_decoder(playing);
    return true;
}

void VideoCompareView::clear() {
    wanted_playing_ = false;
    is_playing_ = false;
    comparison_.reset();
    stop_decoder();
    stop_audio();
    emit playing_changed(false);
    series_.clear();
}

void VideoCompareView::set_position(int position_ms) {
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        native_pool_->seek(position_ms);
        frame_ = native_pool_->frame();
        wake();
        return;
    }
#endif
    if (!comparison_ || comparison_->fps <= 0) {
        return;
    }
    frame_ = std::max(0LL, static_cast<long long>(std::nearbyint(position_ms / 1000.0 * comparison_->fps)));
    restart_decoder(wanted_playing_);
}

void VideoCompareView::set_playing(bool playing) {
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        native_pool_->set_playing(playing);
        wanted_playing_ = is_playing_ = playing;
        wake();
        emit playing_changed(playing);
        return;
    }
#endif
    if (!pool_active_) {
        // Nothing decoding: playing starts the decoders.
        wanted_playing_ = playing;
        if (playing) {
            if (!is_playing_ && comparison_) {
                restart_decoder(true);
            }
        } else {
            is_playing_ = false;
            emit playing_changed(false);
        }
        return;
    }
    clock_frame_ = !playing && presented_ >= 0 ? presented_ : target_frame();
    wanted_playing_ = is_playing_ = playing;
    // Said at once: said only as a frame is shown, it would go on saying
    // "Playing" when paused (by minimizing the window, say).
    pool_status();
    audio_due_ = false;
    audio_resume_at_ = 0.0;
    clock_started_ = playing && !buffering_ ? std::optional(clock()) : std::nullopt;
    if (!playing) {
        stop_audio();
    } else if (!buffering_) {
        start_audio(frame_);
    }
    wake();
    emit playing_changed(playing);
}

void VideoCompareView::show_source(bool showing) {
    showing_source_ = showing;
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        native_pool_->show_source(showing);
    }
#endif
    if (pool_active_) {
        StreamSurface* surface = showing ? source_surface_ : distorted_surface_;
        surface->raise();
        surface->update();  // its frame is current, but it was not repainted while covered
    }
}

void VideoCompareView::set_audio_enabled(bool enabled) {
    audio_enabled_ = enabled;
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        native_pool_->set_audio_enabled(enabled);
        return;
    }
#endif
    if (!enabled) {
        stop_audio();
    } else if (is_playing_) {
        start_audio(frame_);
    }
}

void VideoCompareView::set_color_settings(const PreviewColorSettings& colors) {
    if (colors == colors_) {
        return;
    }
    colors_ = colors;
    if (comparison_) {
        restart_decoder(wanted_playing_);
    }
}

void VideoCompareView::set_decoded_videos(int count) {
    count = std::max(1, count);
    if (count == decoded_videos_) {
        return;
    }
    decoded_videos_ = count;
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        try {
            native_pool_->sync(selected_);
        } catch (const std::exception& error) {
            native_failed(error);
        }
        return;
    }
#endif
    if (pool_active_) {
        sync_pool();
    }
}

void VideoCompareView::restart_decoder(bool realtime) {
    // Starts playback again from the frame shown: natively where it can,
    // else with FFmpeg.
    if (!comparison_) {
        return;
    }
    QString reason;
#ifdef VML_NATIVE_PLAYBACK
    std::optional<NativePool::Parts> parts;
    if (const std::optional<std::string> problem = native_playback_problem(*comparison_, colors_)) {
        reason = QString::fromStdString(*problem);
    } else if (!NativePool::loaded()) {
        // The first start: what native playback needs is made off the
        // window's thread, and playback starts once it is.
        wanted_playing_ = realtime;
        if (native_loading_ == nullptr) {
            native_loading_ = new QFutureWatcher<void>(this);
            connect(native_loading_, &QFutureWatcher<void>::finished, this, [this] {
                native_loading_->deleteLater();
                native_loading_ = nullptr;
                if (comparison_) {
                    restart_decoder(wanted_playing_);
                }
            });
            native_loading_->setFuture(NativePool::load());
        }
        return;
    } else {
        parts = native_parts(colors_, reason);
    }
    if (parts) {
        stop_decoder();
        stop_audio();
        try {
            native_pool_ = std::make_unique<NativePool>(*this, series_, selected_, position(), colors_, realtime,
                                                        source_native_, std::move(*parts));
            is_playing_ = wanted_playing_ = realtime;
            native_pool_->show_source(showing_source_);
            wake();
            emit playing_changed(realtime);
            return;
        } catch (const std::exception& error) {
            native_pool_.reset();
            reason = QString::fromUtf8(error.what());
        }
    }
#endif
    stop_decoder();
    stop_audio();
    start_ffmpeg(realtime, false, reason);
}

void VideoCompareView::native_failed([[maybe_unused]] const std::exception& error) {
#ifdef VML_NATIVE_PLAYBACK
    native_pool_.reset();
    start_ffmpeg(wanted_playing_, false, QString::fromUtf8(error.what()));
#endif
}

void VideoCompareView::start_ffmpeg(bool realtime, bool keep_frame, const QString& reason) {
    pool_reason_ = reason;
    pool_maximum_ = wanted_maximum();
    pool_active_ = true;
    wanted_playing_ = realtime;
    is_playing_ = realtime;
    clock_frame_ = static_cast<long long>(std::nearbyint(position() / 1000.0 * series_.front().fps));
    clock_started_.reset();
    buffering_ = true;
    audio_due_ = false;
    audio_resume_at_ = 0.0;
    presented_ = -1;
    for (StreamSurface* surface : {source_surface_, distorted_surface_}) {
        if (!keep_frame) {
            surface->clear_frame();
        }
        surface->setGeometry(rect());
        surface->show();
    }
    show_source(showing_source_);
    const QString separator = QString::fromUtf8(" \xC2\xB7 ");
    status(QStringLiteral("Preparing source/current/adjacent videos") + separator +
           QStringLiteral("GPU tone mapping and RGB") +
           (reason.isEmpty() ? QString() : separator + QStringLiteral("native fallback: %1").arg(reason)));
    sync_pool();
    wake();
    emit playing_changed(realtime);
}

void VideoCompareView::status(const QString& text) {
    if (text != last_status_) {
        last_status_ = text;
        emit status_changed(text);
    }
}

FrameComparison VideoCompareView::source_recipe() const {
    FrameComparison recipe = source_playback_comparison(*comparison_, source_native_);
    recipe.fps = series_.front().fps;
    return recipe;
}

std::string VideoCompareView::source_key() const {
    for (const auto& [key, wanted] : desired_) {
        if (key.starts_with("s:")) {
            return key;
        }
    }
    return {};
}

void VideoCompareView::sync_pool() {
    // Keeps the decoders still wanted, stops the others, starts the missing.
    if (!pool_active_) {
        return;
    }
    const FrameComparison source = source_recipe();
    const Size size = playback_dimensions(source, pool_maximum_);
    desired_.clear();
    desired_.push_back({"s:" + platform::path_text(source.source_info.path) + "|" + crop_text(source.source_crop) +
                            "|" + std::to_string(size.first) + "x" + std::to_string(size.second),
                        {source, FrameSide::source}});
    for (const int index : neighbour_indices(static_cast<int>(series_.size()), selected_, decoded_videos_)) {
        FrameComparison comparison = series_[static_cast<std::size_t>(index)];
        comparison.fps = series_.front().fps;
        desired_.push_back({"d:" + std::to_string(index), {comparison, FrameSide::distorted}});
    }
    for (auto worker = pool_.begin(); worker != pool_.end();) {
        const bool wanted = std::any_of(desired_.begin(), desired_.end(),
                                        [&worker](const auto& item) { return item.first == worker->first; });
        if (wanted) {
            ++worker;
            continue;
        }
        worker->second->cancel();
        // One that has finished (at the end of a short clip, or failed) goes
        // now: kept, it held its queued frames until the window closed.
        retire(worker->second);
        history_.erase(worker->first);
        details_.erase(worker->first);
        failures_.erase(worker->first);
        worker = pool_.erase(worker);
    }
    launch_missing();
}

void VideoCompareView::launch_missing() {
    if (!pool_active_) {
        return;
    }
    // Decoders still stopping count too, so that quick switching never runs
    // one more than allowed while an old process exits.
    int occupied = static_cast<int>(pool_.size());
    for (QThread* worker : retired_) {
        occupied += worker->isRunning() ? 1 : 0;
    }
    // The pair on screen first: the test videos beside it start once it has
    // shown a frame. Started with it, two of them slowed its first frame at
    // the tab's first opening from 1.1-1.3 s to 1.5-1.6 s.
    const std::string shown_test = "d:" + std::to_string(selected_);
    bool launched = false;
    for (const auto& [key, wanted] : desired_) {
        if (pool_.contains(key) || occupied >= decoder_limit()) {
            continue;
        }
        if (presented_ < 0 && !key.starts_with("s:") && key != shown_test) {
            continue;
        }
        PlaybackDecoder* worker =
            make_decoder(wanted.comparison, wanted.side, target_frame(), colors_, pool_maximum_, this);
        connect(worker, &PlaybackDecoder::ready, this,
                [this, key, worker](const QString& detail) { stream_ready(key, worker, detail); });
        connect(worker, &PlaybackDecoder::failed, this,
                [this, key, worker](const QString& error) { stream_failed(key, worker, error); });
        connect(worker, &QThread::finished, this, [this, key, worker] { stream_finished(key, worker); });
        pool_[key] = worker;
        history_[key] = {};
        worker->start();
        ++occupied;
        launched = true;
    }
    if (launched && presented_ >= 0) {
        pool_status();  // its count of streams
    }
}

void VideoCompareView::stream_ready(const std::string& key, PlaybackDecoder* worker, const QString& detail) {
    if (const auto found = pool_.find(key); found != pool_.end() && found->second == worker) {
        details_[key] = detail;
    }
}

void VideoCompareView::stream_failed(const std::string& key, PlaybackDecoder* worker, const QString& error) {
    const auto found = pool_.find(key);
    if (found == pool_.end() || found->second != worker) {
        return;
    }
    failures_[key] = error;
    // The source is always part of the pair shown; a test video is named by
    // its number in the list.
    emit status_changed(key.starts_with("s:")
                            ? QStringLiteral("Selected video failed: %1").arg(error)
                            : QStringLiteral("Video %1 could not play: %2").arg(std::stoi(key.substr(2)) + 1).arg(error));
}

void VideoCompareView::stream_finished(const std::string& key, PlaybackDecoder* worker) {
    retired_.erase(worker);
    // A worker at the end of a short clip keeps its frames until they are
    // shown, and goes when it is retired; one no longer wanted can go now.
    if (const auto found = pool_.find(key); found == pool_.end() || found->second != worker) {
        worker->deleteLater();
    } else {
        ended_.insert(worker);
    }
    launch_missing();
}

void VideoCompareView::retire(PlaybackDecoder* worker) {
    if (ended_.erase(worker) > 0) {
        worker->deleteLater();  // its finished has come and gone
    } else {
        retired_.insert(worker);  // stream_finished deletes it when its finished comes
    }
}

long long VideoCompareView::target_frame() const {
    // The frame due now by the playback clock.
    if (!clock_started_ || !wanted_playing_) {
        return clock_frame_;
    }
    return clock_frame_ + static_cast<long long>((clock() - *clock_started_) * series_.front().fps);
}

void VideoCompareView::tick() {
    // Advances playback: natively, or by taking each worker's frames due and
    // showing the newest pair both videos have.
#ifdef VML_NATIVE_PLAYBACK
    if (native_pool_) {
        native_tick();
        return;
    }
#endif
    if (!pool_active_) {
        return;
    }
    launch_missing();
    const long long target = target_frame();
    const std::string source = source_key();
    const std::string distorted = "d:" + std::to_string(selected_);
    // A faster video's frames are not dropped while the other is still
    // decoding them: its queue holds it at three frames ahead.
    long long paired_target = target;
    for (const std::string& key : {source, distorted}) {
        std::optional<long long> available;
        if (const auto worker = pool_.find(key); worker != pool_.end()) {
            available = worker->second->latest_frame_number();
        }
        if (!available) {
            // Nothing queued: the newest frame already taken, else the clock's.
            const auto history = history_.find(key);
            available = history == history_.end() || history->second.empty() ? clock_frame_
                                                                              : history->second.rbegin()->first;
        }
        paired_target = std::min(paired_target, *available);
    }
    for (auto& [key, worker] : pool_) {
        std::map<long long, FramePixels>& history = history_[key];
        const long long due = key == source || key == distorted ? paired_target : target;
        for (PlaybackFrame& frame : worker->drain_through(due)) {
            history[frame.number] = std::move(frame.pixels);
        }
        // Three past frames, with the worker's three ahead, bound the memory
        // used, whatever the video's length or the number of files.
        while (history.size() > 3) {
            FramePixels pixels = std::move(history.begin()->second);
            history.erase(history.begin());
            if (!source_surface_->shows(pixels) && !distorted_surface_->shows(pixels)) {
                worker->recycle(std::move(pixels));
            }
        }
    }
    const std::map<long long, FramePixels>& source_frames = history_[source];
    const std::map<long long, FramePixels>& distorted_frames = history_[distorted];
    std::optional<long long> common;
    for (const auto& [number, pixels] : source_frames) {
        if (distorted_frames.contains(number)) {
            common = number;  // the newest, the map being in order
        }
    }
    if (!common) {
        buffering_ = true;
        if (wanted_playing_ && target - std::max(0LL, presented_) > 2) {
            // Starting (a seek, another test video) is not falling behind:
            // the soundtrack starts with the first frame.
            if (presented_ >= 0 && clock_started_) {
                fell_behind();
            } else {
                stop_audio();
            }
        }
        const auto failure = failures_.contains(source) ? failures_.find(source) : failures_.find(distorted);
        status(failure != failures_.end()
                   ? QStringLiteral("Selected video failed: %1").arg(failure->second)
                   : QStringLiteral("Buffering selected comparison; adjacent videos are preparing in the background"));
        pace(failure != failures_.end());
        return;
    }
    const long long frame = *common;
    if (frame == presented_) {
        if (wanted_playing_ && target - frame > 2) {
            // The sound must not run far ahead while frames fail to come;
            // it starts again with them, once they keep up.
            fell_behind();
            buffering_ = true;
        }
        pace(!wanted_playing_ && frame >= target);
        return;
    }
    presented_ = frame;
    const double fps = series_.front().fps;
    if (wanted_playing_ && target - frame > 2) {
        // A decoder slower than real time must not build a queue without
        // end, or let the pictures drift away from the sound.
        clock_frame_ = frame;
        clock_started_ = clock();
        fell_behind();
        buffering_ = true;
    }
    frame_ = static_cast<long long>(std::nearbyint(frame / fps * comparison_->fps));
    const auto source_wanted = std::find_if(desired_.begin(), desired_.end(),
                                            [&source](const auto& item) { return item.first == source; });
    source_surface_->set_frame(source_frames.at(frame), playback_dimensions(source_wanted->second.comparison, pool_maximum_));
    distorted_surface_->set_frame(distorted_frames.at(frame), playback_dimensions(*comparison_, pool_maximum_));
    (showing_source_ ? source_surface_ : distorted_surface_)->update_frame();
    if (!clock_started_ && wanted_playing_) {
        clock_frame_ = frame;
        clock_started_ = clock();
    }
    if (buffering_ && wanted_playing_) {
        audio_due_ = true;
    }
    buffering_ = false;
    if (audio_due_ && wanted_playing_ && clock() >= audio_resume_at_) {
        audio_due_ = false;
        start_audio(frame_);
    }
    emit position_changed(static_cast<int>(std::nearbyint(frame / fps * 1000)));
    pool_status();
    check_end();
    pace(false);  // the next tick starts the test videos decoded beside it
}

bool VideoCompareView::native_active() const {
#ifdef VML_NATIVE_PLAYBACK
    return native_pool_ != nullptr;
#else
    return false;
#endif
}

void VideoCompareView::native_tick() {
#ifdef VML_NATIVE_PLAYBACK
    NativePool& pool = *native_pool_;
    try {
        const int position = pool.poll();
        const auto frame = static_cast<long long>(std::nearbyint(position / 1000.0 * comparison_->fps));
        if (frame != frame_) {
            frame_ = frame;
            emit position_changed(position);
        }
        if (pool.ended() && wanted_playing_) {
            set_playing(false);
        }
        const QString separator = QString::fromUtf8(" \xC2\xB7 ");
        status((wanted_playing_ ? QStringLiteral("Playing") : QStringLiteral("Paused")) + separator +
               QStringLiteral("FFmpeg D3D11") + separator +
               QStringLiteral("%1/%2 streams").arg(pool.decoders()).arg(decoder_limit()) + separator +
               pool.description());
        check_end();
        if (wanted_playing_ && !pool.buffering() && !pool.ended()) {
            // The next tick when the next frame is due, if ticking at all:
            // it is never started here, while off screen (hideEvent).
            if (pool_timer_->isActive()) {
                const std::optional<int> due = pool.next_tick_ms();
                pool_timer_->start(due ? *due : busy_tick_ms());
            }
        } else {
            pace(!wanted_playing_ && pool.paired_on_selected());
        }
    } catch (const std::exception& error) {
        native_failed(error);
    }
#endif
}

#ifdef VML_NATIVE_PLAYBACK
NativeOutput::Placement VideoCompareView::native_placement(const NativeFrame& frame,
                                                                 const FrameComparison& comparison,
                                                                 FrameSide side) const {
    // The frame as the Python app's GStreamer gave it to its presenter: its
    // crop stretched to the comparison's size (`size`), or boxed in black
    // keeping its own shape while black bars are still to be detected.
    const Size size = comparison_dimensions(comparison);
    // A frame already cropped and scaled (the source scaled down) is drawn whole.
    const std::optional<CropBox> crop =
        frame.prepared ? std::nullopt : side == FrameSide::source ? comparison.source_crop : comparison.distorted_crop;
    const int crop_x = crop ? crop->x : 0;
    const int crop_y = crop ? crop->y : 0;
    const int crop_w = crop ? crop->w : frame.width;
    const int crop_h = crop ? crop->h : frame.height;
    const double ratio = devicePixelRatioF();
    NativeOutput::Placement placement;
    if (!zoom_.factor) {
        // Fitted on whole pixels, as the presenter fits a frame (in floats).
        const auto window_w = static_cast<float>(std::max(1L, std::lround(width() * ratio)));
        const auto window_h = static_cast<float>(std::max(1L, std::lround(height() * ratio)));
        const auto frame_w = static_cast<float>(size.first);
        const auto frame_h = static_cast<float>(size.second);
        const float scale = std::min(window_w / frame_w, window_h / frame_h);
        float target_w = std::round(frame_w * scale);
        float target_h = std::round(frame_h * scale);
        float target_x = std::floor((window_w - target_w) / 2);
        float target_y = std::floor((window_h - target_h) / 2);
        if (comparison.auto_crop_pending) {
            const float boxed = std::min(target_w / static_cast<float>(crop_w), target_h / static_cast<float>(crop_h));
            const float inner_w = std::round(static_cast<float>(crop_w) * boxed);
            const float inner_h = std::round(static_cast<float>(crop_h) * boxed);
            target_x += std::floor((target_w - inner_w) / 2);
            target_y += std::floor((target_h - inner_h) / 2);
            target_w = inner_w;
            target_h = inner_h;
        }
        placement.source = {crop_x, crop_y, crop_w, crop_h};
        placement.target = {static_cast<int>(target_x), static_cast<int>(target_y), static_cast<int>(target_w),
                            static_cast<int>(target_h)};
        return placement;
    }
    // Zoomed: the part of the comparison-sized frame in view (the Python
    // app's native_view), then that part in the decoded picture's pixels.
    const std::optional<ZoomContext> context = zoom_context(size);
    if (!context) {
        placement.source = {crop_x, crop_y, crop_w, crop_h};
        placement.target = {0, 0, 1, 1};
        return placement;
    }
    const Zoom::Visible visible = zoom_.visible(context->view, context->frame, context->ratio);
    const auto [left, top, right, bottom] = visible.fractions;
    const auto even = [](double value) { return static_cast<int>(std::nearbyint(value)); };
    placement.target = {even(visible.drawn.x * ratio), even(visible.drawn.y * ratio),
                        std::max(1, even(visible.drawn.width * ratio)), std::max(1, even(visible.drawn.height * ratio))};
    // Sized from the part in view, then placed: rounding each edge on its own
    // can give 4 rows where 5 are in view.
    const int part_w = std::max(1, std::min(size.first, even((right - left) * size.first)));
    const int part_h = std::max(1, std::min(size.second, even((bottom - top) * size.second)));
    const int part_x = std::min(even(left * size.first), size.first - part_w);
    const int part_y = std::min(even(top * size.second), size.second - part_h);
    const double across = static_cast<double>(crop_w) / size.first;
    const double down = static_cast<double>(crop_h) / size.second;
    placement.source = {crop_x + even(part_x * across), crop_y + even(part_y * down), std::max(1, even(part_w * across)),
                        std::max(1, even(part_h * down))};
    return placement;
}
#endif

void VideoCompareView::pool_status() {
    const auto detail = details_.find("d:" + std::to_string(selected_));
    QString said = detail != details_.end() ? detail->second : QStringLiteral("GPU processing starting");
    if (!pool_reason_.isEmpty()) {
        said += QString::fromUtf8(" \xC2\xB7 SDR preview (native playback unavailable)");
    }
    status(QString::fromUtf8("%1 \xC2\xB7 %2/%3 streams \xC2\xB7 %4")
               .arg(wanted_playing_ ? QStringLiteral("Playing") : QStringLiteral("Paused"))
               .arg(pool_.size())
               .arg(decoder_limit())
               .arg(said));
}

void VideoCompareView::check_end() {
    // Stops playing at the end of the shortest video.
    std::optional<double> shortest;
    for (const FrameComparison& item : series_) {
        if (item.frame_count > 0 && item.fps > 0) {
            const double seconds = static_cast<double>(item.frame_count) / item.fps;
            shortest = shortest ? std::min(*shortest, seconds) : seconds;
        }
    }
    if (shortest && wanted_playing_ && position() / 1000.0 >= *shortest - 1 / series_.front().fps) {
        set_playing(false);
        status(QStringLiteral("Playback ended"));
    }
}

void VideoCompareView::start_audio(long long frame) {
    // The source's soundtrack, from the frame on screen: one soundtrack
    // across switches between test videos (several at once would mix).
    if (!audio_enabled_ || !comparison_ || series_.empty()) {
        return;
    }
    const FrameComparison& source = series_.front();
    if (pool_active_) {
        frame = std::max(0LL, static_cast<long long>(std::nearbyint(presented_ / series_.front().fps * source.fps)));
    }
    stop_audio();
    const std::optional<std::filesystem::path> ffplay = ffplay_beside(ffmpeg_());
    if (!ffplay) {
        return;
    }
    const std::optional<std::vector<std::string>> command = build_audio_command(*ffplay, source, frame);
    if (!command) {
        return;
    }
    try {
        audio_.emplace(platform::ChildProcess::start(*command));
    } catch (const platform::ProcessError&) {
        // No soundtrack: the pictures play on.
    }
}

void VideoCompareView::fell_behind() {
    // The soundtrack stops while the frames are behind the clock, and starts
    // again only once they have kept up for a while.
    stop_audio();
    audio_resume_at_ = clock() + audio_resume_seconds;
}

void VideoCompareView::stop_audio() {
    audio_.reset();  // ends ffplay
}

void VideoCompareView::stop_decoder(bool keep_frame) {
    // Stops all decoding, clearing the frame shown unless keep_frame.
#ifdef VML_NATIVE_PLAYBACK
    native_pool_.reset();
#endif
    pool_active_ = false;
    pool_timer_->stop();
    for (const auto& [key, worker] : pool_) {
        worker->cancel();
        retire(worker);
    }
    pool_.clear();
    history_.clear();
    desired_.clear();
    details_.clear();
    failures_.clear();
    if (keep_frame) {
        return;
    }
    source_surface_->clear_frame();
    distorted_surface_->clear_frame();
    source_surface_->hide();
    distorted_surface_->hide();
}

}  // namespace vml::app
