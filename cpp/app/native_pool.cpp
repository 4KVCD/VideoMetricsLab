#include "native_pool.h"

#include <QGuiApplication>
#include <QMouseEvent>
#include <QPainter>
#include <QPointer>
#include <QThread>

#include <QStringList>

#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "native_device.h"
#include "video_compare_view.h"
#include "vml/native_playback.h"
#include "vml/platform/system.h"
#include "vml/video_playback.h"

namespace vml::app {

namespace {

// How long playback may run behind its soundtrack, catching up while the
// sound plays on, before it stops the sound to wait for the decoders. The
// window's thread can be held up now and then (55-60 ms every 15 s on one
// PC), leaving the decoders waiting with full queues: stopping and seeking
// the sound each time would make it break off.
constexpr double catch_up_seconds = 0.2;
// How long the soundtrack may take to be ready to play.
constexpr double audio_preroll_seconds = 15.0;

// Python's round(): halves to even.
long long round_even(double value) { return static_cast<long long>(std::nearbyint(value)); }

// Ends something of native playback's (the presenter, the sound) on a thread
// of its own, so that the window does not wait for its teardown; then
// `finished` on the window's thread.
class Ender : public QThread {
public:
    Ender(std::function<void()> end, QObject* parent) : QThread(parent), end_(std::move(end)) {}

protected:
    void run() override { end_(); }

private:
    std::function<void()> end_;
};

}  // namespace

// ---------------------------------------------------------- the paired frame

PairedFrameWidget::PairedFrameWidget(VideoCompareView* view) : QWidget(view), view_(view) {
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_NativeWindow);
    setAttribute(Qt::WA_DontCreateNativeAncestors);
}

void PairedFrameWidget::paintEvent(QPaintEvent* /*event*/) {
    // Painted dark even while native playback owns it: its presenter's window
    // covers all of it once it shows a frame, but until then (a pool just
    // started) this is what shows. Unpainted, it would show whatever the
    // window last drew there, such as the Videos tab.
    QPainter(this).fillRect(rect(), QColor("#171717"));
}

bool PairedFrameWidget::can_drag() const {
    const std::optional<VideoCompareView::ZoomContext> context = view_->zoom_context();
    return context && view_->zoom().factor && view_->zoom().can_drag(context->view, context->frame, context->ratio);
}

void PairedFrameWidget::refresh_cursor() {
    if (!drag_from_) {
        if (can_drag()) {
            setCursor(Qt::OpenHandCursor);
        } else {
            unsetCursor();
        }
    }
}

void PairedFrameWidget::mousePressEvent(QMouseEvent* event) {
    setFocus(Qt::MouseFocusReason);
    if (event->button() == Qt::LeftButton && can_drag()) {
        drag_from_ = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void PairedFrameWidget::mouseMoveEvent(QMouseEvent* event) {
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

void PairedFrameWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (drag_from_ && event->button() == Qt::LeftButton) {
        drag_from_.reset();
        refresh_cursor();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

// ------------------------------------------------------------------ the pool

namespace {

std::atomic<bool>& native_loaded() {
    static std::atomic<bool> loaded{false};
    return loaded;
}

// The loading started (NativePool::load), until unload.
struct Loading {
    std::mutex mutex;
    std::optional<QFuture<void>> future;
};

Loading& native_loading() {
    static Loading loading;
    return loading;
}

}  // namespace

QFuture<void> NativePool::load() {
    Loading& state = native_loading();
    const std::lock_guard lock(state.mutex);
    std::optional<QFuture<void>>& loading = state.future;
    if (!loading) {
        loading = QtConcurrent::run([] {
            if (!unavailable()) {
                try {
                    (void)NativeDevice::shared();
                    NativePresentation::prepare();
                } catch (const std::exception&) {
                    // Said when playback starts, which makes them again.
                }
            }
            native_loaded() = true;
        });
    }
    return *loading;
}

bool NativePool::unload() {
    {
        Loading& state = native_loading();
        const std::lock_guard lock(state.mutex);
        if (state.future && !state.future->isFinished()) {
            return false;
        }
        state.future.reset();
        native_loaded() = false;
    }
    NativeDevice::release_shared();
    return true;
}

bool NativePool::loaded() {
    return native_loaded() || QGuiApplication::platformName() == QStringLiteral("offscreen");
}

std::optional<QString> NativePool::unavailable() {
    if (QGuiApplication::platformName() == QStringLiteral("offscreen")) {
        // Native windows: none offscreen, which also keeps headless tests from
        // starting the graphics drivers.
        return QStringLiteral("native video output is disabled by the offscreen Qt platform");
    }
    if (const std::optional<std::string> problem = load_ffmpeg_libraries()) {
        return QString::fromStdString(*problem);
    }
    return NativePresentation::unavailable();
}

NativePool::Parts NativePool::default_parts(const PreviewColorSettings& colors) {
    const std::shared_ptr<NativeDevice> device = NativeDevice::shared();
    Parts parts;
    parts.make_video = [device](const FrameComparison& comparison, FrameSide side, const PreviewColorSettings& wanted,
                                int position_ms, QObject* owner) -> NativeVideo* {
        return new FfmpegNativeVideo(device, comparison, side, wanted, position_ms, owner);
    };
    const bool hdr_display = colors.display_hdr_enabled == true;
    parts.make_presentation = [device, hdr_display](PairedFrameWidget& surface) -> std::unique_ptr<NativeOutput> {
        return std::make_unique<NativePresentation>(device, reinterpret_cast<HWND__*>(surface.winId()), hdr_display);
    };
    parts.make_soundtrack = [](const FrameComparison& source, int position_ms,
                               bool enabled) -> std::unique_ptr<Soundtrack> {
        return std::make_unique<FfmpegSoundtrack>(frame_input_path(source, FrameSide::source), position_ms, enabled);
    };
    return parts;
}

NativePool::NativePool(VideoCompareView& view, std::vector<FrameComparison> series, int selected, int position_ms,
                       const PreviewColorSettings& colors, bool playing, bool source_native, Parts parts)
    : view_(view),
      series_(std::move(series)),
      colors_(colors),
      source_native_(source_native),
      parts_(std::move(parts)),
      selected_(selected),
      fps_(series_.empty() ? 0.0 : series_.front().fps),
      position_(position_ms),
      playing_(playing) {
    if (fps_ <= 0 || std::any_of(series_.begin(), series_.end(), [this](const FrameComparison& item) {
            return std::abs(item.distorted_info.fps - fps_) > .01;
        })) {
        throw std::runtime_error("Native frame lock currently requires matching frame rates");
    }
    frame_ = round_even(position_ms * fps_ / 1000);
    const double now = view_.clock();
    anchor_ = now;
    anchor_frame_ = frame_;
    audio_deadline_ = now + audio_preroll_seconds;
    surface_ = new PairedFrameWidget(&view_);
    surface_->setFocusProxy(&view_);
    surface_->setGeometry(view_.rect());
    surface_->show();
    try {
        output_ = parts_.make_presentation(*surface_);
        audio_ = parts_.make_soundtrack(series_.front(), position_ms, view_.audio_enabled_);
        sync(selected);
    } catch (...) {
        stop();
        throw;
    }
}

NativePool::~NativePool() { stop(); }

void NativePool::retire_video(NativeVideo* video) {
    video->cancel();
    view_.retired_.insert(video);
    // isFinished() is true a moment before finished is sent: a video caught
    // in between is deleted below, and finished then finds it gone.
    QObject::connect(video, &QThread::finished, &view_, [view = &view_, video, alive = QPointer<NativeVideo>(video)] {
        view->retired_.erase(video);
        if (alive) {
            alive->deleteLater();
        }
    });
    if (video->isFinished()) {
        view_.retired_.erase(video);
        video->deleteLater();
    }
}

void NativePool::sync(int selected) {
    selected_ = selected;
    source_recipe_ = source_playback_comparison(series_[static_cast<std::size_t>(selected)], source_native_);
    const Size size = comparison_dimensions(source_recipe_);
    const std::optional<CropBox>& crop = source_recipe_.source_crop;
    source_key_ = "s:" + platform::path_text(source_recipe_.source_info.path) + "|" +
                  (crop ? std::to_string(crop->w) + "," + std::to_string(crop->h) + "," + std::to_string(crop->x) + "," +
                              std::to_string(crop->y)
                        : std::string("none")) +
                  "|" + std::to_string(size.first) + "x" + std::to_string(size.second);
    desired_.clear();
    desired_.push_back({source_key_, {source_recipe_, FrameSide::source}});
    for (const int index : neighbour_indices(static_cast<int>(series_.size()), selected, view_.decoded_videos_)) {
        desired_.push_back({distorted_key(index), {series_[static_cast<std::size_t>(index)], FrameSide::distorted}});
    }
    for (auto entry = entries_.begin(); entry != entries_.end();) {
        const bool wanted = std::any_of(desired_.begin(), desired_.end(),
                                        [&entry](const auto& item) { return item.first == entry->first; });
        if (wanted) {
            ++entry;
            continue;
        }
        retire_video(entry->second);
        frames_.erase(entry->first);
        eos_.erase(entry->first);
        entry = entries_.erase(entry);
    }
    // A test video decoded beside the one shown, chosen now, can be far
    // behind it: a decoder slower than playback (4K VVC on the CPU) stops
    // once it falls behind, the few frames it holds kept as ones still to
    // come. Decoded on from there, it would hold playback up for as long as
    // it had been away; sought to the frame shown, it starts as a new
    // decoder does.
    const std::string chosen = distorted_key(selected_);
    if (const auto entry = entries_.find(chosen); entry != entries_.end() && !entry->second->seeking()) {
        std::map<long long, NativeFramePtr>& queue = frames_[chosen];
        if (!queue.empty() && queue.rbegin()->first < frame_ - static_cast<long long>(fps_ / 2)) {
            queue.clear();
            eos_.erase(chosen);
            // One frame early, as seek() does.
            entry->second->seek(std::max(0, position_ - static_cast<int>(round_even(1000 / fps_))));
        }
    }
    launch_missing();
    choose_pair(frame_);
}

void NativePool::launch_missing() {
    // Decoders still stopping count too.
    int occupied = static_cast<int>(entries_.size());
    for (QThread* worker : view_.retired_) {
        occupied += worker->isRunning() ? 1 : 0;
    }
    // Until a pair has been shown, only the two videos shown are started.
    // Started with them, the others take the disk and the GPU from them, and
    // the first pair comes later (132 ms against 120-124 ms, for a 4K source
    // and three test videos). They start once it is shown (poll).
    const bool pair_shown = pair_.has_value();
    for (const auto& [key, wanted] : desired_) {
        if (entries_.contains(key) || occupied >= view_.decoder_limit() || closed_) {
            continue;
        }
        if (!pair_shown && key != source_key_ && key != distorted_key(selected_)) {
            continue;
        }
        NativeVideo* player = parts_.make_video(wanted.first, wanted.second, colors_,
                                                std::max(0, position_ - static_cast<int>(round_even(1000 / fps_))),
                                                &view_);
        entries_[key] = player;
        shadings_[key] = player->shading();
        frames_[key];
        player->start();
        ++occupied;
    }
}

bool NativePool::choose_pair(long long target) {
    const std::map<long long, NativeFramePtr>& source = frames_[source_key_];
    const std::map<long long, NativeFramePtr>& distorted = frames_[distorted_key(selected_)];
    std::optional<long long> frame;
    for (const auto& [number, picture] : source) {
        if (number <= target && distorted.contains(number)) {
            frame = number;  // the newest, the map being in order
        }
    }
    if (!frame || *frame < frame_) {
        return false;
    }
    const bool changed = *frame != frame_ || pair_index_ != selected_ || !pair_;
    frame_ = *frame;
    position_ = static_cast<int>(round_even(static_cast<double>(*frame) * 1000 / fps_));
    pair_ = std::pair(source.at(*frame), distorted.at(*frame));
    pair_index_ = selected_;
    if (changed) {
        present(showing_source_);
    }
    return true;
}

void NativePool::present(bool source) {
    const NativeFramePtr& frame = source ? pair_->first : pair_->second;
    const std::string key = source ? source_key_ : distorted_key(*pair_index_);
    const auto shading = shadings_.find(key);
    const FrameComparison& comparison = source ? source_recipe_ : series_[static_cast<std::size_t>(*pair_index_)];
    output_->present(frame, view_.native_placement(*frame, comparison, source ? FrameSide::source : FrameSide::distorted),
                     shading != shadings_.end() ? shading->second : std::nullopt);
}

void NativePool::place() {
    surface_->refresh_cursor();
    if (pair_ && pair_index_ == selected_) {
        present(showing_source_);
    }
}

void NativePool::show_source(bool showing) {
    showing_source_ = showing;
    if (pair_ && pair_index_ == selected_) {
        present(showing_source_);
    }
}

void NativePool::set_audio_enabled(bool enabled) { audio_->set_enabled(enabled); }

void NativePool::seek(int position_ms) {
    // A decoder that already holds the frame wanted is not sought.
    const long long target = std::max(0LL, round_even(position_ms * fps_ / 1000));
    if (target == frame_) {
        return;
    }
    audio_->set_playing(false);
    audio_running_ = false;
    audio_->seek(static_cast<int>(round_even(static_cast<double>(target) * 1000 / fps_)));
    const double now = view_.clock();
    audio_deadline_ = now + audio_preroll_seconds;
    ended_ = false;
    frame_ = target;
    position_ = static_cast<int>(round_even(static_cast<double>(target) * 1000 / fps_));
    anchor_ = now;
    anchor_frame_ = target;
    late_since_.reset();
    pair_.reset();
    pair_index_.reset();
    for (const auto& [key, player] : entries_) {
        std::map<long long, NativeFramePtr>& queue = frames_[key];
        if (!queue.contains(target)) {
            queue.clear();
            eos_.erase(key);
            // One frame early: rounding a fractional frame time to
            // milliseconds must not seek just beyond the frame wanted.
            player->seek(std::max(0, position_ - static_cast<int>(round_even(1000 / fps_))));
        }
    }
    shown_ = choose_pair(target);
    buffering_ = !shown_;
}

void NativePool::set_playing(bool playing) {
    playing_ = playing;
    anchor_ = view_.clock();
    anchor_frame_ = frame_;
    late_since_.reset();
    if (!playing) {
        audio_->set_playing(false);
        audio_running_ = false;
    }
}

int NativePool::poll() {
    if (const std::optional<QString> error = output_->error()) {
        throw std::runtime_error(error->toStdString());
    }
    launch_missing();
    std::optional<double> audio_ms = audio_->poll();
    double polled_at = view_.clock();
    if (!audio_->ready() && !audio_->failed() && view_.clock() > audio_deadline_) {
        audio_->set_failed(QStringLiteral("Audio preroll timed out"));
    }
    long long target = frame_;
    std::optional<double> clock_ms;
    if (playing_ && !buffering_) {
        if (audio_running_ && audio_ms) {
            target = round_even(*audio_ms * fps_ / 1000);
            clock_ms = audio_ms;
        } else {
            target = anchor_frame_ + static_cast<long long>((view_.clock() - anchor_) * fps_);
        }
    }
    const long long reach = shown_reach();
    for (const auto& [key, player] : entries_) {
        take_frames(key, *player, target, reach);
    }
    // The work above can take time with a newly selected decoder, so the
    // soundtrack's clock is read again before the pair is chosen.
    if (playing_ && !buffering_ && audio_running_ && view_.clock() - polled_at > 0.002) {
        audio_ms = audio_->poll();
        if (audio_ms) {
            target = round_even(*audio_ms * fps_ / 1000);
            clock_ms = audio_ms;
            polled_at = view_.clock();
        }
    }
    clock_ms_ = clock_ms;
    clock_at_ = polled_at;
    const bool matched = shown_ = choose_pair(target);
    keep_up(matched, target);
    check_ended();
    return position_;
}

// The newest frame both videos shown have, which limits the queues' floor
// (take_frames).
long long NativePool::shown_reach() const {
    long long reach = std::numeric_limits<long long>::max();
    for (const std::string& key : {source_key_, distorted_key(selected_)}) {
        const auto queue = frames_.find(key);
        reach = std::min(reach, queue == frames_.end() || queue->second.empty() ? frame_ : queue->second.rbegin()->first);
    }
    return reach;
}

// Takes `player`'s news (an error, its status, its end) and its frames, as
// far as its queue (`key`'s) keeps them, while the frame due is `target`
// and both videos shown have come to `reach`.
void NativePool::take_frames(const std::string& key, NativeVideo& player, long long target, long long reach) {
    const NativeUpdate update = player.poll();
    if (update.error) {
        throw std::runtime_error(update.error->toStdString());
    }
    if (key == source_key_) {
        // The soundtrack's clock counts from the source's first frame
        // too, known from its first frame, before the sound starts.
        if (const std::optional<int> first_ms = player.first_frame_ms();
            first_ms && *first_ms != audio_->offset_ms() && !audio_running_) {
            audio_->set_offset(*first_ms, position_);
        }
    }
    if (update.status) {
        status_details_[key] = *update.status;
    }
    if (update.ended) {
        eos_.insert(key);
    }
    std::map<long long, NativeFramePtr>& queue = frames_[key];
    // Keep two frames before the one shown (while playing, before the one
    // about to be shown), for instant S and arrow-key swaps; the pair
    // shown holds its own. While playing, the floor counts from the
    // target: counted from the frame shown, the queue would still be full
    // of past frames when the clock moved two frames on, and one frame in
    // five would be skipped at 120 fps. The floor also stops at how far
    // both videos shown have come: counted from this queue alone, a video
    // decoded ahead of the other would drop frames the other has yet to
    // reach, so the pair waited for would never come.
    const long long newest = queue.empty() ? frame_ : queue.rbegin()->first;
    const long long floor = std::min({std::max(frame_, target), newest, reach}) - 2;
    std::erase_if(queue, [floor](const auto& item) { return item.first < floor; });
    // Two previous frames, the current one and two upcoming ones: a limit
    // on the GPU surfaces held, while a step back or forward stays
    // instant. Nothing is taken while the decoder's seek is still to be
    // made: until then it can still give frames from before the seek.
    while (queue.size() < pool_kept_frames && !player.seeking()) {
        NativeFramePtr sample = player.try_pull();
        if (!sample) {
            break;
        }
        const long long number = round_even(static_cast<double>(sample->time_ns) / 1e9 * fps_);
        if (number >= frame_) {
            queue[number] = std::move(sample);
        }
    }
}

// Plays on, or buffers: when the pair due is not there (`matched`), or
// playback fell behind its soundtrack and could not catch up, the sound
// stops and is sought again; once the pair is there and the sound ready,
// both go on.
void NativePool::keep_up(bool matched, long long target) {
    // Falling behind is judged by the soundtrack clock's own reading, not the
    // value carried on between its steps, which picks the frame.
    const std::optional<double> reading = audio_running_ ? audio_->reading_ms() : std::nullopt;
    const long long due = reading ? round_even(*reading * fps_ / 1000) : target;
    bool behind = due - frame_ > 1 || (reading && *reading - position_ > 1000 / fps_);
    if (!behind) {
        late_since_.reset();
    } else if (matched && !buffering_) {  // a moment to catch up, with the sound playing on
        const double now = view_.clock();
        if (!late_since_) {
            late_since_ = now;
        }
        behind = now - *late_since_ > catch_up_seconds;
    }
    if (!matched || behind) {
        late_since_.reset();
        if (!buffering_) {
            audio_->set_playing(false);
            audio_running_ = false;
            if (!audio_->failed()) {
                audio_->seek(position_);
                audio_deadline_ = view_.clock() + audio_preroll_seconds;
            }
        }
        buffering_ = true;
    } else if (playing_ && (audio_->ready() || audio_->failed())) {
        if (buffering_) {
            anchor_ = view_.clock();
            anchor_frame_ = frame_;
        }
        buffering_ = false;
        if (!audio_running_ && !audio_->failed()) {
            audio_->set_playing(true);
            audio_running_ = true;
        }
    }
}

// Whether a video shown has ended at the frame shown: playback ends there.
void NativePool::check_ended() {
    for (const std::string& key : {source_key_, distorted_key(selected_)}) {
        const std::map<long long, NativeFramePtr>& queue = frames_[key];
        if (eos_.contains(key) && (queue.empty() || queue.rbegin()->first <= frame_)) {
            ended_ = true;
            audio_->set_playing(false);
        }
    }
}

std::optional<int> NativePool::next_tick_ms() const {
    // 1 ms after the next frame becomes due, which is half a frame before its
    // time, because poll rounds the clock. Ticking every half frame instead,
    // most ticks would find nothing to do (250 a second at 120 fps).
    const double frame_ms = 1000 / fps_;
    double wait = 0;
    if (clock_ms_) {
        wait = (static_cast<double>(frame_) + 0.5) * frame_ms - *clock_ms_ - (view_.clock() - clock_at_) * 1000;
    } else {
        wait = (anchor_ + static_cast<double>(frame_ + 1 - anchor_frame_) / fps_ - view_.clock()) * 1000;
    }
    if (wait <= 0) {
        return std::nullopt;
    }
    return static_cast<int>(std::min(std::ceil(wait) + 1, std::ceil(frame_ms)));
}

QString NativePool::description() const {
    // Paused on the pair on screen is not buffering, even after a seek.
    const QString state = !shown_ || (buffering_ && playing_) ? QStringLiteral("Buffering locked pair")
                                                             : QStringLiteral("Frame-locked GPU pair");
    const QString audio = audio_->failed() ? QStringLiteral("audio unavailable") : QStringLiteral("source soundtrack");
    QStringList parts{NativePresentation::name()};
    for (const std::string& key : {source_key_, distorted_key(selected_)}) {
        if (const auto detail = status_details_.find(key); detail != status_details_.end() && !detail->second.isEmpty()) {
            parts.append(detail->second);
        }
    }
    const QString separator = QString::fromUtf8(" \xC2\xB7 ");
    return state + separator + audio + separator + parts.join(separator);
}

void NativePool::resize() {
    surface_->setGeometry(view_.rect());
    place();
}

void NativePool::stop() {
    if (closed_) {
        return;
    }
    closed_ = true;
    for (const auto& [key, player] : entries_) {
        retire_video(player);
    }
    entries_.clear();
    frames_.clear();
    pair_.reset();
    PairedFrameWidget* surface = surface_;
    if (audio_) {
        audio_->set_enabled(false);
    }
    if (surface != nullptr) {
        surface->hide();
    }
    if (output_ || audio_) {
        // The presenter and the sound end on a thread of their own, each
        // waiting for its thread; the presenter's window sits in the surface,
        // which goes once the presenter has.
        auto* ender = new Ender([output = std::shared_ptr<NativeOutput>(std::move(output_)),
                                 audio = std::shared_ptr<Soundtrack>(std::move(audio_))]() mutable {
            audio.reset();
            output.reset();
        }, &view_);
        view_.stopping_.insert(ender);
        QObject::connect(ender, &QThread::finished, &view_, [view = &view_, ender, surface] {
            view->stopping_.erase(ender);
            ender->deleteLater();
            if (surface != nullptr) {
                surface->deleteLater();
            }
        });
        ender->start();
    } else if (surface != nullptr) {
        surface->deleteLater();
    }
    surface_ = nullptr;
}

}  // namespace vml::app
