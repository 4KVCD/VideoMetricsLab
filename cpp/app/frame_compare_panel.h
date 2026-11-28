// The Video Compare tab (the Python app's vmaf_app/ui/frame_compare_panel.py):
// the source and a test video frame by frame, with zoom, a seek bar, the
// frame's scores and HDR display handling. Still frames are made by FFmpeg,
// from the app's own decoder's pictures for VVC (frame_extract.h); pairs that can play are played by a
// VideoCompareView (video_compare_view.h), and resolution tests and pairs
// that cannot play show as still frames.
#pragma once

#include <QImage>
#include <QMetaType>
#include <QPixmap>
#include <QPoint>
#include <QPointer>
#include <QString>
#include <QThread>
#include <QTimer>
#include <QWidget>

#include <compare>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "quick_save.h"
#include "display_hdr.h"
#include "frame_drag.h"
#include "video_names.h"
#include "worst_frame_list.h"
#include "vml/cancel.h"
#include "vml/frame_extract.h"
#include "vml/platform/system.h"
#include "vml/results.h"
#include "zoom.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QHBoxLayout;
class QLabel;
class QLineEdit;
class QPushButton;
class QSlider;
class QSpacerItem;
class QSpinBox;
class QStackedWidget;
class QTimer;
class QBoxLayout;
class QVBoxLayout;

class QKeyEvent;

namespace vml::app {

class BandSlider;
class ElidedLabel;
class WrappingLabel;
class KeyGuide;
class PlayPauseButton;
class TextPanel;

// One video pair to choose from. `scores` are its run's, if it has one:
// the frames of a pair are worth comparing whether or not anyone has
// measured them.
struct FrameComparisonEntry {
    std::string identity;
    QString label;
    FrameComparison comparison;
    std::shared_ptr<const ComparisonResult> scores;
    // Its row in the Videos tab, which stays the same while `identity`
    // changes with the scores the row gets or loses, so that the video
    // being watched stays chosen; empty for none.
    std::string row;
};

// Seconds, M:SS or H:MM:SS(.sss) as seconds. Throws std::invalid_argument
// with the message to show (translated) when it is not one.
double parse_timestamp(const QString& value);

// The first frame of the loop around frame `centre`: `before` seconds
// before it (at `fps`), at frame 0 at the earliest.
long long loop_start(long long centre, double before, double fps);
// Where playback looping around frame `centre` jumps back to once it shows
// `frame`: the loop runs from `before` seconds before the centre to `after`
// seconds after it (at `fps`), and ends a frame before `last` at the latest
// (playback stops at the last frame). Its first frame when `frame` is past
// its end; nothing while inside it.
std::optional<long long> loop_restart(long long frame, long long centre, double before, double after, double fps,
                                      long long last);
// The frames that loop plays, its first and last (loop_restart's): what
// the loop row says and the timeline shades.
std::pair<long long, long long> loop_frames(long long centre, double before, double after, double fps,
                                            long long last);

// The file name of a still Export frame as PNG writes: the video, the frame (six
// digits at least, as the encoder counts it), its time, the metric and
// score when known ("PSNR 32.64"), and the side ("reference" or "test"),
// with what Windows refuses in a name replaced.
QString still_file_name(const QString& video, long long frame, double seconds, const QString& metric_score,
                        const QString& side);

// A still frame, fitted to the view or zoomed (the tab's one Zoom), dragged
// with the mouse when larger than the view. Without one, a black screen
// with a message and the tab's keys in large type (key_guide.h).
class FrameView : public QWidget {
    Q_OBJECT

public:
    explicit FrameView(Zoom& zoom, QWidget* parent = nullptr);

    // Shows a message instead of a frame, and the keys under it unless not
    // `with_keys`.
    void set_message(const QString& message, bool with_keys = true);
    // The message with nothing to compare, the screen in the theme's colours
    // rather than black (video_surround.h), until a frame or another message.
    void show_empty(const QString& message);
    // Shows a frame.
    void set_image(const QImage& image);
    // Draws the frame again after the zoom changed.
    void zoom_changed();
    // The message shown, if any (for the tests).
    QString message() const;
    const QImage& image() const { return image_; }

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void fit_screen_text();
    // The message's and the keys' colours: the theme's while empty_, else
    // light on black.
    void show_look();

    Zoom& zoom_;
    QWidget* screen_text_ = nullptr;  // the message and the keys
    QLabel* label_ = nullptr;
    KeyGuide* key_guide_ = nullptr;
    bool empty_ = false;  // nothing to compare: the theme's colours (show_empty)
    QImage image_;
    // The frame scaled down smoothly, at the size last drawn: a drag paints
    // often, and scaling a 4K frame each time is slow.
    QPixmap scaled_;
    FrameDrag drag_;
};

// Extracts the frame of each side asked for, in turn, on a thread of its
// own.
class FrameExtractWorker : public QThread {
    Q_OBJECT

public:
    FrameExtractWorker(int generation, std::string ffmpeg, FrameComparison comparison, long long frame,
                       std::vector<FrameSide> sides, PreviewColorSettings colors, QObject* parent = nullptr);
    // Stops the extraction, ending FFmpeg.
    void cancel() { cancel_.cancel(); }

signals:
    // The frame, decoded here, off the window's thread, which a 4K frame
    // would hold up for 64-104 ms. A null image if it could not be read.
    void frame_ready(int generation, int side, const QImage& image);
    void frame_failed(int generation, int side, const QString& error);

protected:
    void run() override;

private:
    int generation_;
    std::string ffmpeg_;
    FrameComparison comparison_;
    long long frame_;
    std::vector<FrameSide> sides_;
    PreviewColorSettings colors_;
    CancelToken cancel_;
};

// Black-bar boxes for a pair's two videos; nothing for a side where none
// could be found.
struct DetectedCrops {
    std::optional<CropBox> source;
    std::optional<CropBox> distorted;
};

// Detects the black bars of a pair's two files on a thread of its own, each
// file's own bars: those not known yet (`known_source`, `known_distorted`),
// both at once; then the boxes the pair compares (`test_full_frame`: Source
// only's), which may need pictures of both videos decoded (crops_placed).
class CropDetectWorker : public QThread {
    Q_OBJECT

public:
    using Known = std::optional<std::optional<CropBox>>;  // nothing: not known yet
    CropDetectWorker(std::string ffmpeg, VideoInfo source, VideoInfo distorted, Known known_source,
                     Known known_distorted, bool test_full_frame, QObject* parent = nullptr);
    void cancel() { cancel_.cancel(); }

signals:
    // Each file's own boxes, and the boxes the pair compares.
    void ready(const vml::app::DetectedCrops& files, const vml::app::DetectedCrops& compared);

protected:
    void run() override;

private:
    std::optional<CropBox> detect(const VideoInfo& info);

    std::string ffmpeg_;
    VideoInfo source_;
    VideoInfo distorted_;
    Known known_source_;
    Known known_distorted_;
    bool test_full_frame_;
    CancelToken cancel_;
};

// Saves the reference's and the test video's frame as PNGs (Export frame as PNG), on
// a thread of its own: Video Compare's still decode of that frame, HDR
// converted to SDR (an 8-bit PNG), cropped and scaled as compared, written
// as FFmpeg made them. Both files or neither: a test still that fails
// removes the reference's.
class StillSaveWorker : public QThread {
    Q_OBJECT

public:
    // `files`: the reference's, then the test video's; `mode`: the tab's
    // colour mode.
    StillSaveWorker(std::string ffmpeg, FrameComparison comparison, long long frame,
                    std::vector<std::filesystem::path> files, PreviewColorMode mode, QObject* parent = nullptr);
    void cancel() { cancel_.cancel(); }

signals:
    void saved(const QStringList& files);
    void failed(const QString& error);

protected:
    void run() override;

private:
    std::string ffmpeg_;
    FrameComparison comparison_;
    long long frame_;
    std::vector<std::filesystem::path> files_;
    PreviewColorMode mode_;
    CancelToken cancel_;
};

class FramePanelTest;
class VideoCompareView;

class FrameComparePanel : public QWidget {
    Q_OBJECT

public:
    // `ffmpeg` gives the FFmpeg to run (the Settings tab can change it);
    // `color_mode` and `decoded_videos` are the saved settings.
    FrameComparePanel(std::function<std::string()> ffmpeg, PreviewColorMode color_mode, int decoded_videos,
                      QWidget* parent = nullptr);
    // Stops its threads and waits for them: a thread must not outlive the
    // panel it writes into (each ends at once, ending its FFmpeg).
    ~FrameComparePanel() override;

    // The pairs to choose from, keeping the one shown if it is still there.
    void set_runs(std::vector<FrameComparisonEntry> entries);
    // How the videos' names are shown (Settings > Window > Video names).
    void set_video_name_rule(const VideoNameRule& rule);
    const FrameComparisonEntry* current_entry() const;
    // Shows frame `frame`.
    void set_frame(long long frame);
    // Chooses the pair `identity` names and shows its frame `frame` (when
    // off screen, once shown). Not when no pair has that identity, or the
    // frame is past the frames the tab offers (the shortest pair's).
    enum class OpenAt { shown, no_pair, past_end };
    OpenAt open_at(const std::string& identity, long long frame);
    // The last frame the tab offers.
    long long last_frame() const;
    // The Worst frames list to step through with Previous / Next worst, at
    // its stop `index` (the one just opened); cleared by anything else
    // opening a frame from outside.
    void set_worst_list(WorstFrameList list, std::size_t index);
    void clear_worst_list();
    // The loop's seconds before and after its frame (the saved settings).
    void set_loop_seconds(double before, double after);
    // Loop ticked or not, as last left (the saved settings).
    void set_loop_on(bool on);
    // Where the frame pair's save icon writes: Settings' Export folder; and
    // the folder its Save as... asks for (nothing: Cancel).
    std::function<QString()> export_folder;
    std::function<QString()> choose_folder;
    // Shows the next (1) or previous (-1) test video.
    void cycle_distorted(int direction);
    // How many test videos playback keeps decoding (the Settings tab).
    void set_decoded_videos(int count);
    // The pointer is over this tab: what playback first needs is made now,
    // so that it is mostly ready by the click, and let go of again if the tab
    // is not shown within release_after_ms.
    void prepare_playback();
    // The threads still running: frame extraction and black-bar detection.
    std::vector<QThread*> live_workers() const;
    // Stops what is pending or running, as the window closes.
    void cancel();

    // Advanced info as it is now: playback's details, a line each, plain
    // (what its panel shows and copies).
    QString advanced_info_text() const;

    // The controls, for the window's tests.
    QPushButton* previous_video_button = nullptr;
    QComboBox* video_combo = nullptr;
    QPushButton* next_video_button = nullptr;
    QComboBox* zoom_combo = nullptr;
    QComboBox* color_mode_combo = nullptr;
    QLabel* color_status_label = nullptr;
    QComboBox* source_resolution_combo = nullptr;
    QPushButton* advanced_info_button = nullptr;
    ElidedLabel* showing_label = nullptr;
    FrameView* viewer = nullptr;
    PlayPauseButton* play_button = nullptr;
    QCheckBox* audio_checkbox = nullptr;
    QPushButton* previous_frame_button = nullptr;
    QSpinBox* frame_spin = nullptr;
    QLineEdit* timestamp_edit = nullptr;
    QPushButton* next_frame_button = nullptr;
    BandSlider* timeline = nullptr;
    // Issue #24: the worst frames handed over, the loop, the stills.
    QPushButton* previous_worst_button = nullptr;
    QLabel* worst_label = nullptr;
    QPushButton* next_worst_button = nullptr;
    QCheckBox* loop_checkbox = nullptr;
    QDoubleSpinBox* loop_before_spin = nullptr;
    QDoubleSpinBox* loop_after_spin = nullptr;
    // The loop's seconds in frames of the pair shown (refresh_loop_frames).
    QSpinBox* loop_before_frames_spin = nullptr;
    QSpinBox* loop_after_frames_spin = nullptr;
    // The sentence's words, and the frames the loop plays (refresh_loop_text).
    QLabel* loop_from_label = nullptr;
    QLabel* loop_before_to_label = nullptr;
    QLabel* loop_after_label = nullptr;
    QLabel* loop_range_label = nullptr;
    SaveButton* save_still_button = nullptr;
    // Previous / Next worst and where the list is: shown while the list's
    // video is (update_worst_row).
    QWidget* worst_row_ = nullptr;
    WrappingLabel* detail_label = nullptr;
    QLabel* guide_label = nullptr;
    // The playback view, made when a pair first plays.
    VideoCompareView* video_view = nullptr;

signals:
    // The colour mode chosen, to save ("display_aware", ...).
    void color_mode_changed(const QString& mode);
    // The loop's seconds changed, to save.
    void loop_seconds_changed(double before, double after);
    // Loop ticked or unticked, to save.
    void loop_changed(bool on);
    // Export frame as PNG wrote `files`, or could not.
    void still_saved(const QStringList& files);
    void still_failed(const QString& error);
    // Previous / Next worst opened `frame`, a stop of the Worst frames list
    // of `identity` and `metric`, so that the list shows that row on the way back.
    void worst_stop_shown(const std::string& identity, const QString& metric, int frame);

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;
    // The tab's keys (key_bindings.h): whether one is taken, or nothing for
    // a key that is none of the tab's.
    std::optional<bool> on_compare_key(const QKeyEvent& key);
    // What a key or mouse gesture of the tab does (key_bindings.h's ids).
    void run_compare_action(std::string_view id);
    // The mouse on the video or still frame; whether it took the event.
    bool on_video_mouse(QObject* watched, QEvent* event);
    // The tab's keys as they are now in its hint line and its buttons' tooltips.
    void show_keys();
    std::optional<QPoint> click_from_;  // where a left press began, until released
    QTimer click_timer_;                // a click, once no double-click can follow

private:
    friend class FramePanelTest;  // the panel's tests (tests/app)

    // A frame kept: whose, which, and what it is made from. A pair keeps its
    // identity while what its frames are made from changes (its black bars
    // found, its scaling edited, another reference chosen), and a frame kept
    // from before must not stand for the new one.
    struct CacheKey {
        std::string identity;  // the pair's
        long long frame = 0;
        int side = 0;
        std::string file;      // the side's video
        std::string timing;    // where its frames start, and the frame rate
        std::string colours;   // PreviewColorSettings::cache_token
        std::string filters;   // the crops, the scaling, the tone mapping
        std::string grain;     // what its film grain depends on (still_grain)
        auto operator<=>(const CacheKey&) const = default;
    };
    // A pair's key for its black bars: the pair, and its two files (each
    // one's path, size and time). A pair keeps its identity when another
    // reference is chosen, and the bars found against the old one are not
    // the new one's.
    struct PairKey {
        std::string identity;
        std::string source_file;
        std::string distorted_file;
        bool test_full_frame = false;  // FrameComparison::test_full_frame: the compared boxes differ
        auto operator<=>(const PairKey&) const = default;
    };
    static PairKey pair_key(const FrameComparisonEntry& entry);

    // The constructor's parts: the panel's rows, top to bottom, added to
    // `root`, and its timers.
    void add_video_row(QBoxLayout* root);
    void add_colour_row(QBoxLayout* root);
    void add_view(QVBoxLayout* root);
    void add_seek_rows(QBoxLayout* root);
    void add_labels(QVBoxLayout* root);
    void add_inspect_row(QBoxLayout* root);
    void make_timers();
    // Issue #24's tools (frame_compare_inspect.cpp).
    void step_worst(int direction);
    void update_worst_row();
    // The pair shown's frame rate, 0 with none (the loop's frame boxes).
    double loop_fps() const;
    // The loop's frame boxes from its seconds, at the pair shown's frame
    // rate; empty and unusable where it has none.
    void refresh_loop_frames();
    // The row's frame numbers and tooltips, the range it plays and its band
    // on the timeline, for the loop's centre now; its boxes usable only while
    // Loop is ticked.
    void refresh_loop_text();
    void on_loop_toggled(bool on);
    void loop_jump_to(long long frame);
    void follow_loop(long long frame);
    void restart_loop_at_end();
    // This frame of the reference and of the test video, as PNG, in `folder`.
    void save_still(const QString& folder);
    void release_held();
    void on_video_selected(int index);
    void on_zoom_chosen(int index);
    void on_zoom_typed();
    void apply_zoom(std::optional<double> factor);
    void show_zoom();
    void on_color_mode_changed(int index);
    void on_timestamp_committed();
    void set_ranges(long long maximum);
    void sync_seek_widgets();
    void update_enabled_state();
    void update_labels();
    // shown_names_ from the entries: the title's names where the whole ones
    // do not fit (the video list keeps the whole names).
    void name_videos();
    void update_frame_line();
    void update_color_status();
    QString score_text(const FrameComparisonEntry& entry) const;
    PreviewColorSettings color_settings() const;
    bool refresh_display_hdr();
    void on_display_maybe_changed();
    std::optional<CacheKey> cache_key(FrameSide side) const;
    const QImage* current_image() const;
    void show_or_request();
    void request_current_frames();
    bool show_kept_frame(const CacheKey& key);
    void show_chosen_video();
    FrameComparisonEntry apply_auto_crop(const FrameComparisonEntry& entry) const;
    void ensure_auto_crop(const FrameComparisonEntry* entry);
    // `compared`: the pair's compared boxes as the worker found them; none:
    // from the files' boxes alone.
    void on_auto_crop_ready(const PairKey& key, const VideoInfo& source, const VideoInfo& distorted,
                            const DetectedCrops& files, const std::optional<DetectedCrops>& compared = std::nullopt);
    void on_frame_ready(int generation, int side, const QImage& image);
    void on_frame_failed(int generation, int side, const QString& error);
    void cancel_workers();
    void cancel_crop_workers();
    FrameSide shown_side() const { return showing_source_ ? FrameSide::source : FrameSide::distorted; }
    // Playback (the pairs that can play).
    bool is_video_mode() const;
    VideoCompareView& ensure_video_view();
    void load_current_video(std::optional<bool> playing = std::nullopt);
    void on_play_clicked();
    void on_video_position_changed(int position_ms);
    void show_video_position();
    void on_video_playing_changed(bool playing);
    void on_video_status_changed(const QString& message);
    QString concise_playback_status() const;
    // Advanced info's panel (TextPanel), under its button; kept up to date
    // while it is open.
    void open_advanced_info();

    std::function<std::string()> ffmpeg_;
    int decoded_videos_;
    std::vector<FrameComparisonEntry> entries_;
    // Each entry's label shortened (video_names.h), in entries_' order: the
    // title over the video where the whole label does not fit.
    std::vector<QString> shown_names_;
    VideoNameRule name_rule_;
    std::size_t current_index_ = 0;
    long long frame_ = 0;
    bool showing_source_ = false;
    int generation_ = 0;
    std::set<FrameExtractWorker*> workers_;
    // Black-bar detections under way, by pair; and those cancelled but not
    // yet ended, which a pair shown again does not wait for: it detects anew.
    std::map<PairKey, CropDetectWorker*> crop_workers_;
    std::set<CropDetectWorker*> ending_crop_workers_;
    // Black bars found. By pair: the boxes compared, the picture both files
    // show. By file (path, size and time): the file's own bars, so that
    // another pair with the same file does not detect them again.
    std::map<PairKey, DetectedCrops> auto_crops_;
    std::map<std::string, std::optional<CropBox>> auto_crop_files_;
    bool window_filters_installed_ = false;
    PreviewColorMode color_mode_;
    platform::DisplayHdrInfo display_hdr_;
    // Asked again once, 200 ms later, when the display could not be known
    // yet (a window shown before it is on a screen): once per show.
    bool display_retried_ = false;
    // The displays' settings changing (macOS), which re-asks; a member, gone
    // before the timers it starts.
    std::unique_ptr<DisplayWatch> display_watch_;
    // The frames extracted, oldest first, and why some could not be.
    std::deque<std::pair<CacheKey, QImage>> cache_;
    std::map<CacheKey, QString> errors_;
    Zoom zoom_;
    QStackedWidget* content_stack_ = nullptr;
    QTimer* seek_timer_ = nullptr;
    QTimer* display_timer_ = nullptr;
    QTimer* position_timer_ = nullptr;
    // Off screen for a while, everything the tab holds is let go of
    // (hideEvent); the GPU's device again later if it was being made for a
    // start meanwhile.
    QTimer* release_timer_ = nullptr;
    enum class ReleaseStage { everything, device };
    ReleaseStage release_stage_ = ReleaseStage::everything;
    // Shown and not hidden since (showEvent, hideEvent): a minimized window
    // hides it too, though isVisible() stays true.
    bool on_screen_ = false;
    // Playback moving the seek widgets, which must not seek it back.
    bool syncing_video_position_ = false;
    // Playback's status as the view said it, in English.
    QString video_status_;
    QStringList advanced_info_lines_;  // playback's details, translated (on_video_status_changed)
    QPointer<TextPanel> advanced_info_panel_;
    // The Worst frames list handed over, and the stop shown last.
    std::optional<WorstFrameList> worst_list_;
    std::size_t worst_index_ = 0;
    // The loop: the frame it is around (the one the user last went to), a
    // jump back made whose position has not come yet, and whether the last
    // pause was the user's (not playback's own stop at the end).
    long long loop_centre_ = 0;
    bool loop_jump_pending_ = false;
    // restart_loop_at_end is under way (it is not entered again).
    bool restarting_loop_ = false;
    bool user_paused_ = false;
    StillSaveWorker* still_worker_ = nullptr;
};

}  // namespace vml::app

Q_DECLARE_METATYPE(vml::app::DetectedCrops)
