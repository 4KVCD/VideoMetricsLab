// The Video Compare tab (the Python app's vmaf_app/ui/frame_compare_panel.py):
// the source and a test video frame by frame, with zoom, a seek bar, the
// frame's scores and HDR display handling.
//
// Ported so far: the still frames, which FFmpeg extracts (frame_extract.h)
// for every pair. Playback (the Python app's VideoCompareView) is step 6 of
// the port: until then its controls are shown but cannot be used, and pairs
// the Python app plays show as still frames, which it also shows for
// resolution tests and pairs it cannot play.
#pragma once

#include <QImage>
#include <QMetaType>
#include <QPixmap>
#include <QPointF>
#include <QString>
#include <QThread>
#include <QWidget>

#include <compare>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/frame_extract.h"
#include "vml/platform/system.h"
#include "vml/results.h"
#include "zoom.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSlider;
class QSpinBox;
class QStackedWidget;
class QTimer;
class QVBoxLayout;

namespace vml::app {

class ElidedLabel;

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

// A still frame, fitted to the view or zoomed (the tab's one Zoom), dragged
// with the mouse when larger than the view.
class FrameView : public QWidget {
    Q_OBJECT

public:
    explicit FrameView(Zoom& zoom, QWidget* parent = nullptr);

    // Shows a message instead of a frame.
    void set_message(const QString& message);
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
    bool can_drag() const;
    void refresh_cursor();

    Zoom& zoom_;
    QLabel* label_ = nullptr;
    QImage image_;
    // The frame scaled down smoothly, at the size last drawn: a drag paints
    // often, and scaling a 4K frame each time is slow.
    QPixmap scaled_;
    std::optional<QPointF> drag_from_;
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
    // The frame, decoded here: by the window's thread a 4K frame held it
    // for 64-104 ms. A null image if it could not be read.
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
// both at once.
class CropDetectWorker : public QThread {
    Q_OBJECT

public:
    using Known = std::optional<std::optional<CropBox>>;  // nothing: not known yet
    CropDetectWorker(std::string ffmpeg, VideoInfo source, VideoInfo distorted, Known known_source,
                     Known known_distorted, QObject* parent = nullptr);
    void cancel() { cancel_.cancel(); }

signals:
    void ready(const vml::app::DetectedCrops& crops);

protected:
    void run() override;

private:
    std::optional<CropBox> detect(const VideoInfo& info);

    std::string ffmpeg_;
    VideoInfo source_;
    VideoInfo distorted_;
    Known known_source_;
    Known known_distorted_;
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
    const FrameComparisonEntry* current_entry() const;
    // Shows frame `frame`.
    void set_frame(long long frame);
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

    // The controls, for the window's tests.
    QPushButton* previous_video_button = nullptr;
    QComboBox* video_combo = nullptr;
    QPushButton* next_video_button = nullptr;
    QComboBox* zoom_combo = nullptr;
    QComboBox* color_mode_combo = nullptr;
    QLabel* color_status_label = nullptr;
    QComboBox* source_resolution_combo = nullptr;
    QPushButton* advanced_info_button = nullptr;
    QLabel* showing_label = nullptr;
    FrameView* viewer = nullptr;
    QPushButton* play_button = nullptr;
    QCheckBox* audio_checkbox = nullptr;
    QPushButton* previous_frame_button = nullptr;
    QSpinBox* frame_spin = nullptr;
    QLineEdit* timestamp_edit = nullptr;
    QPushButton* next_frame_button = nullptr;
    QSlider* timeline = nullptr;
    ElidedLabel* detail_label = nullptr;
    QLabel* guide_label = nullptr;
    // The playback view, made when a pair first plays.
    VideoCompareView* video_view = nullptr;

signals:
    // The colour mode chosen, to save ("display_aware", ...).
    void color_mode_changed(const QString& mode);

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    friend class FramePanelTest;  // the panel's tests (tests/app)
    friend class WindowTest;      // the window's tests: the pairs it lists

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
        auto operator<=>(const PairKey&) const = default;
    };
    static PairKey pair_key(const FrameComparisonEntry& entry);

    // The constructor's parts: the panel's rows, top to bottom, added to
    // `root`, and its timers.
    void add_video_row(QVBoxLayout* root);
    void add_colour_row(QVBoxLayout* root);
    void add_view(QVBoxLayout* root);
    void add_seek_rows(QVBoxLayout* root);
    void add_labels(QVBoxLayout* root);
    void make_timers();
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
    FrameComparisonEntry apply_auto_crop(const FrameComparisonEntry& entry) const;
    void ensure_auto_crop(const FrameComparisonEntry* entry);
    void on_auto_crop_ready(const PairKey& key, const VideoInfo& source, const VideoInfo& distorted,
                            const DetectedCrops& files);
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

    std::function<std::string()> ffmpeg_;
    int decoded_videos_;
    std::vector<FrameComparisonEntry> entries_;
    std::size_t current_index_ = 0;
    long long frame_ = 0;
    bool showing_source_ = false;
    int generation_ = 0;
    std::set<FrameExtractWorker*> workers_;
    std::map<PairKey, CropDetectWorker*> crop_workers_;
    // Black bars found. By pair: the boxes compared, the picture both files
    // show. By file (path, size and time): the file's own bars, so that
    // another pair with the same file does not detect them again.
    std::map<PairKey, DetectedCrops> auto_crops_;
    std::map<std::string, std::optional<CropBox>> auto_crop_files_;
    bool window_filters_installed_ = false;
    PreviewColorMode color_mode_;
    platform::DisplayHdrInfo display_hdr_;
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
};

}  // namespace vml::app

Q_DECLARE_METATYPE(vml::app::DetectedCrops)
