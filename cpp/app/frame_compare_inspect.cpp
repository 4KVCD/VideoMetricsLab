// Video Compare's tools for looking at a dip (issue #24): stepping through
// Metric Graphs' Worst frames list, looping playback around a frame, and
// saving a frame's stills.
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>
#include <fstream>

#include "frame_compare_panel.h"
#include "i18n.h"
#include "video_compare_view.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

// The loop's seconds before and after its frame: the spin boxes' limits,
// to the millisecond. None before or after is the frame alone on that side
// (loop_restart keeps the loop at least a frame long).
constexpr double loop_seconds_minimum = 0.0;
constexpr double loop_seconds_maximum = 30.0;
constexpr double default_loop_seconds = 1.0;
constexpr int loop_seconds_decimals = 3;
constexpr double loop_seconds_step = 0.1;

// A file that is not there yet: `file`, else "name (2).png" and so on.
std::filesystem::path unused_path(const std::filesystem::path& file) {
    std::error_code error;
    std::filesystem::path candidate = file;
    for (int number = 2; std::filesystem::exists(candidate, error); ++number) {
        candidate = file.parent_path() / platform::path_from_text(platform::path_text(file.stem()) + " (" +
                                                                  std::to_string(number) + ")" +
                                                                  platform::path_text(file.extension()));
    }
    return candidate;
}

}  // namespace

long long loop_start(long long centre, double before, double fps) {
    return std::max(0LL, centre - std::llround(before * std::max(0.0, fps)));
}

std::optional<long long> loop_restart(long long frame, long long centre, double before, double after, double fps,
                                      long long last) {
    if (fps <= 0) {
        return std::nullopt;
    }
    const long long first = loop_start(centre, before, fps);
    // At least a frame long, though it may then reach the last frame, where
    // playback stops (restart_loop_at_end starts it again).
    const long long end = std::max(first + 1, std::min(centre + std::llround(after * fps), last - 1));
    return frame > end ? std::optional(first) : std::nullopt;
}

QString still_file_name(const QString& video, long long frame, double seconds, const QString& metric_score,
                        const QString& side) {
    const long long ms = std::llround(std::max(0.0, seconds) * 1000);
    const QString time = QStringLiteral("%1h%2m%3.%4s")
                             .arg(ms / 3'600'000)
                             .arg(ms / 60'000 % 60, 2, 10, QChar('0'))
                             .arg(ms / 1000 % 60, 2, 10, QChar('0'))
                             .arg(ms % 1000, 3, 10, QChar('0'));
    QStringList parts{video, QStringLiteral("frame %1").arg(frame, 6, 10, QChar('0')), time};
    if (!metric_score.isEmpty()) {
        parts.append(metric_score);
    }
    parts.append(side);
    return suggested_file_name(parts.join(QStringLiteral(" - ")), QStringLiteral(".png"));
}

// ------------------------------------------------------------ Export frame as PNG

StillSaveWorker::StillSaveWorker(std::string ffmpeg, FrameComparison comparison, long long frame,
                                 std::vector<std::filesystem::path> files, PreviewColorMode mode, QObject* parent)
    : QThread(parent),
      ffmpeg_(std::move(ffmpeg)),
      comparison_(std::move(comparison)),
      frame_(frame),
      files_(std::move(files)),
      mode_(mode) {}

void StillSaveWorker::run() {
    // Both pictures first, then both files: a picture that cannot be made
    // leaves no file behind.
    std::vector<std::string> pictures;
    try {
        // As the tab shows still frames on an SDR display: HDR mapped to SDR
        // white (100 nits), SDR as it is. The display's own HDR state is
        // left out, as the file outlives it.
        const PreviewColorSettings colors{mode_, std::nullopt, std::nullopt};
        for (const FrameSide side : {FrameSide::source, FrameSide::distorted}) {
            pictures.push_back(
                extract_frame_png(ffmpeg_, comparison_, side, frame_, cancel_, colors, PngEncoding::compressed));
        }
    } catch (const Cancelled&) {
        return;
    } catch (const std::exception& error) {
        emit failed(QString::fromUtf8(error.what()));
        return;
    }
    QStringList written;
    std::error_code no_folder;  // a folder that cannot be made fails at the write below
    for (const std::filesystem::path& file : files_) {
        std::filesystem::create_directories(file.parent_path(), no_folder);
    }
    for (std::size_t index = 0; index < files_.size(); ++index) {
        std::ofstream out(files_[index], std::ios::binary);
        out.write(pictures[index].data(), static_cast<std::streamsize>(pictures[index].size()));
        out.close();
        if (!out) {
            std::error_code ignored;
            for (std::size_t made = 0; made <= index; ++made) {
                std::filesystem::remove(files_[made], ignored);
            }
            emit failed(app::tr("Could not write {path}", {{"path", text_of(files_[index])}}));
            return;
        }
        written.append(text_of(files_[index]));
    }
    emit saved(written);
}

// ------------------------------------------------------------ the row

// Previous / Next worst with where the list is, the loop, and Export frame as PNG:
// centred, as the controls above and the frame's line below are.
void FrameComparePanel::add_inspect_row(QBoxLayout* root) {
    auto* row = new QHBoxLayout;
    inspect_row_ = row;
    row->addStretch(1);
    previous_worst_button = new QPushButton(app::tr("Previous worst"));
    connect(previous_worst_button, &QPushButton::clicked, this, [this] { step_worst(-1); });
    worst_label = new QLabel;
    next_worst_button = new QPushButton(app::tr("Next worst"));
    connect(next_worst_button, &QPushButton::clicked, this, [this] { step_worst(1); });
    for (QWidget* widget : std::initializer_list<QWidget*>{previous_worst_button, worst_label, next_worst_button}) {
        widget->setToolTip(app::tr("Steps through the Worst frames list opened from Metric Graphs."));
        widget->hide();
        row->addWidget(widget);
    }
    worst_spacer_ = new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Minimum);
    row->addSpacerItem(worst_spacer_);  // 24 px while the worst frames are shown (update_worst_row)

    loop_checkbox = new QCheckBox(app::tr("Loop"));
    loop_checkbox->setToolTip(app::tr("Plays the seconds around the frame you went to, again and again: Play starts "
                                      "them from the beginning."));
    connect(loop_checkbox, &QCheckBox::toggled, this, [this](bool on) {
        on_loop_toggled(on);
        emit loop_changed(on);
    });
    // Each side in seconds, to the millisecond, and in frames of the pair
    // shown: either one typed sets the other (refresh_loop_frames). The
    // seconds are what is saved, and what the loop plays.
    const auto seconds_spin = [this] {
        auto* spin = new QDoubleSpinBox;
        spin->setRange(loop_seconds_minimum, loop_seconds_maximum);
        spin->setSingleStep(loop_seconds_step);
        spin->setDecimals(loop_seconds_decimals);
        spin->setValue(default_loop_seconds);
        spin->setSuffix(QStringLiteral(" s"));
        spin->setKeyboardTracking(false);
        connect(spin, &QDoubleSpinBox::valueChanged, this, [this] {
            refresh_loop_frames();
            emit loop_seconds_changed(loop_before_spin->value(), loop_after_spin->value());
        });
        return spin;
    };
    const auto frames_spin = [this](QDoubleSpinBox* seconds) {
        auto* spin = new QSpinBox;
        spin->setSuffix(app::tr(" frames"));
        spin->setKeyboardTracking(false);
        connect(spin, &QSpinBox::valueChanged, this, [this, seconds](int frames) {
            const double fps = loop_fps();
            if (fps > 0) {
                seconds->setValue(static_cast<double>(frames) / fps);  // to its millisecond
            }
        });
        return spin;
    };
    loop_before_spin = seconds_spin();
    loop_after_spin = seconds_spin();
    loop_before_frames_spin = frames_spin(loop_before_spin);
    loop_after_frames_spin = frames_spin(loop_after_spin);
    row->addWidget(loop_checkbox);
    row->addWidget(new QLabel(app::tr("before:")));
    row->addWidget(loop_before_spin);
    row->addWidget(loop_before_frames_spin);
    row->addWidget(new QLabel(app::tr("after:")));
    row->addWidget(loop_after_spin);
    row->addWidget(loop_after_frames_spin);
    row->addSpacing(12);

    save_still_button = new QPushButton(app::tr("Export frame as PNG"));
    save_still_button->setToolTip(app::tr("Saves this frame of the reference and of the test video as PNG files in "
                                          "Settings' Export folder, cropped and scaled as they are compared, HDR "
                                          "mapped to SDR. They are decoded as Video Compare's still frames are."));
    connect(save_still_button, &QPushButton::clicked, this, [this] { save_still(); });
    row->addWidget(save_still_button);
    row->addStretch(1);
    root->addLayout(row);
}

// ------------------------------------------------------------ worst frames

void FrameComparePanel::set_worst_list(WorstFrameList list, std::size_t index) {
    if (list.stops.empty()) {
        clear_worst_list();
        return;
    }
    worst_index_ = std::min(index, list.stops.size() - 1);
    worst_list_ = std::move(list);
    update_worst_row();
}

void FrameComparePanel::clear_worst_list() {
    worst_list_.reset();
    update_worst_row();
}

void FrameComparePanel::step_worst(int direction) {
    if (!worst_list_) {
        return;
    }
    const long long next = static_cast<long long>(worst_index_) + direction;
    if (next < 0 || next >= static_cast<long long>(worst_list_->stops.size())) {
        return;
    }
    worst_index_ = static_cast<std::size_t>(next);
    open_at(worst_list_->identity, worst_list_->stops[worst_index_].frame);
    update_worst_row();
}

// The list's controls: shown while the pair shown is the list's.
void FrameComparePanel::update_worst_row() {
    const FrameComparisonEntry* entry = current_entry();
    const bool shown = worst_list_ && entry != nullptr && entry->identity == worst_list_->identity;
    for (QWidget* widget : std::initializer_list<QWidget*>{previous_worst_button, worst_label, next_worst_button}) {
        widget->setVisible(shown);
    }
    // Room between them and the loop only while they show: else the centred
    // row would sit off its centre by it.
    worst_spacer_->changeSize(shown ? 24 : 0, 0, QSizePolicy::Fixed, QSizePolicy::Minimum);
    inspect_row_->invalidate();
    if (!shown) {
        return;
    }
    const WorstFrameList::Stop& stop = worst_list_->stops[worst_index_];
    worst_label->setText(app::tr("Worst {number} of {count}: {metric} {score}",
                                 {{"number", worst_index_ + 1},
                                  {"count", worst_list_->stops.size()},
                                  {"metric", worst_list_->metric},
                                  {"score", stop.score}}));
    previous_worst_button->setEnabled(worst_index_ > 0);
    next_worst_button->setEnabled(worst_index_ + 1 < worst_list_->stops.size());
}

// ------------------------------------------------------------ the loop

void FrameComparePanel::set_loop_seconds(double before, double after) {
    {
        const QSignalBlocker before_blocker(loop_before_spin);
        const QSignalBlocker after_blocker(loop_after_spin);
        loop_before_spin->setValue(before);
        loop_after_spin->setValue(after);
    }
    refresh_loop_frames();
}

void FrameComparePanel::set_loop_on(bool on) {
    // As it was left: no loop starts (none is playing yet); the next frame
    // gone to is its centre.
    const QSignalBlocker blocker(loop_checkbox);
    loop_checkbox->setChecked(on);
}

double FrameComparePanel::loop_fps() const {
    const FrameComparisonEntry* entry = current_entry();
    return entry != nullptr ? entry->comparison.fps : 0.0;
}

void FrameComparePanel::refresh_loop_frames() {
    // The frame boxes show the seconds in the pair's frames, rounded as the
    // loop rounds them (loop_start), and reach as far as the seconds do. A
    // pair of no known frame rate leaves them empty, and they can't be used.
    const double fps = loop_fps();
    for (const auto& [seconds, frames] : {std::pair{loop_before_spin, loop_before_frames_spin},
                                          std::pair{loop_after_spin, loop_after_frames_spin}}) {
        const QSignalBlocker blocker(frames);
        if (fps <= 0) {
            frames->setRange(0, 0);
            frames->setSpecialValueText(QString::fromUtf8("\xE2\x80\x94"));  // an em dash: no frames
            frames->setValue(0);
            frames->setEnabled(false);
            continue;
        }
        frames->setSpecialValueText(QString());
        frames->setRange(static_cast<int>(std::llround(loop_seconds_minimum * fps)),
                         static_cast<int>(std::llround(loop_seconds_maximum * fps)));
        frames->setValue(static_cast<int>(std::llround(seconds->value() * fps)));
        frames->setEnabled(seconds->isEnabled());
    }
}

void FrameComparePanel::on_loop_toggled(bool on) {
    loop_jump_pending_ = false;
    loop_centre_ = frame_;
    if (on && current_entry() != nullptr && video_view != nullptr && video_view->playback_requested()) {
        // Playing: the loop starts now, from its beginning.
        loop_jump_to(loop_start(loop_centre_, loop_before_spin->value(), current_entry()->comparison.fps));
    }
}

// Playback goes to `frame` for the loop: not a seek of the user's, so the
// loop keeps its centre.
void FrameComparePanel::loop_jump_to(long long frame) {
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr || video_view == nullptr || entry->comparison.fps <= 0) {
        return;
    }
    loop_jump_pending_ = true;
    video_view->set_position(static_cast<int>(std::nearbyint(static_cast<double>(frame) / entry->comparison.fps * 1000)));
}

// Playback showing `frame`: past the loop's end, back to its start. The
// positions still coming from before a jump back are not acted on twice.
void FrameComparePanel::follow_loop(long long frame) {
    if (!loop_checkbox->isChecked() || video_view == nullptr || !video_view->is_playing() ||
        current_entry() == nullptr) {
        return;
    }
    const std::optional<long long> first = loop_restart(frame, loop_centre_, loop_before_spin->value(),
                                                        loop_after_spin->value(), current_entry()->comparison.fps,
                                                        last_frame());
    if (!first) {
        loop_jump_pending_ = false;  // the jump back has landed
    } else if (!loop_jump_pending_) {
        loop_jump_to(*first);
    }
}

// Playback stopped at the last frame while looping, on its own: plays the
// loop again.
void FrameComparePanel::restart_loop_at_end() {
    if (!loop_checkbox->isChecked() || user_paused_ || !on_screen_ || video_view == nullptr ||
        current_entry() == nullptr || frame_ < last_frame()) {
        return;
    }
    loop_jump_to(loop_start(loop_centre_, loop_before_spin->value(), current_entry()->comparison.fps));
    video_view->set_playing(true);
}

// ------------------------------------------------------------ the stills

void FrameComparePanel::save_still() {
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr || still_worker_ != nullptr) {
        return;
    }
    const QString folder = export_folder ? export_folder() : QString();
    if (folder.isEmpty()) {
        emit still_failed(app::tr("No export folder: choose one in Settings."));
        return;
    }
    // The list's metric and score where this frame is one of its stops.
    QString metric_score;
    if (worst_list_ && worst_list_->identity == entry->identity) {
        for (const WorstFrameList::Stop& stop : worst_list_->stops) {
            if (stop.frame == frame_) {
                metric_score = worst_list_->metric + " " + stop.score;
            }
        }
    }
    const double fps = entry->comparison.fps;
    const double seconds = fps > 0 ? static_cast<double>(frame_) / fps : 0.0;
    std::vector<std::filesystem::path> files;
    for (const char* side : {"reference", "test"}) {
        files.push_back(unused_path(path_of(folder) /
                                    path_of(still_file_name(entry->label, frame_, seconds, metric_score, side))));
    }
    auto* worker = new StillSaveWorker(ffmpeg_(), entry->comparison, frame_, files, color_mode_, this);
    connect(worker, &StillSaveWorker::saved, this, [this](const QStringList& written) { emit still_saved(written); });
    connect(worker, &StillSaveWorker::failed, this, [this](const QString& error) { emit still_failed(error); });
    connect(worker, &QThread::finished, this, [this, worker] {
        if (still_worker_ == worker) {
            still_worker_ = nullptr;
        }
        worker->deleteLater();
        update_enabled_state();
    });
    still_worker_ = worker;
    save_still_button->setEnabled(false);
    worker->start();
}

}  // namespace vml::app
