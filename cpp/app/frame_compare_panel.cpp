#include "frame_compare_panel.h"

#include <QAbstractSpinBox>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QRegularExpression>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTimer>
#include <QToolTip>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <thread>

#include "i18n.h"
#include "theme.h"
#include "video_compare_view.h"
#include "widgets.h"
#include "vml/crop_detect.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/python_text.h"
#include "vml/time_format.h"

// The panel's text is app::tr's: inside a Q_OBJECT class, tr() is Qt's.
namespace vml::app {

namespace {

// Shown whenever there is no pair to look at. It does not ask for a run:
// frames are worth comparing before anything has been measured.
const char* const nothing_to_compare = N_("Select a source video and add a test video to compare their frames.");

// The frames kept: at most this many, and while more than two, this many
// bytes.
constexpr std::size_t cache_limit = 12;
constexpr qsizetype cache_byte_limit = 192LL * 1024 * 1024;

QString text_of(const std::string& text) { return QString::fromStdString(text); }

// A file's identity for remembering its black bars: its path, size and time.
std::string crop_file_key(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(path, error);
    const std::uintmax_t size = std::filesystem::file_size(resolved, error);
    const std::optional<long long> modified = platform::modified_ns(resolved);
    if (error || !modified) {
        return platform::path_text(path);
    }
    return platform::path_text(resolved) + "|" + std::to_string(size) + "|" + std::to_string(*modified);
}

bool is_file(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

}  // namespace

double parse_timestamp(const QString& value) {
    const QString usage = app::tr("Use seconds, M:SS, or H:MM:SS.sss");
    const QStringList parts = value.trimmed().split(':');
    if (parts.size() > 3 || std::any_of(parts.begin(), parts.end(), [](const QString& part) {
            return part.trimmed().isEmpty();
        })) {
        throw std::invalid_argument(usage.toStdString());
    }
    std::vector<double> numbers;
    for (const QString& part : parts) {
        const std::optional<double> number = python_float(part.toStdString());
        if (!number) {
            throw std::invalid_argument(usage.toStdString());
        }
        numbers.push_back(*number);
    }
    if (std::any_of(numbers.begin(), numbers.end(), [](double number) { return number < 0; })) {
        throw std::invalid_argument(app::tr("Timestamp cannot be negative").toStdString());
    }
    if (std::any_of(numbers.begin(), numbers.end(), [](double number) { return !std::isfinite(number); })) {
        throw std::invalid_argument(app::tr("Timestamp must be a finite number").toStdString());
    }
    if (std::any_of(numbers.begin(), numbers.end() - 1, [](double number) { return number != std::floor(number); })) {
        throw std::invalid_argument(app::tr("Hours and minutes must be whole numbers").toStdString());
    }
    if (numbers.size() > 1 && numbers.back() >= 60) {
        throw std::invalid_argument(app::tr("Seconds must be below 60").toStdString());
    }
    if (numbers.size() > 2 && numbers[numbers.size() - 2] >= 60) {
        throw std::invalid_argument(app::tr("Minutes must be below 60").toStdString());
    }
    if (numbers.size() == 1) {
        return numbers[0];
    }
    if (numbers.size() == 2) {
        return numbers[0] * 60 + numbers[1];
    }
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2];
}

// ------------------------------------------------------------------ FrameView

FrameView::FrameView(Zoom& zoom, QWidget* parent) : QWidget(parent), zoom_(zoom) {
    setFocusPolicy(Qt::StrongFocus);
    label_ = new QLabel(app::tr(nothing_to_compare), this);
    label_->setAlignment(Qt::AlignCenter);
    label_->setWordWrap(true);
    label_->setStyleSheet("color: #ddd; background: transparent;");
    label_->setAttribute(Qt::WA_TransparentForMouseEvents);
}

void FrameView::set_message(const QString& message) {
    image_ = QImage();
    scaled_ = QPixmap();
    label_->setText(message);
    label_->show();
    zoom_changed();
}

void FrameView::set_image(const QImage& image) {
    image_ = image;
    scaled_ = QPixmap();
    label_->hide();
    zoom_changed();
}

QString FrameView::message() const { return label_->isVisibleTo(this) ? label_->text() : QString(); }

void FrameView::zoom_changed() {
    refresh_cursor();
    update();
}

bool FrameView::can_drag() const {
    return !image_.isNull() &&
           zoom_.can_drag({width(), height()}, {image_.width(), image_.height()}, devicePixelRatioF());
}

void FrameView::refresh_cursor() {
    // An open hand where the frame can be dragged; else the usual one.
    if (!drag_from_) {
        if (can_drag()) {
            setCursor(Qt::OpenHandCursor);
        } else {
            unsetCursor();
        }
    }
}

void FrameView::resizeEvent(QResizeEvent* event) {
    label_->setGeometry(rect());
    QWidget::resizeEvent(event);
    refresh_cursor();
}

void FrameView::mousePressEvent(QMouseEvent* event) {
    // Takes the keyboard focus, for the keys that switch and step, and
    // starts dragging a frame larger than the view.
    setFocus(Qt::MouseFocusReason);
    if (event->button() == Qt::LeftButton && can_drag()) {
        drag_from_ = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void FrameView::mouseMoveEvent(QMouseEvent* event) {
    if (!drag_from_) {
        QWidget::mouseMoveEvent(event);
        return;
    }
    const QPointF position = event->position();
    const QPointF moved = position - *drag_from_;
    drag_from_ = position;
    if (!image_.isNull() &&
        zoom_.drag(moved.x(), moved.y(), {width(), height()}, {image_.width(), image_.height()}, devicePixelRatioF())) {
        update();
    }
    event->accept();
}

void FrameView::mouseReleaseEvent(QMouseEvent* event) {
    if (drag_from_ && event->button() == Qt::LeftButton) {
        drag_from_.reset();
        refresh_cursor();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void FrameView::paintEvent(QPaintEvent* /*event*/) {
    // The frame where the zoom puts it, scaled once per size.
    QPainter painter(this);
    painter.fillRect(rect(), QColor("#171717"));
    if (!image_.isNull()) {
        const double ratio = devicePixelRatioF();
        const Placement place = zoom_.placement({width(), height()}, {image_.width(), image_.height()}, ratio);
        const QSize size(static_cast<int>(std::lround(place.width * ratio)),
                         static_cast<int>(std::lround(place.height * ratio)));
        if (size.width() < image_.width()) {
            // Smaller than its own pixels, as when fitted: scaled smoothly.
            if (scaled_.isNull() || scaled_.size() != size) {
                scaled_ = QPixmap::fromImage(image_.scaled(size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
                scaled_.setDevicePixelRatio(ratio);
            }
            painter.drawPixmap(QPointF(place.x, place.y), scaled_);
        } else {
            // Its own size or larger: each pixel a sharp square, to look at.
            painter.drawImage(QRectF(place.x, place.y, place.width, place.height), image_);
        }
    }
    painter.setPen(QColor("#444"));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
}

// --------------------------------------------------------------- the workers

FrameExtractWorker::FrameExtractWorker(int generation, std::string ffmpeg, FrameComparison comparison,
                                       long long frame, std::vector<FrameSide> sides, PreviewColorSettings colors,
                                       QObject* parent)
    : QThread(parent),
      generation_(generation),
      ffmpeg_(std::move(ffmpeg)),
      comparison_(std::move(comparison)),
      frame_(frame),
      sides_(std::move(sides)),
      colors_(colors) {}

void FrameExtractWorker::run() {
    for (const FrameSide side : sides_) {
        if (cancel_.cancelled()) {
            return;
        }
        std::string png;
        try {
            png = extract_frame_png(ffmpeg_, comparison_, side, frame_, cancel_, colors_);
        } catch (const Cancelled&) {
            return;
        } catch (const std::exception& error) {
            if (!cancel_.cancelled()) {
                emit frame_failed(generation_, static_cast<int>(side), QString::fromUtf8(error.what()));
            }
            continue;
        }
        if (!cancel_.cancelled()) {
            emit frame_ready(generation_, static_cast<int>(side),
                             QImage::fromData(reinterpret_cast<const uchar*>(png.data()),
                                              static_cast<int>(png.size()), "PNG"));
        }
    }
}

CropDetectWorker::CropDetectWorker(std::string ffmpeg, VideoInfo source, VideoInfo distorted, Known known_source,
                                   Known known_distorted, QObject* parent)
    : QThread(parent),
      ffmpeg_(std::move(ffmpeg)),
      source_(std::move(source)),
      distorted_(std::move(distorted)),
      known_source_(std::move(known_source)),
      known_distorted_(std::move(known_distorted)) {}

std::optional<CropBox> CropDetectWorker::detect(const VideoInfo& info) {
    try {
        return detect_crop(ffmpeg_, info, cancel_, std::nullopt);
    } catch (const Cancelled&) {
        throw;
    } catch (const std::exception&) {
        // The preview stays usable when a file has no steady bars or cannot
        // be sampled: a run decides for good.
        return std::nullopt;
    }
}

void CropDetectWorker::run() {
    try {
        const bool same_file = distorted_.path == source_.path;
        DetectedCrops crops;
        if (!known_source_ && !known_distorted_ && !same_file) {
            // Both at once: two decoders, in about half the time.
            std::exception_ptr failure;
            std::thread other([&] {
                try {
                    crops.distorted = detect(distorted_);
                } catch (...) {
                    failure = std::current_exception();
                }
            });
            try {
                crops.source = detect(source_);
            } catch (...) {
                other.join();
                throw;
            }
            other.join();
            if (failure) {
                std::rethrow_exception(failure);
            }
        } else {
            crops.source = known_source_ ? *known_source_ : detect(source_);
            crops.distorted = same_file ? crops.source : known_distorted_ ? *known_distorted_ : detect(distorted_);
        }
        if (!same_file) {
            // The boxes a run would compare (common_picture).
            std::tie(crops.source, crops.distorted) =
                common_picture(source_, distorted_, crops.source, crops.distorted);
        }
        if (!cancel_.cancelled()) {
            emit ready(crops);
        }
    } catch (const Cancelled&) {
    }
}

// ----------------------------------------------------------------- the panel

FrameComparePanel::FrameComparePanel(std::function<std::string()> ffmpeg, PreviewColorMode color_mode,
                                     int decoded_videos, QWidget* parent)
    : QWidget(parent), ffmpeg_(std::move(ffmpeg)), decoded_videos_(std::max(1, decoded_videos)), color_mode_(color_mode) {
    qRegisterMetaType<DetectedCrops>();
    auto* root = new QVBoxLayout(this);

    auto* top = new QHBoxLayout;
    previous_video_button = new QPushButton(QString::fromUtf8("\xE2\x86\x90"));  // ←
    previous_video_button->setToolTip(app::tr("Previous test video (Left arrow)"));
    connect(previous_video_button, &QPushButton::clicked, this, [this] { cycle_distorted(-1); });
    video_combo = new QComboBox;
    video_combo->setMinimumContentsLength(30);
    connect(video_combo, &QComboBox::currentIndexChanged, this, [this](int index) { on_video_selected(index); });
    next_video_button = new QPushButton(QString::fromUtf8("\xE2\x86\x92"));  // →
    next_video_button->setToolTip(app::tr("Next test video (Right arrow)"));
    connect(next_video_button, &QPushButton::clicked, this, [this] { cycle_distorted(1); });
    // One zoom for the still frames and playback alike: switching between
    // them, or between the source and an encode, keeps the same view.
    zoom_combo = new QComboBox;
    zoom_combo->setEditable(true);
    zoom_combo->setInsertPolicy(QComboBox::NoInsert);
    zoom_combo->addItem(app::tr("Fit to window"), QVariant());
    for (const double factor : {0.5, 1.0, 2.0, 4.0}) {
        zoom_combo->addItem(percent_text(factor), factor);
    }
    zoom_combo->setToolTip(app::tr("Fit to window, or a zoom you choose or type in: at 100% one pixel of the video "
                                   "is one pixel of the screen. Drag a zoomed frame to move it."));
    connect(zoom_combo, &QComboBox::activated, this, [this](int index) { on_zoom_chosen(index); });
    connect(zoom_combo->lineEdit(), &QLineEdit::editingFinished, this, [this] { on_zoom_typed(); });
    top->addWidget(new QLabel(app::tr("Test video:")));
    top->addWidget(previous_video_button);
    top->addWidget(video_combo, 1);
    top->addWidget(next_video_button);
    top->addSpacing(12);
    top->addWidget(new QLabel(app::tr("Zoom:")));
    top->addWidget(zoom_combo);
    root->addLayout(top);

    auto* color_row = new QHBoxLayout;
    color_mode_combo = new QComboBox;
    color_mode_combo->addItem(app::tr("Display-aware (recommended)"), "display_aware");
    color_mode_combo->addItem(app::tr("HDR \xE2\x86\x92 SDR (100 nit; assume PQ if untagged)"), "hdr_to_sdr");
    color_mode_combo->addItem(app::tr("Unmanaged (diagnostic)"), "unmanaged");
    color_mode_combo->setCurrentIndex(
        std::max(0, color_mode_combo->findData(text_of(std::string(preview_color_mode_value(color_mode_))))));
    color_mode_combo->setToolTip(app::tr("Display-aware detects PQ/HLG tags and uses the Windows monitor's "
                                         "SDR-white setting. The fixed mode can recover an untagged PQ file."));
    connect(color_mode_combo, &QComboBox::currentIndexChanged, this, [this](int index) { on_color_mode_changed(index); });
    color_status_label = new QLabel;
    theme::style(color_status_label, "color: {muted};");
    color_row->addWidget(new QLabel(app::tr("HDR preview:")));
    color_row->addWidget(color_mode_combo);
    source_resolution_combo = new QComboBox;
    source_resolution_combo->addItem(app::tr("Native resolution"), true);
    source_resolution_combo->addItem(app::tr("Downscale to test resolution"), false);
    source_resolution_combo->setToolTip(app::tr(
        "Playback only. Keep the cropped source at native resolution, or downscale "
        "it to fit the selected encode's cropped resolution. Both fit the window. "
        "Changing this may briefly buffer. Metric calculations are unaffected."));
    source_resolution_combo->setEnabled(false);
    connect(source_resolution_combo, &QComboBox::currentIndexChanged, this, [this](int /*index*/) {
        if (is_video_mode()) {
            load_current_video();
        }
    });
    color_row->addWidget(new QLabel(app::tr("Reference playback:")));
    color_row->addWidget(source_resolution_combo);
    // This row would set the window's minimum width: a drop-down is never
    // narrower than its longest choice by default, and in Spanish, with a
    // wide system font, the two would push the window past the 1280 pixels
    // it opens at. So they may be squeezed (their lists still open at full
    // width), and keep their size while there is room.
    for (QComboBox* combo : {color_mode_combo, source_resolution_combo}) {
        const int full = combo->sizeHint().width();
        combo->setMinimumWidth(std::min(full, 170));
        combo->view()->setMinimumWidth(full);
    }
    color_row->addWidget(color_status_label, 1);
    advanced_info_button = new QPushButton(app::tr("Advanced info"));
    advanced_info_button->setToolTip(app::tr("Playback details will appear here when a video is loaded."));
    color_row->addWidget(advanced_info_button);
    root->addLayout(color_row);

    showing_label = new QLabel(app::tr("No videos loaded"));
    QFont font = showing_label->font();
    font.setBold(true);
    font.setPointSize(font.pointSize() + 1);
    showing_label->setFont(font);
    showing_label->setAlignment(Qt::AlignCenter);
    root->addWidget(showing_label);

    viewer = new FrameView(zoom_);
    content_stack_ = new QStackedWidget;
    content_stack_->addWidget(viewer);
    root->addWidget(content_stack_, 1);

    auto* seek = new QHBoxLayout;
    play_button = new QPushButton(app::tr("\xE2\x96\xB6 Play"));
    play_button->setFont(symbol_font(play_button->font()));
    connect(play_button, &QPushButton::clicked, this, [this] { on_play_clicked(); });
    audio_checkbox = new QCheckBox(app::tr("Audio"));
    audio_checkbox->setToolTip(app::tr(
        "Native playback plays one source soundtrack continuously across S and encode switches. "
        "Missing/unsupported source audio stays silent. The FFmpeg fallback plays the "
        "source's soundtrack too. Video buffering also pauses audio."));
    audio_checkbox->setChecked(true);
    connect(audio_checkbox, &QCheckBox::toggled, this, [this](bool enabled) {
        if (video_view != nullptr) {
            video_view->set_audio_enabled(enabled);
        }
    });
    previous_frame_button = new QPushButton(app::tr("\xE2\x88\x92 Frame"));
    connect(previous_frame_button, &QPushButton::clicked, this, [this] { set_frame(frame_ - 1); });
    frame_spin = new QSpinBox;
    frame_spin->setRange(0, 0);
    frame_spin->setKeyboardTracking(false);
    connect(frame_spin, &QSpinBox::valueChanged, this, [this](int value) { set_frame(value); });
    timestamp_edit = new QLineEdit("0:00:00.000");
    timestamp_edit->setMaximumWidth(120);
    timestamp_edit->setToolTip(app::tr("Enter seconds, M:SS, or H:MM:SS.sss"));
    connect(timestamp_edit, &QLineEdit::editingFinished, this, [this] { on_timestamp_committed(); });
    next_frame_button = new QPushButton(app::tr("+ Frame"));
    connect(next_frame_button, &QPushButton::clicked, this, [this] { set_frame(frame_ + 1); });
    seek->addWidget(play_button);
    seek->addWidget(audio_checkbox);
    seek->addSpacing(12);
    seek->addWidget(previous_frame_button);
    seek->addWidget(new QLabel(app::tr("Frame:")));
    seek->addWidget(frame_spin);
    seek->addWidget(new QLabel(app::tr("Timestamp:")));
    seek->addWidget(timestamp_edit);
    seek->addWidget(next_frame_button);
    root->addLayout(seek);

    timeline = new QSlider(Qt::Horizontal);
    timeline->setRange(0, 0);
    connect(timeline, &QSlider::valueChanged, this, [this](int value) { set_frame(value); });
    root->addWidget(timeline);

    // Elided, so that the text's width is not the page's (see
    // QuietTabWidget).
    detail_label = new ElidedLabel;
    detail_label->setAlignment(Qt::AlignCenter);
    detail_label->set_text(app::tr("No frame selected."));
    root->addWidget(detail_label);
    guide_label = new QLabel(app::tr("Hold S: show source   \xC2\xB7   \xE2\x86\x90/\xE2\x86\x92: switch test video   "
                                     "\xC2\xB7   Space: play/pause"));
    guide_label->setAlignment(Qt::AlignCenter);
    theme::style(guide_label, "color: {muted};");
    guide_label->setWordWrap(true);
    root->addWidget(guide_label);

    seek_timer_ = new QTimer(this);
    seek_timer_->setSingleShot(true);
    seek_timer_->setInterval(120);
    connect(seek_timer_, &QTimer::timeout, this, [this] { request_current_frames(); });
    display_timer_ = new QTimer(this);
    display_timer_->setSingleShot(true);
    display_timer_->setInterval(200);
    connect(display_timer_, &QTimer::timeout, this, [this] { on_display_maybe_changed(); });
    // While playing, the seek widgets and the frame line follow the frames
    // ten times a second, not at each.
    position_timer_ = new QTimer(this);
    position_timer_->setSingleShot(true);
    position_timer_->setInterval(100);
    connect(position_timer_, &QTimer::timeout, this, [this] { show_video_position(); });

    // This panel's own widgets' keys, rather than the whole application's:
    // a filter on the application would keep every window closed alive.
    installEventFilter(this);
    for (QWidget* child : findChildren<QWidget*>()) {
        child->installEventFilter(this);
    }
    update_enabled_state();
}

FrameComparePanel::~FrameComparePanel() {
    cancel();
    for (QThread* worker : live_workers()) {
        worker->wait();
    }
}

// ------------------------------------------------------------- for the window

void FrameComparePanel::set_runs(std::vector<FrameComparisonEntry> entries) {
    const FrameComparisonEntry* old_entry = current_entry();
    const std::optional<std::string> old_identity =
        old_entry != nullptr ? std::optional(old_entry->identity) : std::nullopt;
    const std::optional<std::string> old_row =
        old_entry != nullptr && !old_entry->row.empty() ? std::optional(old_entry->row) : std::nullopt;
    const std::optional<FrameComparison> old_comparison =
        old_entry != nullptr ? std::optional(old_entry->comparison) : std::nullopt;
    // The entry chosen before: its row's, whatever scores it has now.
    const auto was_shown = [&](const FrameComparisonEntry& entry) {
        return old_row ? entry.row == *old_row : old_identity && entry.identity == *old_identity;
    };
    entries_.clear();
    for (const FrameComparisonEntry& entry : entries) {
        entries_.push_back(apply_auto_crop(entry));
    }
    ++generation_;
    cancel_workers();
    // A file that was missing or unreadable may be back since: tried again.
    errors_.clear();

    std::size_t current = 0;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (was_shown(entries_[index])) {
            current = index;
            break;
        }
    }
    current_index_ = entries_.empty() ? 0 : std::min(current, entries_.size() - 1);
    {
        const QSignalBlocker blocker(video_combo);
        video_combo->clear();
        for (const FrameComparisonEntry& entry : entries_) {
            video_combo->addItem(entry.label);
        }
        if (!entries_.empty()) {
            video_combo->setCurrentIndex(static_cast<int>(current_index_));
        }
    }
    long long shortest = 1;
    if (!entries_.empty()) {
        shortest = std::max(1LL, entries_.front().comparison.frame_count);
        for (const FrameComparisonEntry& entry : entries_) {
            shortest = std::min(shortest, std::max(1LL, entry.comparison.frame_count));
        }
    }
    const long long maximum = shortest - 1;
    frame_ = std::min(frame_, maximum);
    set_ranges(maximum);
    update_enabled_state();
    update_labels();
    if (entries_.empty()) {
        viewer->set_message(app::tr(nothing_to_compare));
        if (video_view != nullptr) {
            video_view->clear();
        }
    } else if (isVisible()) {
        if (is_video_mode()) {
            // Only when what is on screen changed: a result arrives for a
            // video several times while a run goes on, which must not
            // restart the video being watched.
            const FrameComparisonEntry* shown = current_entry();
            if (!was_shown(*shown) || shown->comparison != *old_comparison) {
                load_current_video();
            }
        } else {
            show_or_request();
        }
    }
}

const FrameComparisonEntry* FrameComparePanel::current_entry() const {
    return entries_.empty() ? nullptr : &entries_[current_index_];
}

void FrameComparePanel::set_frame(long long frame) {
    if (entries_.empty()) {
        return;
    }
    const long long bounded = std::clamp(frame, 0LL, static_cast<long long>(frame_spin->maximum()));
    const bool changed = bounded != frame_;
    frame_ = bounded;
    sync_seek_widgets();
    update_labels();
    if (is_video_mode()) {
        // Playback seeks there, unless it is playback moving the widgets.
        const double fps = current_entry()->comparison.fps;
        if (!syncing_video_position_ && video_view != nullptr && fps > 0) {
            video_view->set_position(static_cast<int>(std::nearbyint(static_cast<double>(frame_) / fps * 1000)));
        }
    } else if (changed || current_image() == nullptr) {
        ++generation_;
        seek_timer_->start();
    }
}

void FrameComparePanel::cycle_distorted(int direction) {
    if (entries_.size() < 2) {
        return;
    }
    // Around the ends either way, as Python's % wraps a negative number.
    const auto count = static_cast<long long>(entries_.size());
    const long long next = (static_cast<long long>(current_index_) + direction) % count;
    current_index_ = static_cast<std::size_t>((next + count) % count);
    {
        const QSignalBlocker blocker(video_combo);
        video_combo->setCurrentIndex(static_cast<int>(current_index_));
    }
    ++generation_;
    update_labels();
    if (is_video_mode()) {
        load_current_video();
    } else {
        show_or_request();
    }
}

void FrameComparePanel::set_decoded_videos(int count) {
    decoded_videos_ = std::max(1, count);
    if (video_view != nullptr) {
        video_view->set_decoded_videos(decoded_videos_);
    }
}

std::vector<QThread*> FrameComparePanel::live_workers() const {
    std::vector<QThread*> live;
    for (FrameExtractWorker* worker : workers_) {
        if (worker->isRunning()) {
            live.push_back(worker);
        }
    }
    for (const auto& [identity, worker] : crop_workers_) {
        if (worker->isRunning()) {
            live.push_back(worker);
        }
    }
    if (video_view != nullptr) {
        for (QThread* worker : video_view->live_workers()) {
            live.push_back(worker);
        }
    }
    return live;
}

void FrameComparePanel::cancel() {
    seek_timer_->stop();
    ++generation_;
    cancel_workers();
    cancel_crop_workers();
    if (video_view != nullptr) {
        video_view->clear();
    }
}

// ------------------------------------------------------------------ lifecycle

void FrameComparePanel::showEvent(QShowEvent* event) {
    // Watches the window's keys and moves, and shows the current pair.
    QWidget::showEvent(event);
    if (!window_filters_installed_) {
        QWidget* top = window();
        top->installEventFilter(this);
        for (QWidget* child : top->findChildren<QWidget*>()) {
            child->installEventFilter(this);
        }
        window_filters_installed_ = true;
    }
    refresh_display_hdr();
    if (!entries_.empty()) {
        if (is_video_mode()) {
            load_current_video();
        } else {
            show_or_request();
        }
    }
}

void FrameComparePanel::hideEvent(QHideEvent* event) {
    // Stops the workers, and pauses playback.
    seek_timer_->stop();
    display_timer_->stop();
    ++generation_;
    cancel_workers();
    cancel_crop_workers();
    if (video_view != nullptr) {
        video_view->set_playing(false);
    }
    showing_source_ = false;
    update_labels();
    QWidget::hideEvent(event);
}

bool FrameComparePanel::eventFilter(QObject* watched, QEvent* event) {
    // The tab's keys anywhere in the window (S, Space, Left and Right), and
    // the window's moves between displays.
    if (watched == advanced_info_button && event->type() == QEvent::ToolTip) {
        QToolTip::showText(advanced_info_button->mapToGlobal(advanced_info_button->rect().bottomLeft()),
                           advanced_info_button->toolTip(), advanced_info_button, advanced_info_button->rect(), 30000);
        return true;
    }
    if (!isVisible()) {
        return QWidget::eventFilter(watched, event);
    }
    const QEvent::Type type = event->type();
    if (watched == window() && (type == QEvent::Move || type == QEvent::ScreenChangeInternal)) {
        // Asked once Windows has finished the move.
        display_timer_->start();
    }
    const bool key_event = type == QEvent::KeyPress || type == QEvent::KeyRelease;
    const auto* key = key_event ? static_cast<QKeyEvent*>(event) : nullptr;
    if (key != nullptr && key->key() == Qt::Key_S && key->modifiers() == Qt::NoModifier) {
        if (!key->isAutoRepeat()) {
            showing_source_ = type == QEvent::KeyPress;
            update_labels();
            show_or_request();
        }
        return true;
    }
    if (key != nullptr && type == QEvent::KeyPress && key->key() == Qt::Key_Space &&
        key->modifiers() == Qt::NoModifier) {
        QWidget* focus = QApplication::focusWidget();
        if (qobject_cast<QLineEdit*>(focus) == nullptr && qobject_cast<QAbstractSpinBox*>(focus) == nullptr &&
            qobject_cast<QComboBox*>(focus) == nullptr) {
            if (!key->isAutoRepeat()) {
                on_play_clicked();
            }
            return true;
        }
    }
    if (key != nullptr && type == QEvent::KeyPress && (key->key() == Qt::Key_Left || key->key() == Qt::Key_Right)) {
        QWidget* focus = QApplication::focusWidget();
        if (qobject_cast<QLineEdit*>(focus) != nullptr || qobject_cast<QAbstractSpinBox*>(focus) != nullptr ||
            qobject_cast<QSlider*>(focus) != nullptr || qobject_cast<QComboBox*>(focus) != nullptr) {
            return QWidget::eventFilter(watched, event);
        }
        if (!key->isAutoRepeat() && key->modifiers() == Qt::NoModifier) {
            cycle_distorted(key->key() == Qt::Key_Left ? -1 : 1);
        }
        return true;
    }
    if ((type == QEvent::WindowDeactivate || type == QEvent::Hide) && showing_source_) {
        showing_source_ = false;
        update_labels();
        show_or_request();
    }
    return QWidget::eventFilter(watched, event);
}

// ------------------------------------------------------------------- playback

bool FrameComparePanel::is_video_mode() const {
    // A pair that can play is always played (its paused frames too); still
    // frames are for the rest: resolution tests, missing files, raw streams.
    const FrameComparisonEntry* entry = current_entry();
    return entry != nullptr && VideoCompareView::can_play(entry->comparison).first;
}

VideoCompareView& FrameComparePanel::ensure_video_view() {
    // Made the first time a pair plays.
    if (video_view != nullptr) {
        return *video_view;
    }
    auto* view = new VideoCompareView([this] { return ffmpeg_(); }, zoom_);
    view->set_decoded_videos(decoded_videos_);
    connect(view, &VideoCompareView::position_changed, this, [this](int position) { on_video_position_changed(position); });
    connect(view, &VideoCompareView::playing_changed, this, [this](bool playing) { on_video_playing_changed(playing); });
    connect(view, &VideoCompareView::status_changed, this,
            [this](const QString& status) { on_video_status_changed(status); });
    view->set_audio_enabled(audio_checkbox->isChecked());
    content_stack_->addWidget(view);
    // The tab's keys reach the panel from the view too.
    view->installEventFilter(this);
    for (QWidget* child : view->findChildren<QWidget*>()) {
        child->installEventFilter(this);
    }
    video_view = view;
    return *view;
}

void FrameComparePanel::load_current_video(std::optional<bool> playing) {
    // Plays the pair shown, playing or paused as before unless `playing`.
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr) {
        if (video_view != nullptr) {
            video_view->clear();
        }
        return;
    }
    VideoCompareView& view = ensure_video_view();
    // Black bars already known for its files give a cropped copy.
    ensure_auto_crop(entry);
    entry = current_entry();
    seek_timer_->stop();
    cancel_workers();
    content_stack_->setCurrentWidget(&view);
    const bool play = playing.value_or(view.is_playing() || view.playback_requested());
    const double fps = entry->comparison.fps;
    const int position_ms = fps > 0 ? static_cast<int>(std::nearbyint(static_cast<double>(frame_) / fps * 1000)) : 0;
    std::vector<FrameComparison> series;
    for (const FrameComparisonEntry& item : entries_) {
        if (VideoCompareView::can_play(item.comparison).first) {
            series.push_back(item.comparison);
        }
    }
    if (!view.load(entry->comparison, position_ms, play, color_settings(), series,
                   source_resolution_combo->currentData().toBool())) {
        play_button->setText(app::tr("\xE2\x96\xB6 Play"));
    }
    view.show_source(showing_source_);
    update_enabled_state();
}

void FrameComparePanel::on_play_clicked() {
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr) {
        return;
    }
    VideoCompareView& view = ensure_video_view();
    if (const auto [available, reason] = VideoCompareView::can_play(entry->comparison); !available) {
        on_video_status_changed(text_of(reason));
        return;
    }
    view.set_playing(!view.playback_requested());
}

void FrameComparePanel::on_video_position_changed(int position_ms) {
    // The seek widgets follow playback: at once when paused, ten times a
    // second while playing.
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr || !is_video_mode() || entry->comparison.fps <= 0) {
        return;
    }
    const auto frame = static_cast<long long>(std::nearbyint(position_ms / 1000.0 * entry->comparison.fps));
    const long long bounded = std::clamp(frame, 0LL, static_cast<long long>(frame_spin->maximum()));
    if (bounded == frame_) {
        return;
    }
    frame_ = bounded;
    if (video_view != nullptr && video_view->is_playing()) {
        if (!position_timer_->isActive()) {
            position_timer_->start();
        }
        return;
    }
    show_video_position();
}

void FrameComparePanel::show_video_position() {
    // The frame in the seek widgets and the frame line, and only there:
    // what is shown and its colours are the same at every frame.
    position_timer_->stop();
    syncing_video_position_ = true;
    sync_seek_widgets();
    update_frame_line();
    syncing_video_position_ = false;
}

void FrameComparePanel::on_video_playing_changed(bool playing) {
    play_button->setText(playing ? app::tr("\xE2\x9D\x9A\xE2\x9D\x9A Pause") : app::tr("\xE2\x96\xB6 Play"));
    // Said before the frame line is drawn: drawn first, it went on saying
    // "Playing" once playback had stopped.
    video_status_ = playing ? QStringLiteral("Playing") : QStringLiteral("Paused");
    if (!playing && position_timer_->isActive()) {
        show_video_position();  // the frame it stopped on, now
    } else if (is_video_mode()) {
        update_frame_line();
    }
    if (is_video_mode()) {
        update_color_status();
    }
}

void FrameComparePanel::on_video_status_changed(const QString& message) {
    // Playback's status: in full in Advanced info, short in the frame line.
    video_status_ = message;
    QStringList lines;
    for (const QString& part : QString(message).replace("distorted", "test").split(QString::fromUtf8(" \xC2\xB7 "))) {
        lines.append(tr_message(part).toHtmlEscaped());
    }
    const QString tooltip = "<p>" + lines.join("<br>") + "</p>";
    if (advanced_info_button->toolTip() != tooltip) {
        advanced_info_button->setToolTip(tooltip);
    }
    if (is_video_mode()) {
        update_color_status();
        // Drawn only as the frame moved, the line kept the status from
        // before: "Playing" at the end, the last test video's decoder.
        update_frame_line();
    }
}

QString FrameComparePanel::concise_playback_status() const {
    // Playback's status, short and translated, the details left to Advanced
    // info. The view's messages are in English, read here for their words.
    const QString message = QString(video_status_).replace("distorted", "test");
    const QString separator = QString::fromUtf8(" \xC2\xB7 ");
    // Errors stay visible. A reference without a soundtrack ("audio
    // unavailable") is not one.
    const QString problem = message.toLower().replace("audio unavailable", "");
    if (problem.contains("failed") || problem.contains("could not") || problem.contains("error") ||
        problem.contains("unavailable")) {
        QStringList parts;
        for (const QString& part : message.split(separator)) {
            parts.append(tr_message(part));
        }
        return parts.join(separator);
    }
    QStringList parts{tr_message(message.section(separator, 0, 0))};
    for (const auto& [side, shown] : {std::pair("source", N_("Source")), std::pair("test", N_("Test"))}) {
        const QRegularExpression decoder(QStringLiteral("\\b%1 (GPU|software|cpu|cuda|d3d11va|qsv|vaapi)\\b").arg(side),
                                         QRegularExpression::CaseInsensitiveOption);
        if (const QRegularExpressionMatch match = decoder.match(message); match.hasMatch()) {
            const QString used = match.captured(1).toLower();
            const QString mode = used == "software" || used == "cpu" ? QStringLiteral("CPU") : QStringLiteral("GPU");
            parts.append(app::tr("{side}: {mode} decode", {{"side", app::tr(shown)}, {"mode", mode}}));
        }
    }
    return parts.join(separator);
}

// ------------------------------------------------------------------- controls

void FrameComparePanel::on_video_selected(int index) {
    if (index < 0 || static_cast<std::size_t>(index) >= entries_.size() ||
        static_cast<std::size_t>(index) == current_index_) {
        return;
    }
    current_index_ = static_cast<std::size_t>(index);
    ++generation_;
    update_labels();
    if (is_video_mode()) {
        load_current_video();
    } else {
        show_or_request();
    }
}

void FrameComparePanel::on_zoom_chosen(int index) {
    const QVariant chosen = zoom_combo->itemData(index);
    apply_zoom(chosen.isValid() ? std::optional(chosen.toDouble()) : std::nullopt);
}

void FrameComparePanel::on_zoom_typed() {
    const QString text = zoom_combo->currentText();
    if (text.trimmed() == app::tr("Fit to window")) {
        apply_zoom(std::nullopt);
        return;
    }
    const std::optional<double> factor = parse_percent(text);
    if (!factor) {
        show_zoom();  // not a zoom: the one in force again
    } else {
        apply_zoom(factor);
    }
}

void FrameComparePanel::apply_zoom(std::optional<double> factor) {
    // Fitted (nothing) or at `factor`, around the middle of the view.
    const bool changed = factor != zoom_.factor;
    zoom_.factor = factor;
    show_zoom();
    if (changed) {
        viewer->zoom_changed();
        if (video_view != nullptr) {
            video_view->zoom_changed();
        }
    }
}

void FrameComparePanel::show_zoom() {
    int index = -1;
    for (int item = 0; item < zoom_combo->count(); ++item) {
        const QVariant listed = zoom_combo->itemData(item);
        if (zoom_.factor ? listed.isValid() && listed.toDouble() == *zoom_.factor : !listed.isValid()) {
            index = item;
            break;
        }
    }
    if (index >= 0) {
        zoom_combo->setCurrentIndex(index);
    }
    zoom_combo->setEditText(zoom_.factor ? percent_text(*zoom_.factor) : app::tr("Fit to window"));
}

void FrameComparePanel::on_color_mode_changed(int index) {
    const PreviewColorMode mode = preview_color_mode_from(color_mode_combo->itemData(index).toString().toStdString());
    if (mode == color_mode_) {
        return;
    }
    color_mode_ = mode;
    ++generation_;
    cancel_workers();
    update_labels();
    emit color_mode_changed(text_of(std::string(preview_color_mode_value(mode))));
    show_or_request();
}

void FrameComparePanel::on_timestamp_committed() {
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr) {
        return;
    }
    double seconds = 0;
    try {
        seconds = parse_timestamp(timestamp_edit->text());
    } catch (const std::invalid_argument& error) {
        theme::style(timestamp_edit, "border: 1px solid {error_border};");
        timestamp_edit->setToolTip(QString::fromUtf8(error.what()));
        return;
    }
    theme::style(timestamp_edit, "");
    timestamp_edit->setToolTip(app::tr("Enter seconds, M:SS, or H:MM:SS.sss"));
    // As Python's round(): a half to the even frame. Past the last frame is
    // the last frame (and, cast, would not be a number at all).
    const double frame = std::nearbyint(seconds * entry->comparison.fps);
    set_frame(static_cast<long long>(std::min(frame, static_cast<double>(frame_spin->maximum()))));
}

void FrameComparePanel::set_ranges(long long maximum) {
    const QSignalBlocker spin_blocker(frame_spin);
    const QSignalBlocker slider_blocker(timeline);
    frame_spin->setRange(0, static_cast<int>(maximum));
    frame_spin->setValue(static_cast<int>(frame_));
    timeline->setRange(0, static_cast<int>(maximum));
    timeline->setValue(static_cast<int>(frame_));
}

void FrameComparePanel::sync_seek_widgets() {
    {
        const QSignalBlocker spin_blocker(frame_spin);
        const QSignalBlocker slider_blocker(timeline);
        frame_spin->setValue(static_cast<int>(frame_));
        timeline->setValue(static_cast<int>(frame_));
    }
    const FrameComparisonEntry* entry = current_entry();
    const double seconds =
        entry != nullptr && entry->comparison.fps > 0 ? static_cast<double>(frame_) / entry->comparison.fps : 0;
    const QSignalBlocker blocker(timestamp_edit);
    timestamp_edit->setText(text_of(format_hms(seconds, 3)));
}

void FrameComparePanel::update_enabled_state() {
    const bool available = !entries_.empty();
    for (QWidget* widget : std::initializer_list<QWidget*>{video_combo, frame_spin, timestamp_edit, timeline,
                                                          previous_frame_button, next_frame_button}) {
        widget->setEnabled(available);
    }
    const bool several = entries_.size() > 1;
    previous_video_button->setEnabled(several);
    next_video_button->setEnabled(several);
    const bool playable = is_video_mode();
    play_button->setEnabled(available && playable);
    audio_checkbox->setEnabled(available && playable);
    source_resolution_combo->setEnabled(playable);
    color_mode_combo->setEnabled(available);
}

void FrameComparePanel::update_labels() {
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr) {
        showing_label->setText(app::tr("No videos to compare"));
        detail_label->set_text(app::tr("No frame selected."));
        color_status_label->setText(app::tr("No video selected."));
        return;
    }
    const FrameComparison& comparison = entry->comparison;
    const QString side =
        showing_source_ ? app::tr("SOURCE")
                        : app::tr("{side} {number} of {count}",
                                  {{"side", app::tr("TEST")}, {"number", current_index_ + 1}, {"count", entries_.size()}});
    const QString name = showing_source_ ? text_of(platform::path_text(comparison.source_info.path.filename()))
                                         : entry->label;
    showing_label->setText(side + QString::fromUtf8(" \xE2\x80\x94 ") + name);
    update_frame_line();
    update_color_status();
}

void FrameComparePanel::update_frame_line() {
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr) {
        return;
    }
    const FrameComparison& comparison = entry->comparison;
    const double seconds = comparison.fps > 0 ? static_cast<double>(frame_) / comparison.fps : 0;
    QStringList parts{app::tr("Frame {frame:,}", {{"frame", frame_}}), text_of(format_hms(seconds, 3)),
                      score_text(*entry)};
    if (is_video_mode()) {
        parts.append(concise_playback_status());
    }
    if (comparison.auto_crop_pending) {
        // Said, rather than showing frames that look cropped and are not:
        // auto black bars are measured during a run, and until then these
        // are the raw frames.
        parts.append(crop_workers_.contains(pair_key(*entry))
                         ? app::tr("detecting black bars\xE2\x80\xA6")
                         : app::tr("black bars not detected yet \xE2\x80\x94 shown uncropped"));
    }
    detail_label->set_text(parts.join(QString::fromUtf8("   \xC2\xB7   ")));
}

QString FrameComparePanel::score_text(const FrameComparisonEntry& entry) const {
    // What this frame scored, or why there is no number to show.
    if (!entry.scores) {
        return app::tr("No metric results loaded");
    }
    QStringList values;
    for (const MetricDefinition& metric : metrics()) {
        if (metric.kind != MetricKind::frame) {
            continue;
        }
        const FrameMetricResult* scores = entry.scores->metric_results.frame(std::string(metric.key));
        if (scores == nullptr) {
            continue;
        }
        const auto found = std::lower_bound(scores->frame.begin(), scores->frame.end(), frame_);
        if (found == scores->frame.end() || *found != frame_) {
            continue;
        }
        const double value = scores->values[static_cast<std::size_t>(found - scores->frame.begin())];
        if (std::isnan(value)) {
            continue;
        }
        values.append(text_of(std::string(metric.label) + " " + metric.format_value(value) +
                              std::string(metric.value_suffix)));
    }
    if (!values.isEmpty()) {
        return values.join(QString::fromUtf8(" \xC2\xB7 "));
    }
    return app::tr("Metrics not calculated for this frame");
}

PreviewColorSettings FrameComparePanel::color_settings() const {
    return {color_mode_, display_hdr_.hdr_enabled, display_hdr_.sdr_white_nits};
}

bool FrameComparePanel::refresh_display_hdr() {
    // The display's HDR state again; whether it changed.
    const platform::DisplayHdrInfo current =
        platform::query_display_hdr(reinterpret_cast<void*>(window()->winId()));
    if (current == display_hdr_) {
        return false;
    }
    display_hdr_ = current;
    ++generation_;
    cancel_workers();
    update_labels();
    return true;
}

void FrameComparePanel::on_display_maybe_changed() {
    if (isVisible() && refresh_display_hdr()) {
        show_or_request();
    }
}

void FrameComparePanel::update_color_status() {
    // How the frame's colours are shown.
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr) {
        return;
    }
    const VideoInfo& info = frame_video_info(entry->comparison, shown_side());
    const std::optional<std::string> kind = hdr_kind(info);
    const bool video = is_video_mode();
    // FFmpeg's playback (the fallback) is an SDR preview, never native.
    const bool ffmpeg_playback = video_view != nullptr && video_view->plays_with_ffmpeg();
    const QString prefix = video ? app::tr("Video playback \xC2\xB7 ") : QString();
    if (color_mode_ == PreviewColorMode::unmanaged) {
        color_status_label->setText(app::tr("{prefix}{kind} input \xC2\xB7 tone mapping off",
                                            {{"prefix", prefix},
                                             {"kind", kind ? text_of(*kind) : app::tr("SDR / untagged")}}));
        return;
    }
    const PreviewColorSettings settings = color_settings();
    if (color_mode_ == PreviewColorMode::hdr_to_sdr) {
        QString source = kind ? text_of(*kind) : app::tr("Untagged input (assuming HDR10 / PQ)");
        if (video) {
            static const std::set<std::string, std::less<>> untagged{"", "unknown", "unspecified"};
            if (!kind && !untagged.contains(info.color_transfer)) {
                source = app::tr("SDR input");
            }
            color_status_label->setText(
                app::tr("{prefix}{source} \xC2\xB7 SDR preview", {{"prefix", prefix}, {"source", source}}));
            return;
        }
        color_status_label->setText(
            app::tr("{prefix}{source} \xC2\xB7 fixed HDR \xE2\x86\x92 SDR at {target_nits:g} nit",
                    {{"prefix", prefix}, {"source", source}, {"target_nits", settings.target_nits()}}));
        return;
    }
    if (!kind) {
        const std::string primaries = ascii_lower(info.color_primaries);
        const bool wide_gamut = primaries == "bt2020" || primaries == "bt.2020";
        if (video && wide_gamut && display_hdr_.hdr_enabled == true && !ffmpeg_playback) {
            color_status_label->setText(app::tr(
                "{prefix}BT.2020 wide-gamut input \xC2\xB7 Windows HDR on \xC2\xB7 native 10-bit D3D11 presentation",
                {{"prefix", prefix}}));
            return;
        }
        color_status_label->setText(
            app::tr("{prefix}SDR / untagged input \xC2\xB7 no tone mapping", {{"prefix", prefix}}));
        return;
    }
    const QString target = video ? app::tr("SDR preview") : app::tr("SDR at 100 nit");
    if (display_hdr_.hdr_enabled == true) {
        if (video) {
            if (ffmpeg_playback) {
                color_status_label->setText(app::tr("{prefix}{kind} \xC2\xB7 SDR preview (native HDR unavailable)",
                                                    {{"prefix", prefix}, {"kind", text_of(*kind)}}));
                return;
            }
            color_status_label->setText(app::tr(
                "{prefix}{kind} \xC2\xB7 Windows HDR on \xC2\xB7 native 10-bit D3D11 presentation \xC2\xB7 tone "
                "mapping off",
                {{"prefix", prefix}, {"kind", text_of(*kind)}}));
            return;
        }
        color_status_label->setText(app::tr(
            "{prefix}{kind} \xC2\xB7 display-aware HDR \xE2\x86\x92 SDR \xC2\xB7 Windows HDR on \xC2\xB7 SDR white "
            "{target_nits:g} nit",
            {{"prefix", prefix}, {"kind", text_of(*kind)}, {"target_nits", settings.target_nits()}}));
    } else if (display_hdr_.hdr_enabled == false) {
        color_status_label->setText(
            app::tr("{prefix}{kind} \xC2\xB7 HDR \xE2\x86\x92 {target} \xC2\xB7 Windows HDR off",
                    {{"prefix", prefix}, {"kind", text_of(*kind)}, {"target", target}}));
    } else {
        color_status_label->setText(
            app::tr("{prefix}{kind} \xC2\xB7 HDR \xE2\x86\x92 {target} \xC2\xB7 display HDR state unavailable",
                    {{"prefix", prefix}, {"kind", text_of(*kind)}, {"target", target}}));
    }
}

// ------------------------------------------------------------------- decoding

std::optional<FrameComparePanel::CacheKey> FrameComparePanel::cache_key(FrameSide side) const {
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr) {
        return std::nullopt;
    }
    const FrameComparison& comparison = entry->comparison;
    const PreviewColorSettings colors = color_settings();
    const VideoInfo& info = frame_video_info(comparison, side);
    return CacheKey{entry->identity,
                    frame_,
                    static_cast<int>(side),
                    platform::path_text(info.path),
                    python_repr(info.start_offset) + " " + python_repr(comparison.fps),
                    colors.cache_token(),
                    frame_filter(comparison, side, colors)};
}

const QImage* FrameComparePanel::current_image() const {
    const std::optional<CacheKey> key = cache_key(shown_side());
    if (!key) {
        return nullptr;
    }
    const auto found =
        std::find_if(cache_.begin(), cache_.end(), [&key](const auto& item) { return item.first == *key; });
    return found == cache_.end() ? nullptr : &found->second;
}

void FrameComparePanel::show_or_request() {
    // The current frame, extracted first if need be; a pair that plays is
    // the playback view's.
    if (is_video_mode()) {
        if (video_view != nullptr) {
            video_view->set_color_settings(color_settings());
            video_view->show_source(showing_source_);
        }
        return;
    }
    if (video_view != nullptr) {
        video_view->set_playing(false);
    }
    content_stack_->setCurrentWidget(viewer);
    const FrameComparisonEntry* entry = current_entry();
    if (entry != nullptr) {
        if (const std::optional<std::string> problem =
                untimed_pair_problem(entry->comparison.source_info, entry->comparison.distorted_info)) {
            // Not frames placed by a guessed frame rate: what runs say instead.
            viewer->set_message(tr_message(text_of(*problem)));
            return;
        }
    }
    ensure_auto_crop(entry);
    const std::optional<CacheKey> key = cache_key(shown_side());
    if (key) {
        const auto found =
            std::find_if(cache_.begin(), cache_.end(), [&key](const auto& item) { return item.first == *key; });
        if (found != cache_.end()) {
            std::pair<CacheKey, QImage> item = *found;  // kept as the newest
            cache_.erase(found);
            cache_.push_back(std::move(item));
            viewer->set_image(cache_.back().second);
            return;
        }
        if (const auto error = errors_.find(*key); error != errors_.end()) {
            viewer->set_message(app::tr("Could not load frame:\n{error}", {{"error", tr_message(error->second)}}));
        } else {
            viewer->set_message(shown_side() == FrameSide::source
                                    ? app::tr("Loading source frame {frame:,}\xE2\x80\xA6", {{"frame", frame_}})
                                    : app::tr("Loading test frame {frame:,}\xE2\x80\xA6", {{"frame", frame_}}));
        }
    }
    seek_timer_->start();
}

void FrameComparePanel::request_current_frames() {
    // Extracts the current frame of each side not kept yet, the side shown
    // first.
    const FrameComparisonEntry* entry = current_entry();
    if (entry == nullptr || !isVisible() || is_video_mode()) {
        return;
    }
    if (untimed_pair_problem(entry->comparison.source_info, entry->comparison.distorted_info)) {
        show_or_request();  // its explanation, and no frames to extract
        return;
    }
    refresh_display_hdr();
    std::vector<FrameSide> sides;
    const FrameSide preferred = shown_side();
    for (const FrameSide side :
         {preferred, preferred == FrameSide::source ? FrameSide::distorted : FrameSide::source}) {
        const std::optional<CacheKey> key = cache_key(side);
        const bool cached =
            std::any_of(cache_.begin(), cache_.end(), [&key](const auto& item) { return item.first == *key; });
        if (!cached && !errors_.contains(*key)) {
            sides.push_back(side);
        }
    }
    if (sides.empty()) {
        show_or_request();
        return;
    }
    cancel_workers();
    auto* worker = new FrameExtractWorker(generation_, ffmpeg_(), entry->comparison, frame_, std::move(sides),
                                          color_settings(), this);
    connect(worker, &FrameExtractWorker::frame_ready, this,
            [this](int generation, int side, const QImage& image) { on_frame_ready(generation, side, image); });
    connect(worker, &FrameExtractWorker::frame_failed, this,
            [this](int generation, int side, const QString& error) { on_frame_failed(generation, side, error); });
    connect(worker, &QThread::finished, this, [this, worker] {
        workers_.erase(worker);
        worker->deleteLater();
    });
    workers_.insert(worker);
    worker->start();
}

// -------------------------------------------------------------- black bars

FrameComparisonEntry FrameComparePanel::apply_auto_crop(const FrameComparisonEntry& entry) const {
    // The pair with its black bars, once they are known.
    if (!entry.comparison.auto_crop_pending) {
        return entry;
    }
    const auto found = auto_crops_.find(pair_key(entry));
    if (found == auto_crops_.end()) {
        return entry;
    }
    FrameComparisonEntry cropped = entry;
    cropped.comparison.source_crop = found->second.source;
    cropped.comparison.distorted_crop = found->second.distorted;
    cropped.comparison.auto_crop_pending = false;
    return cropped;
}

void FrameComparePanel::ensure_auto_crop(const FrameComparisonEntry* entry) {
    // Starts detecting a pair's black bars, unless known or under way.
    if (entry == nullptr || !entry->comparison.auto_crop_pending) {
        return;
    }
    const std::string identity = entry->identity;
    const PairKey key = pair_key(*entry);
    if (auto_crops_.contains(key)) {
        // Normally applied by set_runs, but an entry may have been replaced
        // while a result arrived.
        for (FrameComparisonEntry& current : entries_) {
            if (current.identity == identity) {
                current = apply_auto_crop(current);
                break;
            }
        }
        return;
    }
    if (crop_workers_.contains(key)) {
        return;
    }
    const VideoInfo source = entry->comparison.source_info;
    const VideoInfo distorted = entry->comparison.distorted_info;
    if (!is_file(source.path) || !is_file(distorted.path)) {
        return;
    }
    const auto known = [this](const std::string& file) -> CropDetectWorker::Known {
        const auto found = auto_crop_files_.find(file);
        return found == auto_crop_files_.end() ? CropDetectWorker::Known() : CropDetectWorker::Known(found->second);
    };
    const CropDetectWorker::Known source_crop = known(key.source_file);
    const CropDetectWorker::Known distorted_crop = known(key.distorted_file);
    if (source_crop && distorted_crop) {
        auto_crops_[key] = {*source_crop, *distorted_crop};
        for (FrameComparisonEntry& current : entries_) {
            current = apply_auto_crop(current);
        }
        update_labels();
        return;
    }
    auto* worker = new CropDetectWorker(ffmpeg_(), source, distorted, source_crop, distorted_crop, this);
    connect(worker, &CropDetectWorker::ready, this,
            [this, key](const DetectedCrops& crops) { on_auto_crop_ready(key, crops); });
    connect(worker, &QThread::finished, this, [this, worker] {
        for (auto active = crop_workers_.begin(); active != crop_workers_.end(); ++active) {
            if (active->second == worker) {
                crop_workers_.erase(active);
                break;
            }
        }
        worker->deleteLater();
    });
    crop_workers_[key] = worker;
    update_labels();
    worker->start();
}

FrameComparePanel::PairKey FrameComparePanel::pair_key(const FrameComparisonEntry& entry) {
    return {entry.identity, crop_file_key(entry.comparison.source_info.path),
            crop_file_key(entry.comparison.distorted_info.path)};
}

void FrameComparePanel::on_auto_crop_ready(const PairKey& key, const DetectedCrops& crops) {
    // Keeps the black bars found for a pair and its two files, and shows the
    // pair cropped if it is still that pair.
    auto_crops_[key] = crops;
    auto_crop_files_[key.source_file] = crops.source;
    auto_crop_files_[key.distorted_file] = crops.distorted;
    std::optional<FrameComparison> previous;
    for (FrameComparisonEntry& entry : entries_) {
        if (pair_key(entry) == key) {
            previous = entry.comparison;
            entry = apply_auto_crop(entry);
            break;
        }
    }
    const FrameComparisonEntry* current = current_entry();
    if (current != nullptr && pair_key(*current) == key) {
        const bool playing = video_view != nullptr && (video_view->is_playing() || video_view->playback_requested());
        ++generation_;
        update_labels();
        if (is_video_mode()) {
            // Loading the pair again would restart every decoder for nothing
            // (0.7-1.3 s of buffering mid-playback, in the Python app).
            if (previous && previous->auto_crop_pending &&
                VideoCompareView::crops_change_nothing(*previous, current->comparison)) {
                return;  // the pair playing already shows what the crops give
            }
            load_current_video(playing);
        } else {
            show_or_request();
        }
    }
}

void FrameComparePanel::cancel_crop_workers() {
    for (const auto& [identity, worker] : crop_workers_) {
        if (worker->isRunning()) {
            worker->cancel();
        }
    }
}

void FrameComparePanel::on_frame_ready(int generation, int side, const QImage& image) {
    // Keeps an extracted frame, and shows it if it is the side shown.
    if (generation != generation_) {
        return;
    }
    if (image.isNull()) {
        on_frame_failed(generation, side, app::tr("ffmpeg returned an unreadable image."));
        return;
    }
    const std::optional<CacheKey> key = cache_key(static_cast<FrameSide>(side));
    if (!key) {
        return;
    }
    std::erase_if(cache_, [&key](const auto& item) { return item.first == *key; });
    cache_.emplace_back(*key, image);
    const auto bytes = [this] {
        qsizetype total = 0;
        for (const auto& [kept_key, kept] : cache_) {
            total += kept.sizeInBytes();
        }
        return total;
    };
    while (cache_.size() > cache_limit || (cache_.size() > 2 && bytes() > cache_byte_limit)) {
        cache_.pop_front();
    }
    if (static_cast<FrameSide>(side) == shown_side()) {
        viewer->set_image(image);
    }
}

void FrameComparePanel::on_frame_failed(int generation, int side, const QString& error) {
    // Says why a frame could not be extracted.
    if (generation != generation_) {
        return;
    }
    if (const std::optional<CacheKey> key = cache_key(static_cast<FrameSide>(side))) {
        errors_[*key] = error;
    }
    if (static_cast<FrameSide>(side) == shown_side()) {
        viewer->set_message(app::tr("Could not load frame:\n{error}", {{"error", tr_message(error)}}));
    }
}

void FrameComparePanel::cancel_workers() {
    for (FrameExtractWorker* worker : workers_) {
        if (worker->isRunning()) {
            worker->cancel();
        }
    }
}

}  // namespace vml::app
