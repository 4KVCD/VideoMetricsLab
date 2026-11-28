#include "chart.h"

#include <QContextMenuEvent>
#include <QEvent>
#include <QKeyEvent>
#include <QFontMetrics>
#include <QLineF>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QRegion>
#include <QResizeEvent>
#include <QScreen>
#include <QTimer>
#include <QWheelEvent>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "screen_frame_clock.h"
#include "theme.h"
#include "window_helpers.h"
#include "vml/number_format.h"
#include "vml/time_format.h"

namespace vml::app {

namespace {

// Room for the axis labels around the plotting area.
constexpr int margin_left = 62;
constexpr int margin_right = 12;
constexpr int margin_top = 10;
constexpr int margin_bottom = 38;

// Tick spacings that read naturally on a time axis, in seconds. The steps
// under a second label a short run, or a graph zoomed to a few seconds, with
// more than the one or two whole seconds it spans.
constexpr std::array<double, 21> time_steps{0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                                            1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400};
constexpr double target_ticks = 8;

// A band (ChartWidget::set_band) takes this share of the plotting area's
// height, at the bottom; it is not drawn shorter than this many lines of
// text, which its ticks would not fit.
constexpr double band_share = 0.3;
constexpr int band_min_text_lines = 3;

// A time axis label, with the decimals that tell ticks `step` apart:
// "0:00:02" a second or more apart, "0:00:00.2", "0:00:00.05".
QString time_text(double seconds, double step) {
    const int decimals = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
    return QString::fromStdString(format_hms(seconds, decimals));
}

// A line's frames drawn point by point, a finite stretch at a time: joined
// across a missing frame, it would invent a line through it, and an
// infinity in a QPointF is undefined. Drawn as segments, not as one
// polyline: with display scaling, which makes the pen wider than a pixel,
// Qt outlines a polyline as a shape and fills it, 19 times slower (three
// lines of 720 frames: 13.7 ms against 0.73 ms at 175%); the segments differ
// from it only by a pixel at the tip of some sharp turns.
template <typename ToX, typename ToY>
void draw_points(QPainter& painter, std::span<const double> times, std::span<const float> values, ToX to_x,
                 ToY to_y) {
    std::vector<QLineF> segments;
    segments.reserve(times.size());
    std::optional<QPointF> previous;  // the stretch's last point
    bool joined = false;              // whether it is in a segment
    const auto end_stretch = [&painter, &previous, &joined] {
        if (previous && !joined) {
            painter.drawPoint(*previous);  // a stretch of one frame
        }
        previous.reset();
        joined = false;
    };
    for (std::size_t i = 0; i < times.size(); ++i) {
        const auto value = static_cast<double>(values[i]);
        if (!std::isfinite(value)) {
            end_stretch();
            continue;
        }
        const QPointF point(to_x(times[i]), to_y(value));
        if (previous) {
            segments.emplace_back(*previous, point);
        }
        joined = previous.has_value();
        previous = point;
    }
    end_stretch();
    painter.drawLines(segments.data(), static_cast<int>(segments.size()));
}

// More frames than pixels: one min/max stroke per column, so that drawing
// costs what the width does and no dip is lost. A column is `cell` pixels of
// the screen wide, about a pixel of the mouse (2 at 175 %: one a screen
// pixel wide would draw twice the strokes, 23 -> 36 ms for twelve two-hour
// lines), and a whole number of them, so that a drag moves it by
// whole pixels. A frame's column is its time's on the grid of columns that
// starts at time 0; on the screen, a time's x is its grid position in screen
// pixels less `origin`, a whole number of pixels. Each stroke also reaches
// the frame before its column, so that the line stays joined where a column
// holds about one frame, and across a jump. An infinity counts as a gap, not
// as a huge value that would hide the line, and is not joined across.
template <typename ToY>
void draw_columns(QPainter& painter, std::span<const double> times, std::span<const float> values,
                  double seconds_per_screen_pixel, int cell, double origin, ToY to_y) {
    const double seconds_per_column = seconds_per_screen_pixel * cell;
    constexpr float infinity = std::numeric_limits<float>::infinity();
    std::vector<QLineF> lines;
    lines.reserve(static_cast<std::size_t>((times.back() - times.front()) / seconds_per_column) + 2);
    std::optional<double> column;  // the column being summed up
    std::optional<float> before;   // the last finite frame so far, unless a gap came after it
    bool inked = false;            // whether the column has a finite frame of its own
    float low = infinity;
    float high = -infinity;
    const auto end_column = [&] {
        if (column && inked) {
            // The column's middle: the stroke covers its `cell` pixels wherever it falls.
            const double x = *column * cell - origin + cell / 2.0;
            lines.emplace_back(x, to_y(high), x, to_y(low));
        }
    };
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double at = std::floor(times[i] / seconds_per_column);
        if (at != column) {
            end_column();
            column = at;
            inked = false;
            low = before.value_or(infinity);
            high = before.value_or(-infinity);
        }
        const float value = values[i];
        if (!std::isfinite(value)) {
            before.reset();
            if (!inked) {  // the gap comes between the frame before and this column's
                low = infinity;
                high = -infinity;
            }
            continue;
        }
        low = std::min(low, value);
        high = std::max(high, value);
        before = value;
        inked = true;
    }
    end_column();
    painter.drawLines(lines.data(), static_cast<int>(lines.size()));
}

// How long the pointer waits for a frame of the screen that was asked for, in frames, should none come.
constexpr int frames_without_a_frame = 3;
// How long before a frame of the screen the pointer is reported. Measured at
// 120 Hz on a busy PC: 97-100% of the frames got one report at 4 ms, against
// 95-99% at 3 ms and 75-99% at 2 ms; the report takes about 0.8 ms.
constexpr int hover_lead_ms = 4;

// The rows a value axis spans in `rect`: from its bottom row (the lowest
// value) to its top row (the highest), both on the frame drawn around it.
// Spanning the rect's height instead put the highest value on the row above
// the frame: a VMAF line at 100 over the line for 100.
double rows_spanned(const QRect& rect) {
    return std::max(1, rect.height() - 1);
}

// The row, in `rect`, of a value `fraction` of the way up its axis.
double row_at(const QRect& rect, double fraction) {
    return rect.bottom() - fraction * rows_spanned(rect);
}

}  // namespace

double nice_time_step(double span) {
    if (span <= 0) {
        return 1.0;
    }
    const double ideal = span / target_ticks;
    for (const double step : time_steps) {
        if (step >= ideal) {
            return step;
        }
    }
    // Beyond the table, whole hours.
    const double hours = std::nearbyint(ideal / 3600) * 3600;
    return hours != 0 ? hours : 3600.0;
}

double nice_value_step(double span, double ticks) {
    if (span <= 0) {
        return 1.0;
    }
    const double raw = span / ticks;
    const double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
    for (const double multiple : {1.0, 2.0, 5.0, 10.0}) {
        if (magnitude * multiple >= raw) {
            return magnitude * multiple;
        }
    }
    return magnitude * 10;
}

ChartWidget::ChartWidget(QString y_axis_label, std::optional<double> fixed_y_max, bool invert_y, QWidget* parent)
    : QWidget(parent), y_axis_label_(std::move(y_axis_label)), fixed_y_max_(fixed_y_max), invert_y_(invert_y) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);  // Tab and a click reach it, for its keys
    setMinimumSize(240, 140);
    setAutoFillBackground(false);
    setAttribute(Qt::WA_OpaquePaintEvent);  // every pixel is painted: no clearing first
    resize_settle_ = new QTimer(this);
    resize_settle_->setSingleShot(true);
    resize_settle_->setInterval(100);
    connect(resize_settle_, &QTimer::timeout, this, [this] { update(); });
    hover_pace_ = new QTimer(this);
    hover_pace_->setSingleShot(true);
    hover_pace_->setTimerType(Qt::PreciseTimer);
    connect(hover_pace_, &QTimer::timeout, this, [this] {
        if (pending_hover_) {
            const QPoint pos = *pending_hover_;
            pending_hover_.reset();
            report_hover(pos);
        }
    });
    paint_pace_ = new QTimer(this);
    paint_pace_->setSingleShot(true);
    paint_pace_->setTimerType(Qt::PreciseTimer);
    connect(paint_pace_, &QTimer::timeout, this, [this] {
        if (paint_due_) {
            paint_due_ = false;
            request_paint();
        }
    });
}

// --------------------------------------------------------------------- data

void ChartWidget::set_series(int key, ChartSeries series) {
    series_.insert_or_assign(key, std::move(series));
    forget_view_off_the_data();
    invalidate(true);
}

void ChartWidget::forget_view_off_the_data() {
    // A zoom into the end of a longer line that has gone would show nothing.
    if (view_) {
        const auto [lo, hi] = data_x_range();
        if (view_->start >= hi || view_->start + view_->span <= lo) {
            view_.reset();
        }
    }
}

void ChartWidget::renumber(const std::map<int, int>& renumbered) {
    std::map<int, ChartSeries> series;
    for (auto& [key, line] : series_) {
        const auto found = renumbered.find(key);
        series.emplace(found == renumbered.end() ? key : found->second, std::move(line));
    }
    series_ = std::move(series);
    invalidate(false);
}

void ChartWidget::set_series_color(int key, const QColor& color) {
    // Drawn again, without reading the scores or resetting the zoom.
    const auto found = series_.find(key);
    if (found != series_.end() && found->second.color != color) {
        found->second.color = color;
        invalidate();
    }
}

void ChartWidget::remove_series(const std::vector<int>& keys) {
    bool removed = false;
    for (const int key : keys) {
        removed = series_.erase(key) > 0 || removed;
    }
    if (removed) {
        forget_view_off_the_data();
        invalidate(true);
    }
}

void ChartWidget::set_series_visible(int key, bool visible) {
    const auto found = series_.find(key);
    if (found != series_.end() && found->second.visible != visible) {
        found->second.visible = visible;
        forget_view_off_the_data();
        invalidate(true);
    }
}

void ChartWidget::set_y_axis_label(const QString& label) {
    if (label != y_axis_label_) {
        y_axis_label_ = label;
        invalidate();
    }
}

void ChartWidget::set_band(std::optional<ChartBand> band) {
    if (!band && !band_) {
        return;  // most calls: no band before or after
    }
    // Fitted here, once, not at each step of a pan: from 0 or the lowest
    // value to 0 or the highest, so that the zero line is always in it and
    // the band's height goes to the values there are (two encodes' difference
    // is mostly of one sign).
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    if (band) {
        for (const float value : band->series.values) {
            if (std::isfinite(value)) {
                lo = std::min(lo, static_cast<double>(value));
                hi = std::max(hi, static_cast<double>(value));
            }
        }
    }
    if (!band || lo > hi) {
        band_.reset();  // no value in common to draw
    } else {
        lo = std::min(lo, 0.0);
        hi = std::max(hi, 0.0);
        const double pad = (hi - lo) * 0.05;
        const double flat = band->flat_half_range;
        band_range_ = hi > lo ? std::pair(lo - pad, hi + pad) : std::pair(-flat, flat);
        band_ = std::move(band);
    }
    invalidate(false);  // the lines' value axis stays as it is
}

void ChartWidget::clear() {
    series_.clear();
    band_.reset();
    invalidate(true);
}

bool ChartWidget::has_data() const { return !visible_series().empty(); }

std::vector<const ChartSeries*> ChartWidget::visible_series() const {
    // The shown lines that have points.
    std::vector<const ChartSeries*> shown;
    for (const auto& [key, series] : series_) {
        if (series.visible && !series.times.empty()) {
            shown.push_back(&series);
        }
    }
    return shown;
}

// ------------------------------------------------------------------- ranges

std::pair<double, double> ChartWidget::data_x_range() const {
    const std::vector<const ChartSeries*> shown = visible_series();
    if (shown.empty()) {
        return {0.0, 1.0};
    }
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const ChartSeries* series : shown) {
        lo = std::min(lo, series->times.front());
        hi = std::max(hi, series->times.back());
    }
    return hi > lo ? std::pair(lo, hi) : std::pair(lo, lo + 1.0);
}

ChartWidget::View ChartWidget::current_view() const {
    if (view_) {
        return *view_;
    }
    const auto [lo, hi] = data_x_range();
    return {lo, hi - lo};
}

std::pair<double, double> ChartWidget::x_range() const {
    return view_ ? std::pair(view_->start, view_->start + view_->span) : data_x_range();
}

void ChartWidget::reset_view() {
    view_.reset();
    invalidate(true);
    emit view_changed();
}

void ChartWidget::bring_into_view(double t) {
    if (!view_ || (t >= view_->start && t <= view_->start + view_->span)) {
        return;
    }
    // Over the data, not drifting off its ends, as a drag keeps it
    const double span = view_->span;
    const auto [data_lo, data_hi] = data_x_range();
    view_ = View{std::clamp(t - span / 2.0, data_lo, std::max(data_lo, data_hi - span)), span};
    invalidate();
    emit view_changed();
}

void ChartWidget::recompute_y_range() {
    // Fits the value axis to the shown lines. A line can be all NaN (a
    // metric with no value for any frame): its range would be NaN and the
    // chart blank, so such lines are left out.
    bool any = false;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const ChartSeries* series : visible_series()) {
        for (const float value : series->values) {
            if (std::isfinite(value)) {
                any = true;
                lo = std::min(lo, static_cast<double>(value));
                hi = std::max(hi, static_cast<double>(value));
            }
        }
    }
    if (!any) {
        y_range_ = {0.0, fixed_y_max_ && *fixed_y_max_ != 0 ? *fixed_y_max_ : 1.0};
        return;
    }
    if (fixed_y_max_) {
        // A bounded scale (VMAF): the metric's ceiling when the scores fit
        // under it, else the highest score, so that real scores are never cut
        // off and no empty stretch is left above them.
        const double top = std::max(*fixed_y_max_, hi);
        const double bottom = std::min(std::floor(lo / 5.0) * 5.0, top - 5);
        y_range_ = {bottom, top};
    } else if (hi > lo) {
        // Unbounded (decibels, SSIM, CAMBI): from the lowest score to the
        // highest, each on the frame (rows_spanned), no empty stretch beyond
        // the best score or the worst.
        y_range_ = {lo, hi};
    } else {
        // A flat line: room either side, not below 0 for scores that cannot
        // be (a clean video's CAMBI of 0 at the frame, not in the middle).
        const double pad = std::max(std::abs(hi) * 0.01, 0.5);
        y_range_ = {lo >= 0 ? std::max(0.0, lo - pad) : lo - pad, hi + pad};
    }
}

// ----------------------------------------------------------------- geometry

QRect ChartWidget::plot_rect() const {
    return {margin_left, margin_top, std::max(1, width() - margin_left - margin_right),
            std::max(1, height() - margin_top - margin_bottom)};
}

std::optional<QRect> ChartWidget::band_rect() const {
    if (!band_) {
        return std::nullopt;
    }
    const QRect plot = plot_rect();
    const int band_height = static_cast<int>(std::lround(plot.height() * band_share));
    if (band_height < band_min_text_lines * fontMetrics().height()) {
        return std::nullopt;  // too short a chart: the lines keep all of it
    }
    return QRect(plot.left(), plot.bottom() + 1 - band_height, plot.width(), band_height);
}

QRect ChartWidget::lines_rect() const {
    const QRect plot = plot_rect();
    const std::optional<QRect> band = band_rect();
    if (!band) {
        return plot;
    }
    // A line of text between them, so that the lines' lowest tick label and
    // the band's highest do not touch.
    const int gap = fontMetrics().height();
    return {plot.left(), plot.top(), plot.width(), std::max(1, band->top() - gap - plot.top())};
}

double ChartWidget::time_at(int px) const {
    const QRect rect = plot_rect();
    const auto [x0, x1] = x_range();
    const double fraction = (px - rect.left()) / static_cast<double>(std::max(1, rect.width()));
    return x0 + fraction * (x1 - x0);
}

double ChartWidget::time_within_data(int px) const {
    // A drag may run past the plotting area: held to the lines' ends.
    const auto [data_lo, data_hi] = data_x_range();
    return std::clamp(time_at(px), data_lo, std::max(data_lo, data_hi));
}

double ChartWidget::value_at(int py) const {
    // height_fraction maps a height back to a value too: the same flip.
    const QRect rect = lines_rect();
    if (py > rect.bottom() && band_rect()) {
        // Over the band: every point is at or below this (MetricPage's hover
        // compares sign * value, the sign flipped as the axis is).
        constexpr double infinity = std::numeric_limits<double>::infinity();
        return invert_y_ ? -infinity : infinity;
    }
    const auto [y0, y1] = y_range_;
    const double fraction = height_fraction((rect.bottom() - py) / rows_spanned(rect));
    return y0 + fraction * (y1 - y0);
}

int ChartWidget::pixel_for_time(double t) const {
    const QRect rect = plot_rect();
    const auto [x0, x1] = x_range();
    return rect.left() + static_cast<int>(std::nearbyint((t - x0) / std::max(1e-9, x1 - x0) * rect.width()));
}

std::vector<std::pair<int, QString>> ChartWidget::time_ticks() const {
    const auto [x0, x1] = x_range();
    const double step = nice_time_step(x1 - x0);
    std::vector<std::pair<int, QString>> ticks;
    // Each tick from its index, not by adding steps up: 0.1 added ten times
    // falls short of 1.0, which would lose a tick at the axis' end.
    const double first = std::ceil(x0 / step - 1e-9);
    for (double i = first; i * step <= x1 + step * 1e-9; ++i) {
        const double tick = i * step;
        ticks.emplace_back(pixel_for_time(tick), time_text(tick, step));
    }
    return ticks;
}

std::vector<std::pair<int, QString>> ChartWidget::value_ticks(std::optional<QRect> rect) const {
    return ticks_for(rect.value_or(lines_rect()), y_range_, target_ticks);
}

std::vector<std::pair<int, QString>> ChartWidget::band_ticks() const {
    const std::optional<QRect> band = band_rect();
    if (!band) {
        return {};
    }
    // About one label for every two lines of text.
    const double count = std::max(2.0, band->height() / (2.0 * fontMetrics().height()));
    return ticks_for(*band, band_range_, count);
}

std::vector<std::pair<int, QString>> ChartWidget::ticks_for(const QRect& rect, std::pair<double, double> range,
                                                            double count) const {
    const auto [y0, y1] = range;
    const double step = nice_value_step(y1 - y0, count);
    std::vector<std::pair<int, QString>> ticks;
    for (double value = std::ceil(y0 / step) * step; value <= y1; value += step) {
        const double fraction = height_fraction((value - y0) / std::max(1e-9, y1 - y0));
        // The sum of steps lands next to 0, not on it, for a range across it
        // (5.55e-17 for a band of +-0.0015), and ceil() gives -0.0 for a range
        // starting just below it: both are labelled "0".
        const double shown = std::abs(value) < step * 1e-9 ? 0.0 : value;
        // On the row the lines draw the value on (draw_series' to_y), rounded.
        ticks.emplace_back(static_cast<int>(std::lround(row_at(rect, fraction))), qt_text(format_general(shown)));
    }
    return ticks;
}

double ChartWidget::seconds_per_pixel() const {
    const auto [x0, x1] = x_range();
    return (x1 - x0) / std::max(1, plot_rect().width());
}

// ----------------------------------------------------------------- painting

void ChartWidget::invalidate(bool refit) {
    // Drawn again, the value axis fitted first with `refit`.
    if (refit) {
        recompute_y_range();
    }
    cache_stale_ = true;
    request_paint();
}

void ChartWidget::request_paint() {
    // Drawn again, at most once a frame of the screen: dragged by a mouse
    // that reports a move every millisecond, the chart would be drawn afresh
    // 390 times a second (87% of a core) where the screen shows 120.
    if (paint_pace_->isActive()) {
        paint_due_ = true;  // drawn when this frame's time is up
        return;
    }
    update();
    paint_pace_->start(frame_ms());
}

int ChartWidget::frame_ms() const {
    // A frame of the screen the chart is on, in whole milliseconds.
    const double rate = screen() != nullptr ? screen()->refreshRate() : 60.0;
    return static_cast<int>(1000.0 / std::clamp(rate, 30.0, 500.0));
}

void ChartWidget::resizeEvent(QResizeEvent* event) {
    if (cache_) {
        if (isVisible()) {
            resize_settle_->start();  // drawn again at the size it settles on (paintEvent)
        } else {
            // Resized while hidden (the window resized on another tab): drawn
            // afresh when shown, not stretched for 100 ms.
            cache_stale_ = true;
        }
    }
    QWidget::resizeEvent(event);
}

void ChartWidget::changeEvent(QEvent* event) {
    // The window's theme changed: drawn again in its colours.
    if (event->type() == QEvent::PaletteChange || event->type() == QEvent::ApplicationPaletteChange ||
        event->type() == QEvent::StyleChange) {
        cache_stale_ = true;
        update();
    }
    QWidget::changeEvent(event);
}

void ChartWidget::paintEvent(QPaintEvent* /*event*/) {
    // The kept drawing, with the crosshair.
    if (cache_stale_ || !cache_matches_display()) {
        if (!cache_stale_ && cache_ && resize_settle_->isActive()) {
            QPainter painter(this);
            painter.drawPixmap(rect(), *cache_);  // stretched, still being resized
            return;
        }
        rebuild_cache();
    }
    QPainter painter(this);
    // Only the damaged area is copied, so moving the crosshair repaints a few
    // pixel columns, not the whole chart. Drawn at the logical origin, so
    // that a high-DPI pixmap's pixels are not taken for logical ones.
    painter.drawPixmap(0, 0, *cache_);
    if (const std::optional<int> x = cursor_pixel()) {
        painter.setPen(QPen(theme::chart_colours().crosshair, 1));
        const QRect plot = plot_rect();
        painter.drawLine(*x, plot.top(), *x, plot.bottom());
    }
}

void ChartWidget::rebuild_cache() {
    // Drawn over the kept drawing when it is the size wanted: a new picture
    // for every step of a zoom or a pan takes 2.7 of the 4.3 ms a 1570x536
    // chart takes to draw (175% display scaling), most of it to get 10 MB of
    // memory from Windows; drawn over, 0.27 ms.
    cache_look_ = theme::look();
    const double ratio = devicePixelRatioF();
    const QSize size(static_cast<int>(std::ceil(width() * ratio)), static_cast<int>(std::ceil(height() * ratio)));
    if (!cache_ || cache_->size() != size || cache_->devicePixelRatioF() != ratio) {
        cache_.emplace(size);
        cache_->setDevicePixelRatio(ratio);
    }
    draw(*cache_, theme::chart_colours());  // the look in use's (Mocha's browns, Paper's card)
    cache_stale_ = false;
    ++renders_;
}

QPixmap ChartWidget::render(const theme::ChartColours& colours) const {
    // The chart without the crosshair, at the screen's pixel ratio.
    const double ratio = devicePixelRatioF();
    QPixmap pixmap(static_cast<int>(std::ceil(width() * ratio)), static_cast<int>(std::ceil(height() * ratio)));
    pixmap.setDevicePixelRatio(ratio);
    draw(pixmap, colours);
    return pixmap;
}

void ChartWidget::draw(QPixmap& pixmap, const theme::ChartColours& colours) const {
    // The chart, over all of `pixmap`, in `colours`.
    pixmap.fill(colours.background);
    QPainter painter(&pixmap);
    painter.setFont(font());
    draw_axes(painter, colours);
    const QRect lines = lines_rect();
    const std::optional<QRect> band = band_rect();
    // The shaded stretches as columns of pixels, those that touch or overlap
    // drawn as one: thousands of weak segments (a long video's) are each a
    // pixel or two at most, and one fill per stretch took 200 ms a paint.
    std::vector<std::pair<int, int>> columns;
    columns.reserve(shaded_.size());
    for (const auto& seconds : shaded_) {
        const int left = std::max(lines.left(), pixel_for_time(seconds.first));
        const int right = std::min(lines.right(), pixel_for_time(seconds.second));
        if (right < lines.left() || left > lines.right()) {
            continue;
        }
        columns.emplace_back(left, std::max(left, right));
    }
    std::ranges::sort(columns);
    std::vector<std::pair<int, int>> merged;
    for (const auto& column : columns) {
        if (!merged.empty() && column.first <= merged.back().second + 1) {
            merged.back().second = std::max(merged.back().second, column.second);
        } else {
            merged.push_back(column);
        }
    }
    for (const auto& [left, right] : merged) {
        painter.fillRect(QRect(QPoint(left, lines.top()), QPoint(right, lines.bottom())), colours.highlight);
        if (band) {
            painter.fillRect(QRect(QPoint(left, band->top()), QPoint(right, band->bottom())), colours.highlight);
        }
    }
    draw_highlight(painter, lines, colours.highlight);
    if (band) {
        draw_highlight(painter, *band, colours.highlight);
    }
    // The frame under the lines: the axis's ends are its top and bottom rows,
    // and a line there (VMAF at 100) is drawn on it, not hidden by it.
    painter.setPen(QPen(colours.axis, 1));
    painter.drawRect(lines.adjusted(0, 0, -1, -1));
    for (const ChartSeries* series : visible_series()) {
        draw_series(painter, lines, *series, y_range_);
    }
    if (band) {
        draw_band(painter, *band, colours);
    }
    painter.end();
}

void ChartWidget::draw_highlight(QPainter& painter, const QRect& rect, const QColor& colour) const {
    if (highlight_) {
        draw_shade(painter, rect, colour, *highlight_);
    }
}

void ChartWidget::draw_shade(QPainter& painter, const QRect& rect, const QColor& colour,
                             const std::pair<double, double>& seconds) const {
    // The times' columns of `rect`, at least one pixel wide, so that a
    // single frame of a long run still shows; nothing out of view.
    const int left = std::max(rect.left(), pixel_for_time(seconds.first));
    const int right = std::min(rect.right(), pixel_for_time(seconds.second));
    if (right < rect.left() || left > rect.right()) {
        return;
    }
    painter.fillRect(QRect(QPoint(left, rect.top()), QPoint(std::max(left, right), rect.bottom())), colour);
}

void ChartWidget::draw_band(QPainter& painter, const QRect& rect, const theme::ChartColours& colours) const {
    // The band's line, its zero line, frame and label (its grid and ticks
    // are draw_axes').
    const auto [y0, y1] = band_range_;
    const double zero_fraction = height_fraction((0.0 - y0) / std::max(1e-9, y1 - y0));
    const auto zero_row = static_cast<int>(std::lround(row_at(rect, zero_fraction)));
    painter.setPen(QPen(colours.axis, 1));
    painter.drawLine(rect.left(), zero_row, rect.right(), zero_row);
    painter.drawRect(rect.adjusted(0, 0, -1, -1));  // under the line, as the lines' frame
    ChartSeries line = band_->series;  // spans and a shared owner: cheap to copy
    line.color = colours.difference;
    draw_series(painter, rect, line, band_range_);
    // Over the line, on a patch of the background, so that it stays
    // readable where the line runs under it (a difference near the top of
    // its range, as most are).
    const QFontMetrics metrics(painter.font());
    const QString label = metrics.elidedText(band_->label, Qt::ElideMiddle, rect.width() - 12);
    QColor patch = colours.background;
    patch.setAlpha(220);
    painter.fillRect(QRect(rect.left() + 2, rect.top() + 2, metrics.horizontalAdvance(label) + 8, metrics.height() + 2),
                     patch);
    painter.setPen(QPen(colours.difference, 1));
    painter.drawText(rect.left() + 6, rect.top() + metrics.ascent() + 3, label);
}

bool ChartWidget::cache_matches_display() const {
    // Whether the kept drawing still fits the size, pixel ratio and theme.
    const double ratio = devicePixelRatioF();
    return cache_ && cache_look_ == theme::look() && cache_->devicePixelRatioF() == ratio &&
           cache_->width() == static_cast<int>(std::ceil(width() * ratio)) &&
           cache_->height() == static_cast<int>(std::ceil(height() * ratio));
}

void ChartWidget::draw_axes(QPainter& painter, const theme::ChartColours& colours) const {
    // The grid, the axes' labels and the value axis's title: the time axis
    // under the plotting area, the lines' value axis beside them, and the
    // band's (if any) beside it.
    const QFontMetrics metrics(painter.font());
    const QRect plot = plot_rect();
    const QRect lines = lines_rect();
    const std::optional<QRect> band = band_rect();

    painter.setPen(QPen(colours.grid, 1));
    const std::vector<std::pair<int, QString>> times = time_ticks();
    for (const auto& [px, label] : times) {
        painter.drawLine(px, lines.top(), px, lines.bottom());
        if (band) {
            painter.drawLine(px, band->top(), px, band->bottom());
        }
    }
    const std::vector<std::pair<int, QString>> ticks = value_ticks(lines);
    const std::vector<std::pair<int, QString>> difference_ticks = band_ticks();
    const auto draw_rows = [&painter](const QRect& rect, const std::vector<std::pair<int, QString>>& rows) {
        for (const auto& [py, label] : rows) {
            painter.drawLine(rect.left(), py, rect.right(), py);
        }
    };
    draw_rows(lines, ticks);
    if (band) {
        draw_rows(*band, difference_ticks);
    }

    painter.setPen(QPen(colours.text, 1));
    for (const auto& [px, label] : times) {
        // Centred on its tick, but kept inside the widget: a tick at the
        // right end lost half its label ("0:00:0").
        const int label_width = metrics.horizontalAdvance(label);
        const int left = std::min(px - label_width / 2, width() - label_width - 1);
        painter.drawText(left, plot.bottom() + metrics.height() + 2, label);
    }
    for (const auto* row_ticks : {&ticks, &difference_ticks}) {
        for (const auto& [py, label] : *row_ticks) {
            painter.drawText(plot.left() - metrics.horizontalAdvance(label) - 6, py + metrics.ascent() / 2, label);
        }
    }
    if (!y_axis_label_.isEmpty()) {
        painter.save();
        painter.translate(14, lines.center().y());
        painter.rotate(-90);
        // Centred as Python's (-width) // 2 centres it: rounded down.
        painter.drawText(-(metrics.horizontalAdvance(y_axis_label_) + 1) / 2, 0, y_axis_label_);
        painter.restore();
    }
}

namespace {

// The dots on a line's points (ChartSeries::marks). Where the marked points
// in view are no more than the columns of `seconds_per_column` (Worst Frames'
// dips, up to 999), every one, each at its own height: drawing costs no more
// than the width allows. More than fit (the Bitrate Viewer's, every frame
// marked with its type): one point a column, on a grid from time 0 as
// draw_columns' so that a drag does not change which, the largest marked one
// (the largest frame). Kept by size, a column's few marked points lost to an
// unmarked neighbour above them (no dot), or the deepest dip to a shallower
// one beside it.
template <typename ToX, typename ToY>
void draw_marks(QPainter& painter, const ChartSeries& series, std::ptrdiff_t lo, std::ptrdiff_t hi,
                double seconds_per_column, double radius, ToX to_x, ToY to_y) {
    painter.setPen(Qt::NoPen);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const auto draw = [&](std::ptrdiff_t point) {
        const std::uint8_t mark = series.marks[point];
        if (mark != 0 && mark < series.mark_colors.size()) {
            painter.setBrush(series.mark_colors[mark]);
            painter.drawEllipse(QPointF(to_x(series.times[point]), to_y(series.values[point])), radius, radius);
        }
    };
    if (hi <= lo) {
        return;  // no point in view
    }
    const double first_column = std::floor(series.times[lo] / seconds_per_column);
    const auto columns = static_cast<std::ptrdiff_t>(std::floor(series.times[hi - 1] / seconds_per_column) - first_column) + 1;
    std::ptrdiff_t marked = 0;
    for (std::ptrdiff_t point = lo; point < hi && marked <= columns; ++point) {
        marked += series.marks[point] != 0 ? 1 : 0;
    }
    if (marked <= columns) {
        for (std::ptrdiff_t point = lo; point < hi; ++point) {
            draw(point);  // an unmarked one draws nothing
        }
        return;
    }
    const auto better = [&series](std::ptrdiff_t point, std::ptrdiff_t than) {
        const bool marked = series.marks[point] != 0;
        const bool than_marked = series.marks[than] != 0;
        return marked != than_marked ? marked : series.values[point] > series.values[than];
    };
    std::ptrdiff_t kept = lo;
    double column = first_column;
    for (std::ptrdiff_t point = lo + 1; point < hi; ++point) {
        const double its_column = std::floor(series.times[point] / seconds_per_column);
        if (its_column != column) {
            draw(kept);
            kept = point;
            column = its_column;
        } else if (better(point, kept)) {
            kept = point;
        }
    }
    draw(kept);
}

}  // namespace

void ChartWidget::draw_series(QPainter& painter, const QRect& rect, const ChartSeries& series,
                              std::pair<double, double> range) const {
    // One line: point by point when there are few, else one min/max stroke
    // per column of about a pixel of the mouse (draw_columns).
    //
    // Dragged left or right, the line must move without changing shape. So
    // it is drawn in the screen's own pixels (at 175 %, a pixel of the mouse
    // is 1.75 of the screen's), on a grid of pixels that starts at time 0,
    // not at the view's start: a frame's place within its pixel, and the
    // frames a column sums up, are the same wherever the view starts, and a
    // drag moves the whole drawing by whole pixels.
    const View view = current_view();
    const double y0 = range.first;  // not a structured binding: the lambda below takes it
    const double y1 = range.second;
    const double ratio = painter.device()->devicePixelRatioF();
    const double seconds_per_screen_pixel = std::max(1e-12, view.span) / (rect.width() * ratio);
    const int cell = std::max(1, static_cast<int>(std::lround(ratio)));  // a column's width (draw_columns)
    // A time's x on the screen: its pixel on the grid, less the grid's pixel
    // at the plot's left edge (a whole number). The view's start falls within
    // half a pixel of the edge.
    const double origin = std::nearbyint(view.start / seconds_per_screen_pixel - rect.left() * ratio);
    const auto to_x = [=](double t) { return t / seconds_per_screen_pixel - origin; };
    const double y_span = std::max(1e-9, y1 - y0);
    const auto to_y = [&](double value) {
        return row_at(rect, height_fraction((value - y0) / y_span)) * ratio;
    };
    // The frames in view, those of the edge columns' pixels outside it, and
    // one more each side, which a line in view is joined to.
    const std::span<const double> times = series.times;
    const auto count_all = static_cast<std::ptrdiff_t>(times.size());
    const double margin = (cell + 1) * seconds_per_screen_pixel;
    std::ptrdiff_t lo = std::lower_bound(times.begin(), times.end(), view.start - margin) - times.begin();
    std::ptrdiff_t hi = std::upper_bound(times.begin(), times.end(), view.start + view.span + margin) - times.begin();
    lo = std::max<std::ptrdiff_t>(0, lo - 1);
    hi = std::min(count_all, hi + 1);
    if (hi - lo < 1) {
        return;
    }
    const std::span<const double> shown_times = times.subspan(lo, hi - lo);
    const std::span<const float> shown_values = series.values.subspan(lo, hi - lo);
    painter.save();
    // What lies outside the view stays out of the side margins. Not clipped
    // above and below: a line at the axis's top or bottom is drawn on the
    // frame's row (to_y), and a pen wider than a screen pixel would lose half
    // of it.
    painter.setClipRect(QRect(rect.left(), 0, rect.width(), height()));
    painter.scale(1.0 / ratio, 1.0 / ratio);
    painter.setPen(QPen(series.color, ratio));  // a pixel of the mouse wide, as before

    // Point by point while the view holds at most two frames a pixel of the
    // mouse. Counted from the line's frame rate, not from the frames in
    // view, which a drag changes by one or two: near the threshold, each
    // drag would switch between the two ways of drawing.
    const double data_span = times.back() - times.front();
    const double frames_in_view = data_span > 0 ? static_cast<double>(count_all - 1) / data_span * view.span : 0.0;
    if (frames_in_view <= rect.width() * 2.0) {
        draw_points(painter, shown_times, shown_values, to_x, to_y);
    } else {
        draw_columns(painter, shown_times, shown_values, seconds_per_screen_pixel, cell, origin, to_y);
    }
    if (series.marks.size() == times.size()) {
        const double radius = 2 * ratio;  // a dot 4 pixels of the mouse across, and as many apart
        draw_marks(painter, series, lo, hi, seconds_per_screen_pixel * 2 * radius, radius, to_x, to_y);
    }
    painter.restore();
}

// ------------------------------------------------------------------- cursor

void ChartWidget::set_shaded(std::vector<std::pair<double, double>> seconds) {
    if (seconds == shaded_) {
        return;
    }
    shaded_ = std::move(seconds);
    invalidate();
}

void ChartWidget::set_highlight(std::optional<std::pair<double, double>> seconds) {
    if (seconds == highlight_) {
        return;
    }
    highlight_ = seconds;
    invalidate();
}

std::optional<int> ChartWidget::cursor_pixel() const {
    if (!cursor_time_) {
        return std::nullopt;
    }
    const int px = pixel_for_time(*cursor_time_);
    const QRect plot = plot_rect();
    return px < plot.left() || px > plot.right() ? std::nullopt : std::optional<int>(px);
}

void ChartWidget::set_cursor_time(std::optional<double> t) {
    const std::optional<int> old = cursor_pixel();
    cursor_time_ = t;
    const std::optional<int> px = cursor_pixel();
    if (px == old) {
        return;
    }
    const QRect plot = plot_rect();
    QRegion strips;
    for (const std::optional<int>& x : {old, px}) {
        if (x) {
            // Pen caps and fractional display scaling can touch pixels beside
            // the line: the fringe is painted again at both ends too.
            strips += QRect(*x - 1, plot.top() - 2, 3, plot.height() + 4);
        }
    }
    // Painted and put on screen now. Left to the next paint, the strips would
    // go with whatever else is due then (the readout below the chart, at each
    // move of the pointer), and Windows copies the rectangle around all of
    // it to the screen: most of the chart, which took more than the rest of
    // the move.
    if (isVisible()) {
        repaint(strips);
    } else {
        update(strips);
    }
}

// -------------------------------------------------------------------- input

void ChartWidget::mouseMoveEvent(QMouseEvent* event) {
    // Pans while dragging; otherwise says the time and value under the
    // pointer.
    const QPoint pos = event->position().toPoint();
    if ((selection_start_ || pan_origin_) && !event->buttons().testFlag(Qt::LeftButton)) {
        // A drag whose release never came here (a dialog took the mouse
        // mid-drag): it ends where the button is first seen up, as the
        // release would have ended it, and the move is a plain move.
        end_drag(pos);
    }
    if (selection_start_) {
        emit range_dragged(*selection_start_, time_within_data(pos.x()), false);
        return;
    }
    if (pan_origin_ && pan_view_) {
        const double span = pan_view_->span;
        const double shift =
            (pan_origin_->x() - pos.x()) / static_cast<double>(std::max(1, plot_rect().width())) * span;
        // Over the data, not drifting off its ends, as the wheel keeps it.
        const auto [data_lo, data_hi] = data_x_range();
        view_ = View{std::clamp(pan_view_->start + shift, data_lo, std::max(data_lo, data_hi - span)), span};
        invalidate();
        emit view_changed();
        return;
    }
    if (!plot_rect().contains(pos)) {
        forget_hover();
        emit left();
        return;
    }
    if (waiting_for_hover_frame()) {
        pending_hover_ = pos;  // reported at this frame's time
        return;
    }
    report_hover(pos);
    ask_for_hover_frame();
}

void ChartWidget::report_hover(QPoint pos) {
    key_time_ = time_at(pos.x());  // the keys go on from where the pointer is
    emit hovered(*key_time_, value_at(pos.y()));
}

bool ChartWidget::waiting_for_hover_frame() const {
    return hover_frame_asked_ && hover_frame_asked_at_.elapsed() < frames_without_a_frame * frame_ms();
}

void ChartWidget::ask_for_hover_frame() {
    // A mouse reports far more moves than a screen shows: each would repaint
    // the readout, 500 times a second while the pointer sweeps the chart,
    // keeping a core busy. So the pointer is reported once a frame of the
    // screen. A timer cannot keep to that: it drifts against the frames, and a
    // whole frame's wait (8.8 ms against a 120 Hz screen's 8.33) left 6-9% of
    // them without a new position.
    ScreenFrameClock& clock = ScreenFrameClock::instance();
    const unsigned request = ++hover_frame_request_;
    hover_frame_asked_ = true;
    hover_frame_asked_at_.start();
    connect(
        &clock, &ScreenFrameClock::frame_started, this,
        [this, request] {
            if (request == hover_frame_request_) {  // not one asked for before the pointer left
                on_hover_frame();
            }
        },
        Qt::SingleShotConnection);
    clock.request_frame(window()->windowHandle());
}

void ChartWidget::on_hover_frame() {
    hover_frame_asked_ = false;
    if (!pending_hover_) {
        return;  // the pointer has stopped: its next move is reported at once
    }
    // The next frame is asked for now: asked for after the report, close to the
    // frame's end, it can come a frame late (61-96% of a 120 Hz screen's frames
    // got a report).
    ask_for_hover_frame();
    // The report is made shortly before the next frame, so that the frame shows
    // the pointer's newest position: made now, it would be nearly a frame old.
    hover_pace_->start(std::max(0, frame_ms() - hover_lead_ms));
}

void ChartWidget::forget_hover() {
    ++hover_frame_request_;
    hover_frame_asked_ = false;
    hover_pace_->stop();
    pending_hover_.reset();
}

void ChartWidget::leaveEvent(QEvent* event) {
    forget_hover();
    emit left();
    QWidget::leaveEvent(event);
}

void ChartWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        forget_hover();
        if (range_selection_ && event->modifiers().testFlag(Qt::ShiftModifier) && has_data()) {
            selection_start_ = time_within_data(event->position().toPoint().x());
            setCursor(Qt::SizeHorCursor);
            emit range_dragged(*selection_start_, *selection_start_, false);
            return;
        }
        pan_origin_ = event->position().toPoint();
        pan_view_ = current_view();
        setCursor(Qt::ClosedHandCursor);
    }
}

void ChartWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        end_drag(event->position().toPoint());
    }
}

void ChartWidget::end_drag(QPoint pos) {
    pan_origin_.reset();
    pan_view_.reset();
    unsetCursor();
    if (const std::optional<double> start = std::exchange(selection_start_, std::nullopt)) {
        emit range_dragged(*start, time_within_data(pos.x()), true);
    }
}

void ChartWidget::mouseDoubleClickEvent(QMouseEvent* /*event*/) { reset_view(); }

void ChartWidget::wheelEvent(QWheelEvent* event) {
    // Zooms the time axis around the pointer: what is under it stays under it.
    // A trackpad's glide after the fingers lift zooms no further; and the
    // wheel zooms the same way round whichever way the system scrolls
    // (macOS's natural scrolling, inverted, scrolls the other way).
    if (event->phase() == Qt::ScrollMomentum) {
        event->accept();
        return;
    }
    const double steps = (event->inverted() ? -1 : 1) * event->angleDelta().y() / 120.0;
    if (steps == 0) {
        return;
    }
    zoom_by(time_at(event->position().toPoint().x()), std::pow(0.8, steps));
}

void ChartWidget::zoom_by(double anchor, double factor) {
    const View view = current_view();
    // No narrower than five of the time axis' finest steps: zoomed past them,
    // it would have one label or none.
    const double new_span = std::max(view.span * factor, 5 * time_steps.front());
    const auto [data_lo, data_hi] = data_x_range();
    if (new_span >= data_hi - data_lo) {
        reset_view();
        return;
    }
    const double left_share = (anchor - view.start) / std::max(1e-9, view.span);
    double new_x0 = anchor - left_share * new_span;
    // Over the data, not drifting off its ends.
    if (new_x0 < data_lo) {
        new_x0 = data_lo;
    }
    if (new_x0 + new_span > data_hi) {
        new_x0 = data_hi - new_span;
    }
    view_ = View{new_x0, new_span};
    invalidate();
    emit view_changed();
}

bool ChartWidget::event(QEvent* event) {
    if (event->type() == QEvent::ContextMenu &&
        static_cast<QContextMenuEvent*>(event)->reason() == QContextMenuEvent::Keyboard) {
        const std::span<const double> times = key_times();
        if (!times.empty()) {
            if (!key_time_) {  // none yet: the view's middle
                const auto [lo, hi] = x_range();
                move_key_cursor((lo + hi) / 2.0);
            }
            // At the cursor, at the top of the lines: a height every point
            // is at or below, as the keys' readout (move_key_cursor).
            emit customContextMenuRequested(QPoint(pixel_for_time(*key_time_), lines_rect().top() + 1));
            event->accept();
            return true;
        }
    }
    return QWidget::event(event);
}

std::span<const double> ChartWidget::key_times() const {
    for (const ChartSeries* series : visible_series()) {
        if (!series->times.empty()) {
            return series->times;
        }
    }
    return {};
}

void ChartWidget::move_key_cursor(double t) {
    key_time_ = t;
    bring_into_view(t);
    // Read out as the pointer is, at a height every point is at or below
    // (value_at over the band): the frame at the cursor itself, not a dip
    // near it, which the pointer's readout snaps to.
    constexpr double infinity = std::numeric_limits<double>::infinity();
    emit hovered(t, invert_y_ ? -infinity : infinity);
}

void ChartWidget::keyPressEvent(QKeyEvent* event) {
    // Left and Right a point (Shift: ten), Home and End the first and last,
    // Page Up and Down a view's width, + and - zoom around the cursor, 0 the
    // whole of the data, Esc the cursor away. (The menu key: event.)
    const std::span<const double> times = key_times();
    const Qt::KeyboardModifiers modifiers = event->modifiers() & ~Qt::KeypadModifier;
    const bool plain = modifiers == Qt::NoModifier;
    const bool shift = modifiers == Qt::ShiftModifier;
    if (times.empty() || !(plain || shift)) {
        QWidget::keyPressEvent(event);
        return;
    }
    const auto [view_lo, view_hi] = x_range();
    // The point at the cursor, or, with none yet, the one in the view's middle.
    const double at = key_time_.value_or((view_lo + view_hi) / 2.0);
    const auto nearest = [&times](double t) {
        const auto found = std::lower_bound(times.begin(), times.end(), t);
        if (found == times.end()) {
            return times.size() - 1;
        }
        const auto index = static_cast<std::size_t>(found - times.begin());
        return index > 0 && t - times[index - 1] < *found - t ? index - 1 : index;
    };
    const std::size_t index = nearest(at);
    const auto step_to = [&](long long offset) {
        const long long last = static_cast<long long>(times.size()) - 1;
        // Without a cursor yet, the first press puts it in the view's middle.
        const long long to = key_time_ ? std::clamp(static_cast<long long>(index) + offset, 0LL, last)
                                       : static_cast<long long>(index);
        move_key_cursor(times[static_cast<std::size_t>(to)]);
    };
    switch (event->key()) {
    case Qt::Key_Left:
    case Qt::Key_Right:
        step_to((event->key() == Qt::Key_Left ? -1 : 1) * (shift ? 10 : 1));
        break;
    case Qt::Key_Home:
    case Qt::Key_End:
        if (!plain) {
            QWidget::keyPressEvent(event);
            return;
        }
        move_key_cursor(event->key() == Qt::Key_Home ? times.front() : times.back());
        break;
    case Qt::Key_PageUp:
    case Qt::Key_PageDown:
        if (!plain) {
            QWidget::keyPressEvent(event);
            return;
        }
        move_key_cursor(times[nearest(at + (event->key() == Qt::Key_PageUp ? -1 : 1) * (view_hi - view_lo))]);
        break;
    case Qt::Key_Plus:
    case Qt::Key_Equal:
    case Qt::Key_Minus:
        zoom_by(at, event->key() == Qt::Key_Minus ? 2.0 : 0.5);
        break;
    case Qt::Key_0:
        reset_view();
        break;
    case Qt::Key_Escape:
        if (!key_time_) {
            QWidget::keyPressEvent(event);
            return;
        }
        key_time_.reset();
        emit left();
        break;
    default:
        QWidget::keyPressEvent(event);
        return;
    }
    event->accept();
}

QPixmap ChartWidget::render_to_pixmap(const theme::ChartColours& colours) const {
    QPixmap exported = render(colours);
    exported.setDevicePixelRatio(1.0);  // what export compositors lay out by
    return exported;
}

}  // namespace vml::app
