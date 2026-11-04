#include "chart.h"

#include <QEvent>
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

// Tick spacings that read naturally on a time axis, in seconds.
constexpr std::array<double, 15> time_steps{1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400};
constexpr double target_ticks = 8;

// A band (ChartWidget::set_band) takes this share of the plotting area's
// height, at the bottom; it is not drawn shorter than this many lines of
// text, which its ticks would not fit.
constexpr double band_share = 0.3;
constexpr int band_min_text_lines = 3;

QString time_text(double seconds) { return QString::fromStdString(format_hms(seconds)); }

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
    invalidate(true);
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
        invalidate(true);
    }
}

void ChartWidget::set_series_visible(int key, bool visible) {
    const auto found = series_.find(key);
    if (found != series_.end() && found->second.visible != visible) {
        found->second.visible = visible;
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
        // under it, a higher axis when a model can go above it (VMAF v1's 4K
        // models): real scores are never cut off.
        const double top = std::max(*fixed_y_max_, std::ceil(hi / 5.0) * 5.0);
        const double bottom = std::min(std::floor(lo / 5.0) * 5.0, top - 5);
        y_range_ = {bottom, top};
    } else {
        // Unbounded (decibels, SSIM): a little room, so that the extremes are
        // not drawn on the frame.
        double pad = (hi - lo) * 0.05;
        if (pad == 0) {
            pad = std::max(std::abs(hi) * 0.01, 0.5);
        }
        y_range_ = {lo - pad, hi + pad};
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
    const double fraction = height_fraction((rect.bottom() - py) / static_cast<double>(std::max(1, rect.height())));
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
    for (double tick = std::ceil(x0 / step) * step; tick <= x1; tick += step) {
        ticks.emplace_back(pixel_for_time(tick), time_text(tick));
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
        // starting just below it (Butteraugli's room): both are labelled "0".
        const double shown = std::abs(value) < step * 1e-9 ? 0.0 : value;
        ticks.emplace_back(rect.bottom() - static_cast<int>(fraction * rect.height()), qt_text(format_general(shown)));
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
    if (cursor_x_) {
        painter.setPen(QPen(theme::chart_colours(cache_dark_).crosshair, 1));
        const QRect plot = plot_rect();
        painter.drawLine(*cursor_x_, plot.top(), *cursor_x_, plot.bottom());
    }
}

void ChartWidget::rebuild_cache() {
    // Drawn over the kept drawing when it is the size wanted: a new picture
    // for every step of a zoom or a pan takes 2.7 of the 4.3 ms a 1570x536
    // chart takes to draw (175% display scaling), most of it to get 10 MB of
    // memory from Windows; drawn over, 0.27 ms.
    cache_dark_ = theme::is_dark();
    const double ratio = devicePixelRatioF();
    const QSize size(static_cast<int>(std::ceil(width() * ratio)), static_cast<int>(std::ceil(height() * ratio)));
    if (!cache_ || cache_->size() != size || cache_->devicePixelRatioF() != ratio) {
        cache_.emplace(size);
        cache_->setDevicePixelRatio(ratio);
    }
    draw(*cache_, cache_dark_);
    cache_stale_ = false;
    ++renders_;
}

QPixmap ChartWidget::render(bool dark) const {
    // The chart without the crosshair, at the screen's pixel ratio.
    const double ratio = devicePixelRatioF();
    QPixmap pixmap(static_cast<int>(std::ceil(width() * ratio)), static_cast<int>(std::ceil(height() * ratio)));
    pixmap.setDevicePixelRatio(ratio);
    draw(pixmap, dark);
    return pixmap;
}

void ChartWidget::draw(QPixmap& pixmap, bool dark) const {
    // The chart, over all of `pixmap`.
    const theme::ChartColours colours = theme::chart_colours(dark);
    pixmap.fill(colours.background);
    QPainter painter(&pixmap);
    painter.setFont(font());
    draw_axes(painter, dark);
    const QRect lines = lines_rect();
    for (const ChartSeries* series : visible_series()) {
        draw_series(painter, lines, *series, y_range_);
    }
    painter.setPen(QPen(colours.axis, 1));
    painter.drawRect(lines.adjusted(0, 0, -1, -1));
    if (const std::optional<QRect> band = band_rect()) {
        draw_band(painter, *band, dark);
    }
    painter.end();
}

void ChartWidget::draw_band(QPainter& painter, const QRect& rect, bool dark) const {
    // The band's line, its zero line, frame and label (its grid and ticks
    // are draw_axes').
    const theme::ChartColours colours = theme::chart_colours(dark);
    const auto [y0, y1] = band_range_;
    const double zero_fraction = height_fraction((0.0 - y0) / std::max(1e-9, y1 - y0));
    const int zero_row = rect.bottom() - static_cast<int>(zero_fraction * rect.height());
    painter.setPen(QPen(colours.axis, 1));
    painter.drawLine(rect.left(), zero_row, rect.right(), zero_row);
    ChartSeries line = band_->series;  // spans and a shared owner: cheap to copy
    line.color = colours.difference;
    draw_series(painter, rect, line, band_range_);
    painter.setPen(QPen(colours.axis, 1));
    painter.drawRect(rect.adjusted(0, 0, -1, -1));
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
    return cache_ && cache_dark_ == theme::is_dark() && cache_->devicePixelRatioF() == ratio &&
           cache_->width() == static_cast<int>(std::ceil(width() * ratio)) &&
           cache_->height() == static_cast<int>(std::ceil(height() * ratio));
}

void ChartWidget::draw_axes(QPainter& painter, bool dark) const {
    // The grid, the axes' labels and the value axis's title: the time axis
    // under the plotting area, the lines' value axis beside them, and the
    // band's (if any) beside it.
    const QFontMetrics metrics(painter.font());
    const theme::ChartColours colours = theme::chart_colours(dark);
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

// The dots on a line's points (ChartSeries::marks). More than fit: the largest
// frame of each `seconds_per_column`, on a grid from time 0 as draw_columns' so
// that a drag does not change which, and drawing costs what the width does.
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
    std::ptrdiff_t largest = lo;
    double column = std::floor(series.times[lo] / seconds_per_column);
    for (std::ptrdiff_t point = lo + 1; point < hi; ++point) {
        const double its_column = std::floor(series.times[point] / seconds_per_column);
        if (its_column != column) {
            draw(largest);
            largest = point;
            column = its_column;
        } else if (series.values[point] > series.values[largest]) {
            largest = point;
        }
    }
    draw(largest);
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
        return (rect.bottom() - height_fraction((value - y0) / y_span) * rect.height()) * ratio;
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
    // above and below: the axis's top value is drawn on the row above the
    // plot (to_y), and a VMAF line at 100 would be lost.
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

void ChartWidget::set_cursor_time(std::optional<double> t) {
    std::optional<int> px;
    const QRect plot = plot_rect();
    if (t) {
        px = pixel_for_time(*t);
        if (*px < plot.left() || *px > plot.right()) {
            px.reset();
        }
    }
    if (px == cursor_x_) {
        return;
    }
    const std::optional<int> old = cursor_x_;
    cursor_x_ = px;
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
    if (pan_origin_ && pan_view_) {
        const double span = pan_view_->span;
        const double shift =
            (pan_origin_->x() - pos.x()) / static_cast<double>(std::max(1, plot_rect().width())) * span;
        view_ = View{pan_view_->start + shift, span};
        invalidate();
        emit view_changed();
        return;
    }
    if (!plot_rect().contains(pos)) {
        forget_hover();
        emit left();
        return;
    }
    if (hover_pace_->isActive()) {
        pending_hover_ = pos;  // reported when this frame's time is up
        return;
    }
    report_hover(pos);
}

void ChartWidget::report_hover(QPoint pos) {
    // Says the time and value at `pos`, then waits a frame of the screen
    // before saying the next. A mouse reports far more moves than a screen
    // shows: each would repaint the readout, 500 times a second while the
    // pointer sweeps the chart, keeping a core busy, and the moves between
    // two frames are never seen.
    emit hovered(time_at(pos.x()), value_at(pos.y()));
    hover_pace_->start(frame_ms());
}

void ChartWidget::forget_hover() {
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
        pan_origin_ = event->position().toPoint();
        pan_view_ = current_view();
        setCursor(Qt::ClosedHandCursor);
    }
}

void ChartWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        pan_origin_.reset();
        pan_view_.reset();
        unsetCursor();
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
    const double factor = std::pow(0.8, steps);
    const View view = current_view();
    const double anchor = time_at(event->position().toPoint().x());
    double new_span = view.span * factor;
    const auto [data_lo, data_hi] = data_x_range();
    if (new_span >= data_hi - data_lo) {
        reset_view();
        return;
    }
    new_span = std::max(new_span, seconds_per_pixel() * 8);  // not past a few pixels of data
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

QPixmap ChartWidget::render_to_pixmap() const {
    QPixmap exported = render(false);
    exported.setDevicePixelRatio(1.0);  // what export compositors lay out by
    return exported;
}

}  // namespace vml::app
