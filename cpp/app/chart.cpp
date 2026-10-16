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

QString time_text(double seconds) { return QString::fromStdString(format_hms(seconds)); }

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

double nice_value_step(double span) {
    if (span <= 0) {
        return 1.0;
    }
    const double raw = span / target_ticks;
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

void ChartWidget::clear() {
    series_.clear();
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

std::pair<double, double> ChartWidget::x_range() const { return view_x_ ? *view_x_ : data_x_range(); }

void ChartWidget::reset_view() {
    view_x_.reset();
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

double ChartWidget::time_at(int px) const {
    const QRect rect = plot_rect();
    const auto [x0, x1] = x_range();
    const double fraction = (px - rect.left()) / static_cast<double>(std::max(1, rect.width()));
    return x0 + fraction * (x1 - x0);
}

double ChartWidget::value_at(int py) const {
    // height_fraction maps a height back to a value too: the same flip.
    const QRect rect = plot_rect();
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
    const QRect plot = rect.value_or(plot_rect());
    const auto [y0, y1] = y_range_;
    const double step = nice_value_step(y1 - y0);
    std::vector<std::pair<int, QString>> ticks;
    for (double value = std::ceil(y0 / step) * step; value <= y1; value += step) {
        const double fraction = height_fraction((value - y0) / std::max(1e-9, y1 - y0));
        // "+ 0.0" turns the -0.0 that ceil() gives for a range starting just
        // below zero (Butteraugli's room) into 0, not "-0".
        ticks.emplace_back(plot.bottom() - static_cast<int>(fraction * plot.height()), qt_text(format_general(value + 0.0)));
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
    const QRect plot = plot_rect();
    draw_axes(painter, plot, dark);
    for (const ChartSeries* series : visible_series()) {
        draw_series(painter, plot, *series);
    }
    painter.setPen(QPen(colours.axis, 1));
    painter.drawRect(plot.adjusted(0, 0, -1, -1));
    painter.end();
}

bool ChartWidget::cache_matches_display() const {
    // Whether the kept drawing still fits the size, pixel ratio and theme.
    const double ratio = devicePixelRatioF();
    return cache_ && cache_dark_ == theme::is_dark() && cache_->devicePixelRatioF() == ratio &&
           cache_->width() == static_cast<int>(std::ceil(width() * ratio)) &&
           cache_->height() == static_cast<int>(std::ceil(height() * ratio));
}

void ChartWidget::draw_axes(QPainter& painter, const QRect& rect, bool dark) const {
    // The grid, the axes' labels and the value axis's title.
    const QFontMetrics metrics(painter.font());
    const theme::ChartColours colours = theme::chart_colours(dark);

    painter.setPen(QPen(colours.grid, 1));
    const std::vector<std::pair<int, QString>> times = time_ticks();
    for (const auto& [px, label] : times) {
        painter.drawLine(px, rect.top(), px, rect.bottom());
    }
    const std::vector<std::pair<int, QString>> ticks = value_ticks(rect);
    for (const auto& [py, label] : ticks) {
        painter.drawLine(rect.left(), py, rect.right(), py);
    }

    painter.setPen(QPen(colours.text, 1));
    for (const auto& [px, label] : times) {
        // Centred on its tick, but kept inside the widget: a tick at the
        // right end lost half its label ("0:00:0").
        const int label_width = metrics.horizontalAdvance(label);
        const int left = std::min(px - label_width / 2, width() - label_width - 1);
        painter.drawText(left, rect.bottom() + metrics.height() + 2, label);
    }
    for (const auto& [py, label] : ticks) {
        painter.drawText(rect.left() - metrics.horizontalAdvance(label) - 6, py + metrics.ascent() / 2, label);
    }
    if (!y_axis_label_.isEmpty()) {
        painter.save();
        painter.translate(14, rect.center().y());
        painter.rotate(-90);
        // Centred as Python's (-width) // 2 centres it: rounded down.
        painter.drawText(-(metrics.horizontalAdvance(y_axis_label_) + 1) / 2, 0, y_axis_label_);
        painter.restore();
    }
}

void ChartWidget::draw_series(QPainter& painter, const QRect& rect, const ChartSeries& series) const {
    // One line: point by point when there are few, else one min/max stroke
    // per pixel column.
    const auto [x0, x1] = x_range();
    const double y0 = y_range_.first;  // not a structured binding: the lambda below takes it
    const double y1 = y_range_.second;
    const std::span<const double> times = series.times;
    const std::span<const float> values = series.values;
    const auto count_all = static_cast<std::ptrdiff_t>(times.size());
    std::ptrdiff_t lo = std::lower_bound(times.begin(), times.end(), x0) - times.begin();
    std::ptrdiff_t hi = std::upper_bound(times.begin(), times.end(), x1) - times.begin();
    lo = std::max<std::ptrdiff_t>(0, lo - 1);
    hi = std::min(count_all, hi + 1);
    if (hi - lo < 1) {
        return;
    }
    const int width = rect.width();
    painter.setPen(QPen(series.color, 1));
    const double y_span = std::max(1e-9, y1 - y0);
    const auto to_py = [&](double value) {
        return rect.bottom() - height_fraction((value - y0) / y_span) * rect.height();
    };

    if (hi - lo <= static_cast<std::ptrdiff_t>(width) * 2) {
        // Few enough points to draw honestly as a line, a finite stretch at a
        // time: joined across a missing frame, it would invent a line through
        // it, and an infinity in a QPointF is undefined. Drawn as segments,
        // not as one polyline: with display scaling, which makes the 1 px pen
        // wider than a pixel, Qt outlines a polyline as a shape and fills it,
        // 19 times slower (three lines of 720 frames: 13.7 ms against 0.73 ms
        // at 175%); the segments differ from it only by a pixel at the tip
        // of some sharp turns.
        const double x_span = std::max(1e-9, x1 - x0);
        std::vector<QLineF> segments;
        segments.reserve(static_cast<std::size_t>(hi - lo));
        std::optional<QPointF> previous;  // the stretch's last point
        bool joined = false;              // whether it is in a segment
        const auto end_stretch = [&painter, &previous, &joined] {
            if (previous && !joined) {
                painter.drawPoint(*previous);  // a stretch of one frame
            }
            previous.reset();
            joined = false;
        };
        for (std::ptrdiff_t i = lo; i < hi; ++i) {
            const auto value = static_cast<double>(values[static_cast<std::size_t>(i)]);
            if (!std::isfinite(value)) {
                end_stretch();
                continue;
            }
            const QPointF point(rect.left() + (times[static_cast<std::size_t>(i)] - x0) / x_span * width, to_py(value));
            if (previous) {
                segments.emplace_back(*previous, point);
            }
            joined = previous.has_value();
            previous = point;
        }
        end_stretch();
        painter.drawLines(segments.data(), static_cast<int>(segments.size()));
        return;
    }

    // More frames than pixels: one min/max pair per pixel column, so that
    // drawing costs what the width does and no dip is lost. The columns'
    // first frames are numpy's linspace(lo, hi, width + 1), truncated; a
    // column whose first frame is the one before's has none of its own. An
    // infinity counts as a gap, not as a huge value that would hide the line.
    const double step = static_cast<double>(hi - lo) / width;
    std::vector<std::pair<int, std::ptrdiff_t>> columns;  // (column, its first frame)
    std::ptrdiff_t previous = -1;
    for (int column = 0; column < width; ++column) {
        const auto first = static_cast<std::ptrdiff_t>(column * step + static_cast<double>(lo));
        if (column == 0 || first > previous) {
            columns.emplace_back(column, first);
        }
        previous = first;
    }
    std::vector<QLineF> lines;
    lines.reserve(columns.size());
    for (std::size_t j = 0; j < columns.size(); ++j) {
        const std::ptrdiff_t end = j + 1 < columns.size() ? columns[j + 1].second : hi;
        float low = std::numeric_limits<float>::infinity();
        float high = -std::numeric_limits<float>::infinity();
        for (std::ptrdiff_t i = columns[j].second; i < end; ++i) {
            const float value = values[static_cast<std::size_t>(i)];
            if (std::isfinite(value)) {
                low = std::min(low, value);
                high = std::max(high, value);
            }
        }
        if (!std::isfinite(low) || !std::isfinite(high)) {
            continue;
        }
        const double x = rect.left() + columns[j].first;
        lines.emplace_back(x, to_py(high), x, to_py(low));
    }
    painter.drawLines(lines.data(), static_cast<int>(lines.size()));
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
        const double span = pan_view_->second - pan_view_->first;
        const double shift =
            (pan_origin_->x() - pos.x()) / static_cast<double>(std::max(1, plot_rect().width())) * span;
        view_x_ = std::pair(pan_view_->first + shift, pan_view_->second + shift);
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
        pan_view_ = x_range();
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
    const double steps = event->angleDelta().y() / 120.0;
    if (steps == 0) {
        return;
    }
    const double factor = std::pow(0.8, steps);
    const auto [x0, x1] = x_range();
    const double anchor = time_at(event->position().toPoint().x());
    double new_span = (x1 - x0) * factor;
    const auto [data_lo, data_hi] = data_x_range();
    if (new_span >= data_hi - data_lo) {
        reset_view();
        return;
    }
    new_span = std::max(new_span, seconds_per_pixel() * 8);  // not past a few pixels of data
    const double left_share = (anchor - x0) / std::max(1e-9, x1 - x0);
    double new_x0 = anchor - left_share * new_span;
    double new_x1 = new_x0 + new_span;
    // Over the data, not drifting off its ends.
    if (new_x0 < data_lo) {
        new_x0 = data_lo;
        new_x1 = data_lo + new_span;
    }
    if (new_x1 > data_hi) {
        new_x0 = data_hi - new_span;
        new_x1 = data_hi;
    }
    view_x_ = std::pair(new_x0, new_x1);
    invalidate();
    emit view_changed();
}

QPixmap ChartWidget::render_to_pixmap() const {
    QPixmap exported = render(false);
    exported.setDevicePixelRatio(1.0);  // what export compositors lay out by
    return exported;
}

}  // namespace vml::app
