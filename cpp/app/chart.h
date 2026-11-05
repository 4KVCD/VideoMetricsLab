// A line chart made for per-frame metric scores (the Python app's
// vmaf_app/ui/chart.py). It is drawn by hand, as there:
// - each line is reduced to one min/max pair per column of about a pixel of
//   the mouse (a whole number of screen pixels) before it is drawn, so the
//   cost of drawing depends on the window's width, not the video's length,
//   and no dip is lost; the columns are fixed to the time axis, so a drag
//   moves the lines without changing them;
// - the drawing is kept in a pixmap, so moving the crosshair repaints only
//   the few pixel columns that changed.
#pragma once

#include <QColor>
#include <QElapsedTimer>
#include <QPixmap>
#include <QPoint>
#include <QRect>
#include <QString>
#include <QWidget>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

class QTimer;

namespace vml::app {

// One line: its times (seconds, ascending) and scores, as many of each,
// which `owner` keeps alive (they are usually a result's own arrays, not a
// copy); its colour, and whether it is shown.
struct ChartSeries {
    std::span<const double> times;
    std::span<const float> values;
    std::shared_ptr<const void> owner;
    QColor color;
    bool visible = true;
    // Empty, or a value for each point: a dot on it in mark_colors[value], none for 0.
    std::span<const std::uint8_t> marks = {};
    std::span<const QColor> mark_colors = {};
};

// A line drawn in a band of its own across the bottom of the chart, under
// the others, on the same time axis with its own value axis: Metric Graphs'
// difference of two lines. Its colour is the theme's (ChartColours), not
// `series.color`.
struct ChartBand {
    ChartSeries series;
    QString label;  // drawn at the band's top left
    // Half the value range when every value is 0: a unit of the scores'
    // precision, so that the axis reads as the scores do.
    double flat_half_range = 1.0;
};

// A tick spacing that reads naturally on a time axis, giving about eight
// ticks across `span` seconds.
double nice_time_step(double span);
// A 1, 2 or 5 times a power of ten spacing for a value axis, giving about
// `ticks` ticks across `span`.
double nice_value_step(double span, double ticks = 8);

class ChartWidget : public QWidget {
    Q_OBJECT

public:
    // `fixed_y_max`: the top of a bounded scale (VMAF's 100). `invert_y`:
    // the lowest value at the top, for a metric where lower is better
    // (Butteraugli), so that up is better on every graph.
    explicit ChartWidget(QString y_axis_label = {}, std::optional<double> fixed_y_max = std::nullopt,
                         bool invert_y = false, QWidget* parent = nullptr);

    // Adds or replaces a line.
    void set_series(int key, ChartSeries series);
    // Draws a line in another colour, without fitting the axes again.
    void set_series_color(int key, const QColor& color);
    // The value axis' name, for a chart whose lines change what they show
    // (the Bitrate tab's frame sizes or bitrates).
    void set_y_axis_label(const QString& label);
    const QString& y_axis_label() const { return y_axis_label_; }
    // Removes lines, fitting the axes again once however many go.
    void remove_series(const std::vector<int>& keys);
    // Gives the lines the new keys `renumbered` names (old to new), and so
    // the new keys' order: the last is drawn over the others.
    void renumber(const std::map<int, int>& renumbered);
    void set_series_visible(int key, bool visible);
    // Draws `band` under the lines (nothing: no band). A band without a
    // finite value is not drawn.
    void set_band(std::optional<ChartBand> band);
    bool has_band() const { return band_.has_value(); }
    // The band's value axis's range: from 0 or its lowest value to 0 or its
    // highest, with a little room.
    std::pair<double, double> band_range() const { return band_range_; }
    void clear();
    // Whether any shown line has points.
    bool has_data() const;

    // The time range of the shown lines; the time range in view (zoomed, or
    // all of it); the value axis's range.
    std::pair<double, double> data_x_range() const;
    void forget_view_off_the_data();
    std::pair<double, double> x_range() const;
    std::pair<double, double> y_range() const { return y_range_; }
    // Shows all of the time range again.
    void reset_view();

    // The plotting area, inside the axes, which the time axis, the pointer
    // and the crosshair span; with a band, its upper part, where the lines
    // are drawn, and the band below it; without one, the lines' is all of it.
    QRect plot_rect() const;
    QRect lines_rect() const;
    std::optional<QRect> band_rect() const;
    double time_at(int px) const;
    // The lines' value at row `py`. Over the band (and the gap above it):
    // the end of their scale every point is at or below, so that the
    // pointer there picks the frame nearest in time.
    double value_at(int py) const;
    int pixel_for_time(double t) const;
    // The grid lines drawn: the time axis's as (pixel column, label), in
    // order, and the lines' value axis's (in `rect`, lines_rect if none) and
    // the band's as (pixel row, label), lowest value first.
    std::vector<std::pair<int, QString>> time_ticks() const;
    std::vector<std::pair<int, QString>> value_ticks(std::optional<QRect> rect = std::nullopt) const;
    std::vector<std::pair<int, QString>> band_ticks() const;
    double seconds_per_pixel() const;

    // Moves the crosshair (nothing hides it), repainting only the columns it
    // leaves and the ones it takes.
    void set_cursor_time(std::optional<double> t);

    // The chart for a PNG export: in the light theme's colours whatever the
    // window shows, as an exported file is read elsewhere.
    QPixmap render_to_pixmap() const;

signals:
    // The time and value under the pointer.
    void hovered(double time, double value);
    // The pointer left the plotting area.
    void left();
    // Zoomed or panned.
    void view_changed();

protected:
    void resizeEvent(QResizeEvent* event) override;
    void changeEvent(QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    friend class ChartTest;         // the chart's tests (tests/app)
    friend class GraphPanelTest;    // and the graph's
    friend class BitratePanelTest;  // and the Bitrate tab's

    // The time range in view, as where it starts and how long it is: a drag
    // moves the start alone, so that the seconds a pixel holds, on which
    // draw_series' grid of pixels rests, stay exactly the same (an end moved
    // with the start would change the length by a rounding).
    struct View {
        double start = 0.0;
        double span = 1.0;
    };
    View current_view() const;  // view_, or all of the data
    std::vector<const ChartSeries*> visible_series() const;
    void recompute_y_range();
    double height_fraction(double value_fraction) const { return invert_y_ ? 1.0 - value_fraction : value_fraction; }
    void invalidate(bool refit = false);
    void rebuild_cache();
    QPixmap render(bool dark) const;
    void draw(QPixmap& pixmap, bool dark) const;
    bool cache_matches_display() const;
    void draw_axes(QPainter& painter, bool dark) const;
    void draw_band(QPainter& painter, const QRect& rect, bool dark) const;
    // Ticks for `range` drawn in `rect`, about `count` of them.
    std::vector<std::pair<int, QString>> ticks_for(const QRect& rect, std::pair<double, double> range,
                                                   double count) const;
    void draw_series(QPainter& painter, const QRect& rect, const ChartSeries& series,
                     std::pair<double, double> range) const;
    void report_hover(QPoint pos);
    bool waiting_for_hover_frame() const;
    void ask_for_hover_frame();
    void on_hover_frame();
    void forget_hover();
    void request_paint();
    int frame_ms() const;

    QString y_axis_label_;
    std::optional<double> fixed_y_max_;
    bool invert_y_ = false;
    std::map<int, ChartSeries> series_;  // by key; keys rise as lines are added
    std::optional<ChartBand> band_;
    std::pair<double, double> band_range_{-1.0, 1.0};  // fitted once, when the band is set
    std::optional<QPixmap> cache_;
    bool cache_stale_ = true;  // to be drawn again before it is shown
    bool cache_dark_ = false;  // the theme the cache was drawn in
    int renders_ = 0;          // how many times the cache has been drawn (for the tests)
    std::optional<int> cursor_x_;
    std::optional<View> view_;  // nothing: all of the data
    std::pair<double, double> y_range_{0.0, 1.0};
    std::optional<QPoint> pan_origin_;
    std::optional<View> pan_view_;
    // While the size keeps changing (the window dragged by an edge), the last
    // drawing is shown stretched, and drawn again once the size settles.
    // Drawn at every size, a dozen two-hour lines took 54 ms a step.
    QTimer* resize_settle_ = nullptr;
    // The pointer is reported once per frame of the screen (ask_for_hover_frame):
    // the last move since, when this timer, started as the frame starts, is up.
    QTimer* hover_pace_ = nullptr;
    std::optional<QPoint> pending_hover_;
    bool hover_frame_asked_ = false;
    QElapsedTimer hover_frame_asked_at_;
    unsigned hover_frame_request_ = 0;  // the frame asked for last, of those still to come
    // The chart is drawn again at most once per frame too (request_paint).
    QTimer* paint_pace_ = nullptr;
    bool paint_due_ = false;
};

}  // namespace vml::app
