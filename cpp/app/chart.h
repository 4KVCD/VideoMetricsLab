// A line chart made for per-frame metric scores (the Python app's
// vmaf_app/ui/chart.py). It is drawn by hand, as there:
// - each line is reduced to one min/max pair per pixel column before it is
//   drawn, so the cost of drawing depends on the window's width, not the
//   video's length, and no dip is lost;
// - the drawing is kept in a pixmap, so moving the crosshair repaints only
//   the few pixel columns that changed.
#pragma once

#include <QColor>
#include <QPixmap>
#include <QPoint>
#include <QRect>
#include <QString>
#include <QWidget>

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
};

// A tick spacing that reads naturally on a time axis, giving about eight
// ticks across `span` seconds.
double nice_time_step(double span);
// A 1, 2 or 5 times a power of ten spacing for a value axis, giving about
// eight ticks across `span`.
double nice_value_step(double span);

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
    void clear();
    // Whether any shown line has points.
    bool has_data() const;

    // The time range of the shown lines; the time range in view (zoomed, or
    // all of it); the value axis's range.
    std::pair<double, double> data_x_range() const;
    std::pair<double, double> x_range() const;
    std::pair<double, double> y_range() const { return y_range_; }
    // Shows all of the time range again.
    void reset_view();

    // Where the lines are drawn, inside the axes.
    QRect plot_rect() const;
    double time_at(int px) const;
    double value_at(int py) const;
    int pixel_for_time(double t) const;
    // The grid lines drawn: the time axis's as (pixel column, label), in
    // order, and the value axis's as (pixel row, label), lowest value first.
    std::vector<std::pair<int, QString>> time_ticks() const;
    std::vector<std::pair<int, QString>> value_ticks(std::optional<QRect> rect = std::nullopt) const;
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

    std::vector<const ChartSeries*> visible_series() const;
    void recompute_y_range();
    double height_fraction(double value_fraction) const { return invert_y_ ? 1.0 - value_fraction : value_fraction; }
    void invalidate(bool refit = false);
    void rebuild_cache();
    QPixmap render(bool dark) const;
    void draw(QPixmap& pixmap, bool dark) const;
    bool cache_matches_display() const;
    void draw_axes(QPainter& painter, const QRect& rect, bool dark) const;
    void draw_series(QPainter& painter, const QRect& rect, const ChartSeries& series) const;
    void report_hover(QPoint pos);
    void forget_hover();
    void request_paint();
    int frame_ms() const;

    QString y_axis_label_;
    std::optional<double> fixed_y_max_;
    bool invert_y_ = false;
    std::map<int, ChartSeries> series_;  // by key; keys rise as lines are added
    std::optional<QPixmap> cache_;
    bool cache_stale_ = true;  // to be drawn again before it is shown
    bool cache_dark_ = false;  // the theme the cache was drawn in
    int renders_ = 0;          // how many times the cache has been drawn (for the tests)
    std::optional<int> cursor_x_;
    std::optional<std::pair<double, double>> view_x_;  // nothing: all of the data
    std::pair<double, double> y_range_{0.0, 1.0};
    std::optional<QPoint> pan_origin_;
    std::optional<std::pair<double, double>> pan_view_;
    // While the size keeps changing (the window dragged by an edge), the last
    // drawing is shown stretched, and drawn again once the size settles.
    // Drawn at every size, a dozen two-hour lines took 54 ms a step.
    QTimer* resize_settle_ = nullptr;
    // The pointer is reported at most once per frame of the screen
    // (report_hover): the last move since, when the frame's time is up.
    QTimer* hover_pace_ = nullptr;
    std::optional<QPoint> pending_hover_;
    // The chart is drawn again at most once per frame too (request_paint).
    QTimer* paint_pace_ = nullptr;
    bool paint_due_ = false;
};

}  // namespace vml::app
