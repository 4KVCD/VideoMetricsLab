// The BD-rate tab's chart: one sequence's rate curves for one metric, the
// anchor's and the test's, as the BD-rate sees them. Bitrate on a log axis
// across, the score up (lower is better turned over, so up is better), a dot
// for each encode, the interpolated curve through them, and the scores both
// curves cover shaded: the range the BD-rate averages over. Drawn by hand;
// the window's ChartWidget is made for long per-frame lines on a time axis.
#pragma once

#include <QString>
#include <QWidget>

#include <optional>
#include <vector>

#include "vml/bd_rate.h"

namespace vml::app {

class RateCurveChart : public QWidget {
    Q_OBJECT

public:
    explicit RateCurveChart(QWidget* parent = nullptr);

    // One encoder's encodes for the metric: its name, its points (each with
    // a rate above 0 and finite numbers), and the curve through them (empty
    // where there is none, as with tied scores: the dots alone are drawn).
    struct Curve {
        QString name;
        std::vector<RatePoint> points;
        std::vector<RatePoint> line;
    };
    // What to draw: the two curves, the metric's name and direction, and the
    // label at the top left (the BD-rate). Nothing: an empty chart with
    // `hint` in it.
    struct Plot {
        Curve anchor;
        Curve test;
        QString metric_label;
        bool lower_is_better = false;
        QString label;
    };
    void set_plot(std::optional<Plot> plot, const QString& hint = {});
    const std::optional<Plot>& plot() const { return plot_; }

    // The shaded range of scores, both curves', when they overlap.
    std::optional<std::pair<double, double>> shared_scores() const;

    QSize sizeHint() const override { return {640, 300}; }

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::optional<Plot> plot_;
    QString hint_;
};

}  // namespace vml::app
