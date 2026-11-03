#include "rate_curve_chart.h"

#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>
#include <limits>

#include "chart.h"
#include "i18n.h"
#include "theme.h"

namespace vml::app {

namespace {

// Room around the plotting area for the axes' labels, the legend and the
// BD-rate label.
constexpr int margin_left = 58;
constexpr int margin_right = 16;
constexpr int margin_top = 30;
constexpr int margin_bottom = 40;
constexpr double dot_radius = 4.0;

// The lowest and highest of `values`, widened by `pad` of their span on each
// side (a span of nothing by `flat` either way).
std::pair<double, double> padded_range(const std::vector<double>& values, double pad, double flat) {
    const auto [lowest, highest] = std::ranges::minmax(values);
    const double span = highest - lowest;
    if (span <= 0) {
        return {lowest - flat, highest + flat};
    }
    return {lowest - span * pad, highest + span * pad};
}

// A rate as an axis label: "500", "2k", "1.5k", "20k".
QString rate_label(double kbps) {
    return kbps >= 1000 ? QString::number(kbps / 1000, 'g', 3) + QStringLiteral("k") : QString::number(kbps, 'g', 3);
}

// The rates at 1, 2 and 5 times each power of ten within [lo, hi] (log10).
std::vector<double> rate_ticks(double lo, double hi) {
    std::vector<double> ticks;
    for (int decade = static_cast<int>(std::floor(lo)); decade <= static_cast<int>(std::ceil(hi)); ++decade) {
        for (const double step : {1.0, 2.0, 5.0}) {
            const double kbps = step * std::pow(10.0, decade);
            if (std::log10(kbps) >= lo && std::log10(kbps) <= hi) {
                ticks.push_back(kbps);
            }
        }
    }
    return ticks;
}

// The decimals that tell ticks `step` apart: 0 for 5, 1 for 0.5, 2 for 0.02.
int decimals_for(double step) { return std::max(0, static_cast<int>(std::ceil(-std::log10(step) - 1e-9))); }

// The chart's label, bold, at the top left above the plotting area.
void draw_label(QPainter& painter, const QRect& area, const QString& label) {
    QFont bold = painter.font();
    bold.setBold(true);
    painter.save();
    painter.setFont(bold);
    painter.setPen(theme::chart_colours().text);
    const int text_height = painter.fontMetrics().height();
    painter.drawText(QRect(area.left(), 4, area.width() / 2, text_height + 4), Qt::AlignLeft | Qt::AlignVCenter,
                     label);
    painter.restore();
}

}  // namespace

RateCurveChart::RateCurveChart(QWidget* parent) : QWidget(parent) { setMinimumHeight(220); }

void RateCurveChart::set_plot(std::optional<Plot> plot, const QString& hint) {
    plot_ = std::move(plot);
    hint_ = hint;
    update();
}

std::optional<std::pair<double, double>> RateCurveChart::shared_scores() const {
    if (!plot_ || plot_->anchor.line.empty() || plot_->test.line.empty()) {
        return std::nullopt;
    }
    const auto range_of = [](const std::vector<RatePoint>& line) {
        const auto [lowest, highest] = std::ranges::minmax(line, {}, &RatePoint::score);
        return std::pair{lowest.score, highest.score};
    };
    const auto [anchor_low, anchor_high] = range_of(plot_->anchor.line);
    const auto [test_low, test_high] = range_of(plot_->test.line);
    const double low = std::max(anchor_low, test_low);
    const double high = std::min(anchor_high, test_high);
    return high > low ? std::optional(std::pair{low, high}) : std::nullopt;
}

void RateCurveChart::paintEvent(QPaintEvent*) {
    const theme::ChartColours colours = theme::chart_colours();
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), colours.background);
    const QRect area = rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom);
    if (!plot_ || area.width() < 40 || area.height() < 40) {
        painter.setPen(theme::color("muted"));
        painter.drawText(rect(), Qt::AlignCenter | Qt::TextWordWrap, hint_);
        return;
    }
    const Plot& plot = *plot_;

    // The axes' ranges, from every dot and every curve's point
    std::vector<double> log_rates;
    std::vector<double> scores;
    for (const Curve* curve : {&plot.anchor, &plot.test}) {
        for (const std::vector<RatePoint>* points : {&curve->points, &curve->line}) {
            for (const RatePoint& point : *points) {
                log_rates.push_back(std::log10(point.kbps));
                scores.push_back(point.score);
            }
        }
    }
    if (log_rates.empty()) {
        draw_label(painter, area, plot.label);  // the n/a and why: there is no encode to draw
        return;
    }
    const auto [x_low, x_high] = padded_range(log_rates, 0.05, 0.1);
    const auto [y_low, y_high] = padded_range(scores, 0.08, 1.0);
    const auto x_of = [&](double kbps) {
        return area.left() + (std::log10(kbps) - x_low) / (x_high - x_low) * area.width();
    };
    const auto y_of = [&](double score) {
        double fraction = (score - y_low) / (y_high - y_low);
        if (plot.lower_is_better) {
            fraction = 1 - fraction;  // up is better on every chart
        }
        return area.bottom() - fraction * area.height();
    };

    // The scores both curves cover, shaded across the plot
    const QColor test_colour = theme::color("link");
    if (const std::optional<std::pair<double, double>> shared = shared_scores()) {
        QColor shade = test_colour;
        shade.setAlpha(36);
        const double top = std::min(y_of(shared->first), y_of(shared->second));
        const double bottom = std::max(y_of(shared->first), y_of(shared->second));
        painter.fillRect(QRectF(area.left(), top, area.width(), bottom - top), shade);
    }

    // The grid, its labels and the axes' names
    painter.setPen(colours.grid);
    const double score_step = nice_value_step(y_high - y_low, 6);
    const int score_decimals = decimals_for(score_step);
    std::vector<std::pair<double, QString>> score_ticks;
    for (double score = std::ceil(y_low / score_step) * score_step; score <= y_high; score += score_step) {
        score_ticks.emplace_back(score, QString::number(score, 'f', score_decimals));
    }
    const std::vector<double> kbps_ticks = rate_ticks(x_low, x_high);
    for (const auto& [score, _] : score_ticks) {
        painter.drawLine(QPointF(area.left(), y_of(score)), QPointF(area.right(), y_of(score)));
    }
    for (const double kbps : kbps_ticks) {
        painter.drawLine(QPointF(x_of(kbps), area.top()), QPointF(x_of(kbps), area.bottom()));
    }
    painter.setPen(colours.axis);
    painter.drawRect(area);
    painter.setPen(colours.text);
    const int text_height = painter.fontMetrics().height();
    for (const auto& [score, label] : score_ticks) {
        painter.drawText(QRectF(0, y_of(score) - text_height / 2.0, margin_left - 6, text_height),
                         Qt::AlignRight | Qt::AlignVCenter, label);
    }
    for (const double kbps : kbps_ticks) {
        painter.drawText(QRectF(x_of(kbps) - 30, area.bottom() + 3, 60, text_height), Qt::AlignCenter,
                         rate_label(kbps));
    }
    painter.drawText(QRect(area.left(), area.bottom() + 3 + text_height, area.width(), text_height), Qt::AlignCenter,
                     app::tr("kb/s (log scale)"));
    painter.save();
    painter.translate(12, area.center().y());
    painter.rotate(-90);
    painter.drawText(QRect(-area.height() / 2, -text_height / 2, area.height(), text_height), Qt::AlignCenter,
                     plot.metric_label);
    painter.restore();

    // The curves and their dots: the anchor's dashed and grey, the test's
    // solid in the accent colour
    const QColor anchor_colour = theme::color("muted");
    painter.setClipRect(area);
    for (const auto& [curve, colour, style] : {std::tuple{&plot.anchor, anchor_colour, Qt::DashLine},
                                               std::tuple{&plot.test, test_colour, Qt::SolidLine}}) {
        if (curve->line.size() > 1) {
            QPainterPath path;
            path.moveTo(x_of(curve->line.front().kbps), y_of(curve->line.front().score));
            for (const RatePoint& point : curve->line) {
                path.lineTo(x_of(point.kbps), y_of(point.score));
            }
            painter.setPen(QPen(colour, 2, style, Qt::RoundCap, Qt::RoundJoin));
            painter.setBrush(Qt::NoBrush);
            painter.drawPath(path);
        }
        painter.setPen(QPen(colours.background, 1.5));
        painter.setBrush(colour);
        for (const RatePoint& point : curve->points) {
            painter.drawEllipse(QPointF(x_of(point.kbps), y_of(point.score)), dot_radius, dot_radius);
        }
    }
    painter.setClipping(false);

    // The BD-rate at the top left, the legend at the top right
    draw_label(painter, area, plot.label);
    int right = area.right();
    for (const auto& [curve, colour, style] : {std::tuple{&plot.test, test_colour, Qt::SolidLine},
                                               std::tuple{&plot.anchor, anchor_colour, Qt::DashLine}}) {
        const int width = painter.fontMetrics().horizontalAdvance(curve->name);
        const int text_left = right - width;
        painter.setPen(colours.text);
        painter.drawText(QRect(text_left, 4, width, text_height + 4), Qt::AlignVCenter, curve->name);
        painter.setPen(QPen(colour, 2, style, Qt::RoundCap));
        const int y = 4 + (text_height + 4) / 2;
        painter.drawLine(text_left - 26, y, text_left - 6, y);
        right = text_left - 40;
    }
}

}  // namespace vml::app
