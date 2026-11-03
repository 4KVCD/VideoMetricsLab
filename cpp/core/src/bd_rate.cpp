#include "vml/bd_rate.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "vml/translatable.h"

namespace vml {

namespace {

int sign(double value) { return (value > 0) - (value < 0); }

// An end point's slope: Moler's three-point formula (Numerical Computing with
// MATLAB, pchiptx.m), as scipy's _edge_case: 0 where it would turn against the
// first secant, and at most three times that secant where the two secants
// differ in sign, so that the curve does not overshoot.
double edge_slope(double h0, double h1, double m0, double m1) {
    const double slope = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (sign(slope) != sign(m0)) {
        return 0.0;
    }
    if (sign(m0) != sign(m1) && std::abs(slope) > 3 * std::abs(m0)) {
        return 3 * m0;
    }
    return slope;
}

// The integral of the Hermite cubic of one interval (width h, values y0 and
// y1, slopes d0 and d1) from its start to the fraction s of it: the integrals
// of the four Hermite basis functions, weighted.
double hermite_integral(double h, double y0, double y1, double d0, double d1, double s) {
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;
    const double h00 = s4 / 2 - s3 + s;
    const double h10 = s4 / 4 - 2 * s3 / 3 + s2 / 2;
    const double h01 = -s4 / 2 + s3;
    const double h11 = s4 / 4 - s3 / 3;
    return h * (y0 * h00 + h * d0 * h10 + y1 * h01 + h * d1 * h11);
}

// A curve as PCHIP takes it: the scores ascending (x) and the log rates (y).
struct Curve {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> slopes;
};

// The curve of `points`, or why there is none.
BdRateGap make_curve(std::span<const RatePoint> points, bool lower_is_better, Curve& curve) {
    if (points.size() < 2) {
        return BdRateGap::too_few_points;
    }
    std::vector<RatePoint> sorted;
    for (const RatePoint& point : points) {
        if (!std::isfinite(point.score) || !std::isfinite(point.kbps) || point.kbps <= 0) {
            return BdRateGap::bad_number;
        }
        sorted.push_back({point.kbps, lower_is_better ? -point.score : point.score});
    }
    std::ranges::stable_sort(sorted, {}, &RatePoint::score);
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        if (index > 0 && sorted[index].score <= sorted[index - 1].score) {
            return BdRateGap::tied_scores;
        }
        curve.x.push_back(sorted[index].score);
        curve.y.push_back(std::log(sorted[index].kbps));
    }
    curve.slopes = pchip_slopes(curve.x, curve.y);
    return BdRateGap::none;
}

// The curve's integral from `lo` to `hi`, both within its scores.
double integral(const Curve& curve, double lo, double hi) {
    double total = 0.0;
    for (std::size_t index = 0; index + 1 < curve.x.size(); ++index) {
        const double start = curve.x[index];
        const double end = curve.x[index + 1];
        const double from = std::max(lo, start);
        const double to = std::min(hi, end);
        if (to <= from) {
            continue;
        }
        const double h = end - start;
        const auto part = [&](double at) {
            return hermite_integral(h, curve.y[index], curve.y[index + 1], curve.slopes[index],
                                    curve.slopes[index + 1], (at - start) / h);
        };
        total += part(to) - part(from);
    }
    return total;
}

}  // namespace

std::vector<double> pchip_slopes(std::span<const double> x, std::span<const double> y) {
    const std::size_t count = x.size();
    std::vector<double> widths(count - 1);
    std::vector<double> secants(count - 1);
    for (std::size_t index = 0; index + 1 < count; ++index) {
        widths[index] = x[index + 1] - x[index];
        secants[index] = (y[index + 1] - y[index]) / widths[index];
    }
    if (count == 2) {
        return {secants[0], secants[0]};  // a straight line
    }
    std::vector<double> slopes(count, 0.0);
    // Inside: the weighted harmonic mean of the two secants, 0 where they
    // differ in sign or one is flat (a peak or plateau stays one)
    for (std::size_t index = 1; index + 1 < count; ++index) {
        const double before = secants[index - 1];
        const double after = secants[index];
        if (sign(before) != sign(after) || before == 0 || after == 0) {
            continue;
        }
        const double w1 = 2 * widths[index] + widths[index - 1];
        const double w2 = widths[index] + 2 * widths[index - 1];
        slopes[index] = (w1 + w2) / (w1 / before + w2 / after);
    }
    slopes[0] = edge_slope(widths[0], widths[1], secants[0], secants[1]);
    slopes[count - 1] =
        edge_slope(widths[count - 2], widths[count - 3], secants[count - 2], secants[count - 3]);
    return slopes;
}

BdRateResult bd_rate(std::span<const RatePoint> anchor, std::span<const RatePoint> test, bool lower_is_better) {
    Curve anchor_curve;
    Curve test_curve;
    for (const auto& [points, curve] : {std::pair{anchor, &anchor_curve}, std::pair{test, &test_curve}}) {
        if (const BdRateGap gap = make_curve(points, lower_is_better, *curve); gap != BdRateGap::none) {
            return {std::nullopt, gap};
        }
    }
    const double lo = std::max(anchor_curve.x.front(), test_curve.x.front());
    const double hi = std::min(anchor_curve.x.back(), test_curve.x.back());
    if (hi <= lo) {
        return {std::nullopt, BdRateGap::no_overlap};
    }
    const double mean_log_ratio = (integral(test_curve, lo, hi) - integral(anchor_curve, lo, hi)) / (hi - lo);
    return {(std::exp(mean_log_ratio) - 1) * 100, BdRateGap::none};
}

std::vector<RatePoint> pchip_curve(std::span<const RatePoint> points, bool lower_is_better, int samples) {
    Curve curve;
    if (make_curve(points, lower_is_better, curve) != BdRateGap::none || samples < 2) {
        return {};
    }
    std::vector<RatePoint> line;
    std::size_t index = 0;
    for (int sample = 0; sample < samples; ++sample) {
        const double fraction = static_cast<double>(sample) / (samples - 1);
        const double x = curve.x.front() + fraction * (curve.x.back() - curve.x.front());
        while (index + 2 < curve.x.size() && x > curve.x[index + 1]) {
            ++index;
        }
        // The interval's Hermite cubic at x: the four basis functions, weighted
        const double h = curve.x[index + 1] - curve.x[index];
        const double t = (x - curve.x[index]) / h;
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double log_rate = (2 * t3 - 3 * t2 + 1) * curve.y[index] + (t3 - 2 * t2 + t) * h * curve.slopes[index] +
                                (-2 * t3 + 3 * t2) * curve.y[index + 1] + (t3 - t2) * h * curve.slopes[index + 1];
        line.push_back({std::exp(log_rate), lower_is_better ? -x : x});
    }
    return line;
}

std::string bd_rate_gap_text(BdRateGap gap) {
    switch (gap) {
        case BdRateGap::none: return {};
        case BdRateGap::too_few_points: return N_("fewer than 2 points");
        case BdRateGap::tied_scores: return N_("scores not strictly increasing");
        case BdRateGap::no_overlap: return N_("the score ranges do not overlap");
        case BdRateGap::bad_number: return N_("a rate not above 0, or a number not finite");
    }
    return {};
}

std::optional<double> mean_of(const std::vector<std::optional<double>>& values) {
    double total = 0.0;
    std::size_t count = 0;
    for (const std::optional<double>& value : values) {
        if (value) {
            total += *value;
            ++count;
        }
    }
    return count == 0 ? std::nullopt : std::optional(total / static_cast<double>(count));
}

std::optional<double> median_of(const std::vector<std::optional<double>>& values) {
    std::vector<double> present;
    for (const std::optional<double>& value : values) {
        if (value) {
            present.push_back(*value);
        }
    }
    if (present.empty()) {
        return std::nullopt;
    }
    std::ranges::sort(present);
    const std::size_t middle = present.size() / 2;
    return present.size() % 2 == 1 ? present[middle] : (present[middle - 1] + present[middle]) / 2;
}

}  // namespace vml
