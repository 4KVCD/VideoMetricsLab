#include "vml/stats.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>

namespace vml {

namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();

// numpy's pairwise_sum for float64 (numpy/_core/src/umath/loops_utils.h.src),
// step for step: fewer than 8 values one by one; up to 128 in eight running
// sums added in pairs; more split in two at a multiple of 8.
double pairwise_sum(std::span<const double> values) {
    const std::size_t n = values.size();
    if (n < 8) {
        double sum = 0.0;
        for (const double value : values) {
            sum += value;
        }
        return sum;
    }
    if (n <= 128) {
        // Eight running sums, the j-th over the values at j, j + 8, j + 16...
        std::array<double, 8> sums{};
        std::copy_n(values.begin(), 8, sums.begin());
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                sums[j] += values[i + j];
            }
        }
        double sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for (; i < n; ++i) {  // the last n % 8 values, one by one
            sum += values[i];
        }
        return sum;
    }
    std::size_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(values.first(half)) + pairwise_sum(values.subspan(half));
}

// np.mean.
double numpy_mean(std::span<const double> values) {
    return numpy_sum(values) / static_cast<double>(values.size());
}

// np.std: the population standard deviation, in numpy's two passes.
double numpy_std(std::span<const double> values) {
    const double mean = numpy_mean(values);
    std::vector<double> squares(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double deviation = values[i] - mean;
        squares[i] = deviation * deviation;
    }
    return std::sqrt(numpy_mean(squares));
}

// np.median of sorted values: the middle one, or the mean of the middle two.
double numpy_median(std::span<const double> ordered) {
    const std::size_t middle = ordered.size() / 2;
    return ordered.size() % 2 == 1 ? numpy_mean(ordered.subspan(middle, 1)) : numpy_mean(ordered.subspan(middle - 1, 2));
}

// The q-th percentile of sorted values, interpolated linearly as numpy does.
double percentile(std::span<const double> ordered, double q) {
    const double position = static_cast<double>(ordered.size() - 1) * q / 100.0;
    const double lower_position = std::floor(position);
    const auto lower = static_cast<std::size_t>(lower_position);
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double a = ordered[lower];
    const double b = ordered[upper];
    if (lower == upper || a == b) {
        return a;
    }
    // Between a finite value and an infinity, every point in between is
    // infinite. Between opposite infinities, it is undefined.
    if (std::isinf(a) || std::isinf(b)) {
        if (a == -infinity && b == infinity) {
            return not_a_number;
        }
        return std::isinf(a) ? a : b;
    }
    return a + (b - a) * (position - lower_position);
}

// XPSNR's average over the video from per-frame decibels:
// -20 * log10(mean(10^(-xpsnr / 20))), FFmpeg's own average written in the
// per-frame values (stats.py explains the derivation). An identical frame
// (+inf) adds 0 to the sum and 1 to the count, as in FFmpeg.
double square_mean_root_db(std::span<const double> decibels) {
    std::vector<double> distortion(decibels.size());
    for (std::size_t i = 0; i < decibels.size(); ++i) {
        distortion[i] = std::pow(10.0, -decibels[i] / 20.0);
    }
    const double mean = numpy_mean(distortion);
    // With every frame identical there is no error to average, and infinity
    // is FFmpeg's answer too.
    return mean <= 0.0 ? infinity : -20.0 * std::log10(mean);
}

// aggregate_scores of scores that have no NaN, at least one of them.
double aggregate(std::span<const double> data, MetricAggregation aggregation) {
    return aggregation == MetricAggregation::square_mean_root_db ? square_mean_root_db(data) : numpy_mean(data);
}

std::vector<double> without_nan(std::span<const double> values) {
    std::vector<double> kept;
    kept.reserve(values.size());
    std::copy_if(values.begin(), values.end(), std::back_inserter(kept), [](double v) { return !std::isnan(v); });
    return kept;
}

}  // namespace

double numpy_sum(std::span<const double> values) {
    // numpy's reduction starts from addition's identity, 0.0.
    return 0.0 + pairwise_sum(values);
}

std::optional<double> aggregate_scores(std::span<const double> values, MetricAggregation aggregation) {
    const std::vector<double> data = without_nan(values);
    if (data.empty()) {
        return std::nullopt;
    }
    return aggregate(data, aggregation);
}

ScoreStats compute_stats(std::span<const double> values, std::span<const Threshold> thresholds,
                         MetricAggregation aggregation, MetricDirection direction) {
    ScoreStats stats;
    stats.worst_is_high = direction == MetricDirection::lower_is_better;
    const std::vector<double> data = without_nan(values);
    const std::size_t n = data.size();
    if (n == 0) {
        return stats;
    }
    stats.count = n;

    // A frame identical to the source scores +inf, which XPSNR reports as it
    // is. Such frames are counted, the mean handles them by the metric's way
    // of averaging, and the bands below include them: "better than 38 dB" is
    // exactly what a perfect frame is.
    stats.identical = static_cast<std::size_t>(std::count(data.begin(), data.end(), infinity));
    std::vector<double> finite;
    std::copy_if(data.begin(), data.end(), std::back_inserter(finite), [](double v) { return std::isfinite(v); });

    // One sort, then every percentile is a lookup into it. Order statistics
    // are well defined with infinities present.
    std::vector<double> ordered = data;
    std::sort(ordered.begin(), ordered.end());
    // When every frame is +inf (or every one -inf), so is every percentile.
    const bool all_one_infinity = std::isinf(ordered.front()) && ordered.front() == ordered.back();
    const auto worst_tail = [&](double percent_of_frames) {
        if (all_one_infinity) {
            return ordered.front();
        }
        return percentile(ordered, stats.worst_is_high ? 100.0 - percent_of_frames : percent_of_frames);
    };
    stats.worst_10_percent = worst_tail(10.0);
    stats.worst_5_percent = worst_tail(5.0);
    stats.worst_1_percent = worst_tail(1.0);
    stats.worst_0_1_percent = worst_tail(0.1);

    for (const Threshold& threshold : thresholds) {
        const auto count = static_cast<std::size_t>(std::count_if(data.begin(), data.end(), [&threshold](double v) {
            return threshold.comparison == '>' ? v > threshold.value : v < threshold.value;
        }));
        stats.thresholds.push_back({threshold, count, 100.0 * static_cast<double>(count) / static_cast<double>(n)});
    }

    stats.mean = aggregate(data, aggregation);
    // The standard deviation is over the finite frames only: a spread of NaN
    // whenever one frame happens to be identical says less than the spread
    // of the frames that do differ.
    stats.stdev = finite.empty() ? not_a_number : numpy_std(finite);
    stats.median = numpy_median(ordered);
    stats.minimum = ordered.front();
    stats.maximum = ordered.back();
    return stats;
}

}  // namespace vml
