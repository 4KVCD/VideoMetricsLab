// Statistics of a metric's per-frame scores (compute_stats), and the one
// number for a whole run by the metric's own way of averaging
// (aggregate_scores). A port of vmaf_app/core/stats.py.
//
// Sums are made as numpy makes them (numpy_sum), so on Windows the results
// are the Python app's to the last digit; elsewhere the tests ask for 12
// significant digits.
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "vml/metrics.h"

namespace vml {

// How many frames score in a band, such as "> 95".
struct ThresholdCount {
    Threshold threshold;
    std::size_t count;
    double percentage;  // of the frames that have a score
};

struct ScoreStats {
    std::size_t count = 0;      // frames with a score (NaN frames are left out)
    std::size_t identical = 0;  // frames scoring +inf: identical to the source
    double mean = 0.0;          // by the metric's own way of averaging
    double median = 0.0;
    double stdev = 0.0;         // population; over the finite frames only
    double minimum = 0.0;
    double maximum = 0.0;
    // The worst frames' tail: the score below which the worst 10%, 5%, 1%
    // and 0.1% of frames fall. For a lower-is-better metric (Butteraugli)
    // the worst frames are the high ones, so these are the 90th, 95th, 99th
    // and 99.9th percentiles instead.
    double worst_10_percent = 0.0;
    double worst_5_percent = 0.0;
    double worst_1_percent = 0.0;
    double worst_0_1_percent = 0.0;  // needs a lot of frames to mean anything
    std::vector<ThresholdCount> thresholds;
    bool worst_is_high = false;
};

// The sum of `values` as numpy's np.sum gives it: pairwise, so the rounding
// error grows with log2(n) rather than n, in numpy's exact order.
double numpy_sum(std::span<const double> values);

namespace detail {

// numpy's pairwise_sum for float64 (numpy/_core/src/umath/loops_utils.h.src),
// step for step, over the values value(begin) to value(end - 1): fewer than
// 8 values one by one; up to 128 in eight running sums added in pairs; more
// split in two at a multiple of 8. Each value is asked for once, in order.
template <typename Value>
double pairwise_sum(std::size_t begin, std::size_t end, const Value& value) {
    const std::size_t n = end - begin;
    if (n < 8) {
        double sum = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            sum += value(i);
        }
        return sum;
    }
    if (n <= 128) {
        // Eight running sums, the j-th over the values at j, j + 8, j + 16...
        double sums[8];
        for (std::size_t j = 0; j < 8; ++j) {
            sums[j] = value(begin + j);
        }
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                sums[j] += value(begin + i + j);
            }
        }
        double sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for (; i < n; ++i) {  // the last n % 8 values, one by one
            sum += value(begin + i);
        }
        return sum;
    }
    std::size_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(begin, begin + half, value) + pairwise_sum(begin + half, end, value);
}

}  // namespace detail

// numpy_sum of the `count` values value(0) to value(count - 1), made as
// they are asked for (once each, in order), so that they need no array.
template <typename Value>
double numpy_sum_of(std::size_t count, const Value& value) {
    return 0.0 + detail::pairwise_sum(0, count, value);  // numpy starts from addition's identity, 0.0
}

// np.median of sorted values: the middle one, or the mean of the middle two.
double numpy_median(std::span<const double> ordered);

// One number for a whole run, by the metric's way of averaging, or nothing
// when no frame has a score. NaN frames (not calculated) are left out;
// infinities are not, because for XPSNR they mean an identical frame.
std::optional<double> aggregate_scores(std::span<const double> values, MetricAggregation aggregation);

// The statistics of a metric's per-frame scores, with its score bands
// (`thresholds`; empty leaves them out), its way of averaging and its
// direction, which decides which end is the worst frames' tail. NaN frames
// are left out. With no frames left, everything is 0.
ScoreStats compute_stats(std::span<const double> values, std::span<const Threshold> thresholds,
                         MetricAggregation aggregation, MetricDirection direction);

}  // namespace vml
