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
