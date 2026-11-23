// BD-rate (Bjontegaard delta rate): how much more bitrate (positive) or less
// (negative) a test encoder needs than an anchor for the same quality, in
// percent, averaged over the quality range both rate curves cover. The
// command line's bdrate.
//
// The method is the VVenC test scripts' (fasteval.py's bd_rate), so that
// VideoMetricsLab gives their numbers: each curve is the
// natural log of the rate as a function of the score, interpolated with
// PCHIP (scipy's PchipInterpolator: a monotone piecewise cubic), integrated
// in closed form over the scores both curves have; BD-rate =
// (exp((test's integral - anchor's) / (the range's width)) - 1) * 100.
#pragma once

#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vml {

// One encode of a rate curve: its bitrate and its quality score.
struct RatePoint {
    double kbps = 0;
    double score = 0;
};

// Why a BD-rate could not be calculated.
enum class BdRateGap {
    none,
    too_few_points,  // a curve has fewer than 2 points
    tied_scores,     // a curve's scores are not all different (fasteval: not strictly increasing);
                     // with copies_once, a point exactly like another (rate and score) counts once
    no_overlap,      // the curves' score ranges do not overlap
    bad_number,      // a rate that is not above 0, or a number that is not finite
};

struct BdRateResult {
    std::optional<double> percent;  // the BD-rate, none when `gap` says why
    BdRateGap gap = BdRateGap::none;
};

// `test`'s BD-rate against `anchor`; with `lower_is_better` (Butteraugli) the
// scores are negated first, as fasteval does. With `copies_once`, a point
// exactly like another (rate and score) is the same encode and counts once
// (the BD-rate tab: one run file of an encode added twice); else it ties,
// as fasteval has it.
BdRateResult bd_rate(std::span<const RatePoint> anchor, std::span<const RatePoint> test, bool lower_is_better = false,
                     bool copies_once = false);

// The curve bd_rate integrates for `points`, for drawing: `samples` points
// evenly spaced in score across the points' scores, lowest score first (for
// lower_is_better, highest first), each with its interpolated rate. Empty
// where bd_rate would have no curve (fewer than 2 points, tied scores, a
// number that is not finite or a rate not above 0).
std::vector<RatePoint> pchip_curve(std::span<const RatePoint> points, bool lower_is_better, int samples,
                                   bool copies_once = false);

// An encode's score with what made it: its rate in kb/s, or a setting such
// as its CRF.
struct SettingPoint {
    double setting = 0;
    double score = 0;
};

// Where a curve of encodes reaches a score (setting_for_score).
struct TargetResult {
    std::optional<double> setting;  // none when `gap` says why
    // Beyond the encodes' scores: from the straight line through the two
    // nearest, rather than between two.
    bool extrapolated = false;
    // Indexes into the points: the two encodes either side of the score
    // (lower score first), or the two it was extrapolated from; for a score
    // an encode has, that encode twice.
    std::optional<std::pair<std::size_t, std::size_t>> bracket;
    BdRateGap gap = BdRateGap::none;
};

// The setting at which `points`' curve reaches `score`: the curve bd_rate
// integrates, the setting (its log with `log_setting`, for a rate) against
// the score, interpolated with PCHIP. Beyond the scores the points have, the
// straight line through the two nearest goes on to it. The direction of the
// metric does not matter: the curve is read at a score. Gaps as bd_rate's:
// fewer than 2 points, tied scores, a number that is not finite (`score`
// too), or with `log_setting` a setting not above 0.
TargetResult setting_for_score(std::span<const SettingPoint> points, double score, bool log_setting);

// The PCHIP slopes at each of the points (x ascending, all different), as
// scipy's PchipInterpolator works them out. Exposed for its test.
std::vector<double> pchip_slopes(std::span<const double> x, std::span<const double> y);

// A gap as the warnings say it: "scores not strictly increasing" and so on,
// in English (the window translates it).
std::string bd_rate_gap_text(BdRateGap gap);

// The mean and median of the values there are (numpy's nanmean and
// nanmedian: an even count's median is the mean of the middle two); none
// for none.
std::optional<double> mean_of(const std::vector<std::optional<double>>& values);
std::optional<double> median_of(const std::vector<std::optional<double>>& values);

}  // namespace vml
