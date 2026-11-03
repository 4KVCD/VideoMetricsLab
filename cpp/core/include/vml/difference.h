// Two runs' difference, the first's scores minus the second's: Metric Graphs'
// Difference band and readout, its CSV export, and the command line's
// --difference all pair the runs here, so that they agree.
//
// Points are paired by frame number, as the runs are scored (frame N with N)
// and Go to frame shows them; CVVDP's per-second points by second number, as
// its seconds follow the video's whole seconds. A point the second run lacks
// has no difference (NaN), nor one whose score is missing in either; XPSNR's
// infinity (a frame identical to its reference) in both is no difference
// (0), in one alone none to give.
#pragma once

#include <optional>
#include <span>
#include <vector>

namespace vml {

// One pair of scores' difference, a - b, or NaN (see above): of two points,
// or of two videos' whole scores.
double point_difference(double a, double b);

// The number a per-second point is paired by: its second's, from its start.
long long second_number(double start);

// `first` minus `second` at each of the first's points, NaN where there is
// none. Both runs' points are in order of their frames (or starts).
std::vector<double> frame_difference(std::span<const int> first_frames, std::span<const float> first_values,
                                     std::span<const int> second_frames, std::span<const float> second_values);
std::vector<double> second_difference(std::span<const double> first_starts, std::span<const float> first_values,
                                      std::span<const double> second_starts, std::span<const float> second_values);

// The differences there are (NaN left out): their mean, lowest, highest and
// how many; nothing when there is none.
struct DifferenceSummary {
    double mean = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    long long paired = 0;
};
std::optional<DifferenceSummary> summarize_difference(std::span<const double> differences);

}  // namespace vml
