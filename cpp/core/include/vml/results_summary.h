// Comparisons' results summed up for a reader or a script, in one shape: what
// the command line's --json writes and the window's Export results saves
// (issue #8). Each test video is an entry of "videos", with each metric's
// score and the statistics Metric Graphs shows; as JSON, or as CSV with a
// row for each video.
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>

#include "vml/results.h"
#include "vml/video.h"

namespace vml {

// The JSON's version of the shape; a later one only adds to it.
inline constexpr int results_format = 1;

// A number as the JSON has it: "inf" or "-inf" (JSON has no infinity), and
// null for none or NaN.
nlohmann::ordered_json json_number(const std::optional<double>& value);
// A summary's number back as a number.
std::optional<double> summary_number(const nlohmann::ordered_json& number);

// One metric of `result`: its score, how many frames it covers and where it
// was calculated, with the statistics Metric Graphs shows. For a per-frame
// metric those are of its frames: median, standard deviation, minimum,
// maximum, the mean of the worst 10, 5, 1 and 0.1 %, the frames identical to
// the reference (XPSNR's, where any), and the share of frames in each of the
// metric's bands ("percent_of_frames", "> 95": 1.3). For a whole-video score
// with a timeline (CVVDP's seconds), the same of its seconds, under
// "per_second". Statistics of no frames (every score NaN) are null. Nothing
// if the result has none of the metric.
std::optional<nlohmann::ordered_json> metric_summary(const ComparisonResult& result, const std::string& key);

// A crop as the summary's JSON has it (x, y, width, height), or null for none.
nlohmann::ordered_json summary_crop_json(const std::optional<CropBox>& box);

// A video's media information, added to its entry.
void add_media(nlohmann::ordered_json& entry, const VideoInfo& info);

// The document's videos as CSV: a row for each, and a column for each value
// any of them has, named by its place in the JSON ("metrics.vmaf.median"),
// in the order they first come. Numbers as the JSON has them; UTF-8 with a
// byte order mark (for Excel), lines ended with CR LF, a cell quoted where it
// holds a comma, a quote or a line break.
std::string results_csv(const nlohmann::ordered_json& document);

}  // namespace vml
