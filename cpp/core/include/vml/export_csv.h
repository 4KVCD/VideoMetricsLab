// A result's per-frame scores as CSV, as the window's Export CSV writes
// them. A port of export_csv and its file-name helpers in
// vmaf_app/core/run_io.py.
#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "vml/results.h"

namespace vml {

// Writes a result as CSV: one row per frame any metric scored, a column per
// frame metric. The first seven columns are always frame, time_s, vmaf,
// vmaf_neg, psnr, ssim, xpsnr (scripts rely on them), then the other frame
// metrics in the registry's order; a metric with no score for a frame
// leaves its cell blank. Throws std::runtime_error when the file cannot be
// written.
void export_csv(const ComparisonResult& result, const std::filesystem::path& path);

// A label reduced to something usable as a file name: letters, digits and
// "-_." kept, everything else "_".
std::string safe_filename_stem(const std::string& label);

// A path in `folder` for `label` that clashes with nothing on disk or in
// `reserved` (which it is added to): label.csv, then label_2.csv...
std::filesystem::path unique_output_path(const std::filesystem::path& folder, const std::string& label,
                                         const std::string& suffix, std::set<std::filesystem::path>& reserved);
// unique_output_path for several files of one label: one name free for every
// suffix (label_bitrate_frames.csv and label_bitrate_seconds.csv both, or
// label_2_...), so that the files stay together. The paths in the suffixes'
// order.
std::vector<std::filesystem::path> unique_output_paths(const std::filesystem::path& folder, const std::string& label,
                                                       const std::vector<std::string>& suffixes,
                                                       std::set<std::filesystem::path>& reserved);

}  // namespace vml
