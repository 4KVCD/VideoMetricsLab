// A result's per-frame scores as CSV, as the window's Export CSV writes
// them (a port of export_csv and its file-name helpers in
// vmaf_app/core/run_io.py), or as JSON.
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
// export_csv's file as text.
std::string per_frame_csv(const ComparisonResult& result);

// The same table as JSON, for scripts that read JSON: "reference" and
// "test" (the videos' paths), "fps", "columns" (the CSV's header) and
// "rows", each one of the CSV's rows as a list, a blank cell null. Numbers
// as Python's json writes them (an endless PSNR Infinity), each row on a
// line of its own. export_per_frame_json writes it to `path`, throwing as
// export_csv does.
std::string per_frame_json(const ComparisonResult& result);
void export_per_frame_json(const ComparisonResult& result, const std::filesystem::path& path);

// Writes `first` minus `second` as CSV, paired as vml/difference.h pairs
// them: export_csv's rows, times and header for `first`, each cell the
// difference at that frame (CVVDP's: at that second; its video JOD, the two
// videos' JODs subtracted). A metric only one result has, or a frame without
// a pair, leaves its cells blank. Throws as export_csv does.
void export_difference_csv(const ComparisonResult& first, const ComparisonResult& second,
                           const std::filesystem::path& path);

// A label reduced to something usable as a file name: letters, digits and
// "-_." kept, everything else "_".
std::string safe_filename_stem(const std::string& label);
// The name a label's output file starts with: safe_filename_stem, or "run" for
// a label with nothing left.
std::string output_stem(const std::string& label);

// The label of export_difference_csv's file, "<first>_minus_<second>", each
// name cut to 60 characters so that two long names still make a usable path.
std::string difference_label(const std::string& first, const std::string& second);

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
