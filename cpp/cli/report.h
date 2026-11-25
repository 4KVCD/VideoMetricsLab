// What a command-line run prints and writes: each video's table and the
// summary, the --json document, and the --csv files. A port of the results
// part of vmaf_app/cli.py, written the same way character for character.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "test_video.h"

namespace vml::cli {

struct CompareArguments;

// A video's metrics, each as the reports give it: its score, and the
// statistics the Metric Graphs tab shows (results_summary.h's
// metric_summary).
using Summaries = std::map<std::string, nlohmann::ordered_json>;

// Each video's summaries of the metrics it asked for that were scored:
// worked out once for the tables and the JSON (the statistics sort every
// frame's score).
std::vector<Summaries> summarize(const std::vector<TestVideo>& videos);

// ok: every metric asked for; partial: some; failed: none; cancelled: the
// run was stopped before it had them all.
std::string video_status(const TestVideo& video, bool cancelled);

// --difference: the first video minus the second, for each metric both were
// asked for and scored, paired as vml/difference.h pairs them: "score" (the
// two scores subtracted), and "mean", "minimum", "maximum" and "paired" of
// the frames' differences (CVVDP's seconds'; null with none paired), by key
// in the first video's order. Nothing unless there are two videos with a
// result.
std::optional<nlohmann::ordered_json> difference_metrics(const std::vector<TestVideo>& videos,
                                                         const std::vector<Summaries>& summaries);

// The run as plain-text tables (printed unless the command says --json -),
// with the difference's (`difference`: difference_metrics, when asked for).
std::string results_table(const VideoInfo& source, const std::vector<TestVideo>& videos,
                          const std::vector<Summaries>& summaries, bool cancelled,
                          const std::optional<nlohmann::ordered_json>& difference = std::nullopt);

// The run as --json writes it, in the shape of results_summary.h, which the
// window's Export results saves too, with the run's record (`record`,
// run_record.h) and each video's; with --difference, a "difference"
// section: the two videos' paths and difference_metrics.
nlohmann::ordered_json results_document(const VideoInfo& source, const std::vector<TestVideo>& videos,
                                        const std::vector<Summaries>& summaries, bool cancelled,
                                        const std::optional<nlohmann::ordered_json>& difference = std::nullopt,
                                        const RunRecord* record = nullptr);

// The results of a run that compared nothing, as --json writes them: why
// (`error`, and its code, codes.h), with `reference` (its path), no videos
// and the run's record where there is one; with no reference (the --batch
// list could not be read), an empty "batch" in place of the videos.
nlohmann::ordered_json error_document(const std::optional<std::string>& reference, const std::string& error,
                                      const std::string& code, const RunRecord* record = nullptr);

// Each result's per-frame scores as the window's per-frame CSV export writes them, in
// `folder`; `message` says what was written and `failed` what could not be
// or was written under another name (shown with -q too). False if one could
// not be. Each video's csv_file says where it went. With `json`
// (--frames-json), the same as the window's per-frame JSON export writes it
// (per_frame_json), as <video>.frames.json, each in its frames_json_file.
bool write_csvs(std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                const std::function<void(const std::string&)>& message,
                const std::function<void(const std::string&)>& failed, bool json = false);

// --score-summary: the videos' scores as score summary (score_summary.h),
// all of them in one JSON and one CSV file in `folder`, named after the
// reference (<reference>.score-summary.json and .csv), with the columns and
// CVVDP's displays `arguments` carries; reported as write_csvs reports.
bool write_score_summary(const std::vector<TestVideo>& videos, const std::filesystem::path& reference,
                         const std::filesystem::path& folder, const CompareArguments& arguments,
                         const std::function<void(const std::string&)>& message,
                         const std::function<void(const std::string&)>& failed);

// --frames-json's files' ending.
inline constexpr const char* frames_json_suffix = ".frames.json";
// --run-dir: each result as a saved run (run_file.h), the file Save analysis
// results writes, in `folder` as <video>.metrics.json (a further display's
// as <video>_<display>.metrics.json), labelled with that name; reported as
// write_csvs reports. Each video's run_file says where it went.
bool write_run_files(std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                     const std::function<void(const std::string&)>& message,
                     const std::function<void(const std::string&)>& failed);
// --difference with --csv: the first video's per-frame scores minus the
// second's (export_difference_csv) in `folder` as <first>_minus_<second>.csv,
// when both have a result; reported as write_csvs reports. False if it could
// not be written.
bool write_difference_csv(const std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                          const std::function<void(const std::string&)>& message,
                          const std::function<void(const std::string&)>& failed);

}  // namespace vml::cli
