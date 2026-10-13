// What a command-line run prints and writes: each video's table and the
// summary, the --json document, and the --csv files. A port of the results
// part of vmaf_app/cli.py, written the same way character for character.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "test_video.h"

namespace vml::cli {

// A video's metrics, each as the reports give it: its score, and for a
// per-frame metric the statistics the Metric Graphs tab shows.
using Summaries = std::map<std::string, nlohmann::ordered_json>;

// Each video's summaries of the metrics it asked for that were scored:
// worked out once for the tables and the JSON (the statistics sort every
// frame's score).
std::vector<Summaries> summarize(const std::vector<TestVideo>& videos);

// ok: every metric asked for; partial: some; failed: none; cancelled: the
// run was stopped before it had them all.
std::string video_status(const TestVideo& video, bool cancelled);

// The run as plain-text tables (printed unless the command says --json -).
std::string results_table(const VideoInfo& source, const std::vector<TestVideo>& videos,
                          const std::vector<Summaries>& summaries, bool cancelled);

// The run as --json writes it.
nlohmann::ordered_json results_document(const VideoInfo& source, const std::vector<TestVideo>& videos,
                                        const std::vector<Summaries>& summaries, bool cancelled);

// Each result's per-frame scores as the window's Export CSV writes them, in
// `folder`; `message` says what was written. False if one could not be.
bool write_csvs(const std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                const std::function<void(const std::string&)>& message);

}  // namespace vml::cli
