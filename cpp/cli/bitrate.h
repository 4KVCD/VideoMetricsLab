// The bitrate command: each video's video stream read with ffprobe as the
// Bitrate Viewer tab reads it (vml/bitrate.h): its codec, length, frames and
// bitrate (the average, and the lowest and highest one-second bitrate), as a
// table or JSON, and with --csv its graphs' figures as CSV files.
#pragma once

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "arguments.h"
#include "vml/bitrate.h"
#include "vml/difference.h"
#include "vml/video.h"

namespace vml::cli {

// Runs the bitrate command; returns the exit code: 0 when every video was
// read, 1 when one was not (or a file could not be written), 2 for a wrong
// command, 130 for Ctrl+C.
int bitrate(const BitrateArguments& arguments);

// A video of the command: what ffprobe says of it and its packets, or why it
// could not be read.
struct BitrateVideo {
    std::filesystem::path path;
    std::optional<VideoInfo> info;
    std::optional<BitrateData> data;
    std::string error;
};

// --difference: the first video's bitrate minus the second's, from each
// stream's start.
struct BitrateDifference {
    std::filesystem::path first;
    std::filesystem::path second;
    double average_kbps = 0;                   // the whole streams' averages subtracted
    std::optional<double> average_percent;     // that against the second's average; none when it is 0
    std::optional<DifferenceSummary> seconds;  // of the seconds both have, by number (seconds_difference)
};

// The difference of two videos both read; nothing otherwise.
std::optional<BitrateDifference> bitrate_difference(const std::vector<BitrateVideo>& videos);

// The videos read, as the command prints them: a row each, rates in Mb/s;
// then the difference's table, if any.
std::string bitrate_table(const std::vector<BitrateVideo>& videos,
                          const std::optional<BitrateDifference>& difference = std::nullopt);

// The videos as the command writes them in JSON: each read one's figures (its
// length, frames and bytes from its packets), each other's error; and the
// difference, if any.
nlohmann::ordered_json bitrate_json(const std::vector<BitrateVideo>& videos,
                                    const std::optional<BitrateDifference>& difference = std::nullopt);

// The first video's three CSV files of its difference from the second in
// `folder` (vml/bitrate.h's frame_difference_csv and step_difference_csv, of
// the seconds' and groups of pictures' graphs), named
// <first>_minus_<second>_bitrate_frames.csv and so on; the paths written.
// Throws std::runtime_error when one cannot be written.
std::vector<std::filesystem::path> write_difference_csvs(const BitrateVideo& first, const BitrateVideo& second,
                                                         const std::filesystem::path& folder, bool adjust_start,
                                                         std::set<std::filesystem::path>& reserved);

// A video's three CSV files in `folder` (vml/bitrate.h's frames_csv,
// seconds_csv and gops_csv), named after its file with one name free for all
// three (unique_output_paths); the paths written. Throws std::runtime_error
// when one cannot be written.
std::vector<std::filesystem::path> write_bitrate_csvs(const BitrateVideo& video, const std::filesystem::path& folder,
                                                      bool adjust_start, std::set<std::filesystem::path>& reserved);

}  // namespace vml::cli
