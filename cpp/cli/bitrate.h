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

// The videos read, as the command prints them: a row each, rates in Mb/s.
std::string bitrate_table(const std::vector<BitrateVideo>& videos);

// The videos as the command writes them in JSON: each read one's figures (its
// length, frames and bytes from its packets), each other's error.
nlohmann::ordered_json bitrate_json(const std::vector<BitrateVideo>& videos);

// A video's three CSV files in `folder` (vml/bitrate.h's frames_csv,
// seconds_csv and gops_csv), named after its file with one name free for all
// three (unique_output_paths); the paths written. Throws std::runtime_error
// when one cannot be written.
std::vector<std::filesystem::path> write_bitrate_csvs(const BitrateVideo& video, const std::filesystem::path& folder,
                                                      bool adjust_start, std::set<std::filesystem::path>& reserved);

}  // namespace vml::cli
