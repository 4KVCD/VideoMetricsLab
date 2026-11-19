// The bitrate saved with the saved scores (metric_cache.h): each video file's
// packets as the Bitrate tab reads them (bitrate.h), so that a file read
// once is not read again, and its bitrate graphs can be drawn after the file
// has gone. A file is known by its content (sampled_file_hash): renamed,
// moved or copied it keeps its saved bitrate; a remux, whose packets are
// its container's, is read again.
//
// <base>/bitrate/<sampled hash>.npz, beside the scores' v2 folder (the Python
// app does not read it): the packets' times, durations, sizes and keyframes,
// the frame types where they were read (the window reads them; the command
// line does not), and the video's details.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vml/bitrate.h"
#include "vml/video.h"

namespace vml {

struct SavedBitrate {
    VideoInfo info;
    BitrateData data;
    // A type per frame, or none where they cannot be known; not read at all
    // (the command line's reading): nothing.
    std::optional<std::vector<std::uint8_t>> frame_types;
};

// The folder the bitrate is saved in under `base`.
std::filesystem::path bitrate_store_folder(const std::filesystem::path& base);

// The bitrate saved for the file of `sampled_hash`, its data's path `path`
// (the file's as it is named now); nothing when none is saved or it cannot
// be read.
std::optional<SavedBitrate> load_saved_bitrate(const std::filesystem::path& base, const std::string& sampled_hash,
                                               const std::filesystem::path& path);

// Saves `bitrate` for the file of `sampled_hash`. Throws when the file
// cannot be written.
void save_bitrate(const std::filesystem::path& base, const std::string& sampled_hash, const SavedBitrate& bitrate);

}  // namespace vml
