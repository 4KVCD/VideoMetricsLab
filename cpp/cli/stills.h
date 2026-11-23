// --stills: PNGs of the frames a run found worst, so that a person can look
// at an automated run's failures without opening the window. For each
// frame, the reference's and the test video's pictures as the run compared
// them (cropped and scaled, HDR mapped to SDR as Video Compare's Export frame
// as PNG makes them: frame_extract.h) and the two side by side.
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "test_video.h"
#include "vml/results.h"
#include "vml/weak_frames.h"

namespace vml::cli {


// The frames of `result` to make stills of, in frame order:
// - "worst": each metric's `count` worst dips (worst_frames, as --worst lists
//   them), a frame two metrics chose once;
// - "segment": the worst frame of each of the `count` weakest of the weak
//   segments `rules` find (by the first condition's metric that has scores:
//   its lowest score in the segment, or highest for a '>' condition).
std::vector<StillFrame> still_frames(const ComparisonResult& result, const std::string& from, std::size_t count,
                                     const SegmentRules& rules);

// The stills of each video with a result, written to `folder` as
// <video>.frame<NNNNNN>.reference.png, .test.png and .side-by-side.png; the
// frames and files kept in each video's `stills` for the JSON. `message` and
// `failed` as write_run_files'. False when a video's stills could not all be
// made or written (its video gone, FFmpeg's error); the others are made.
bool write_stills(std::vector<TestVideo>& videos, const std::string& from, std::size_t count,
                  const SegmentRules& rules, const std::filesystem::path& folder,
                  const std::function<void(const std::string&)>& message,
                  const std::function<void(const std::string&)>& failed);

// A video's stills as the results' JSON has them ("stills").
nlohmann::ordered_json stills_json(const std::vector<VideoStill>& stills);

}  // namespace vml::cli
