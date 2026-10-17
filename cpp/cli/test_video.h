// One test video of a command-line run: what the window keeps in a row.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/analysis_request.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml::cli {

struct TestVideo {
    std::filesystem::path path;
    std::optional<VideoInfo> info;
    // The reference as this video is compared with: the run's, with this
    // video's colour tags where only this video has them (colours_taken).
    VideoInfo source;
    // Which of the two took the other's colour tags, having none of its own
    // ("the reference" or "the test video"), or nothing.
    std::string colours_taken;
    VmafOptions options;
    std::vector<std::string> metrics;  // the metric keys it is compared on
    // What it asks for, as its saved scores are looked up and saved under.
    AnalysisRequest request;
    // Whether its scores are read from and saved to the saved scores: not
    // those of a video read with --colour or with the other video's colour
    // tags, nor of a raw stream, as the saved scores know a video by its
    // file, not by the colours or frame rate it was read with.
    bool save_scores = true;
    // What the reader should know: metrics left out and why.
    std::vector<std::string> notes;
    std::string error;  // why it was not compared at all
    std::optional<ComparisonResult> result;
    // The metrics that failed and why, in the order they failed.
    std::vector<std::pair<std::string, std::string>> failures;
    bool from_cache = false;  // nothing had to be calculated
    std::string display;      // the preset CVVDP models, when CVVDP is asked for
    // CVVDP on each further --cvvdp-display: a TestVideo of its own for each,
    // with CVVDP alone, run as a comparison of its own.
    std::vector<TestVideo> other_displays;
    bool for_display_only = false;  // one of another TestVideo's other_displays

    // Its name as the reports give it.
    std::string name() const;
    // Its name as progress lines give it: with the display, for one of
    // another video's other displays.
    std::string label() const;
};

}  // namespace vml::cli
