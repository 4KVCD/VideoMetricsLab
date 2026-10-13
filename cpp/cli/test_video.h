// One test video of a command-line run: what the window keeps in a row.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vml/options.h"
#include "vml/results.h"
#include "vml/video.h"

namespace vml::cli {

struct TestVideo {
    std::filesystem::path path;
    std::optional<VideoInfo> info;
    VmafOptions options;
    std::vector<std::string> metrics;  // the metric keys it is compared on
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
