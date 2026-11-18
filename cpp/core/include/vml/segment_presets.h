// Weak segment presets: starting points for a weak segment profile
// (segment_profiles.h) for a kind of video, with the metrics and thresholds
// that suit it. They are made to be adjusted: a profile made from one is
// the user's own. The command line takes them as --segments preset:KEY.
#pragma once

#include <string_view>
#include <vector>

#include "vml/weak_frames.h"

namespace vml {

struct SegmentPreset {
    std::string_view key;    // the command line's name: "animation"
    const char* name;        // shown, translated by the window: "Animation"
    const char* about;       // what it is for and why, translated by the window
    SegmentRules rules;
};

// Every preset, in the order shown: animation, CGI, grain-free digital,
// grainy film, HDR, low-bitrate preview.
const std::vector<SegmentPreset>& segment_presets();

// The preset `name` names: its key or its English name, in any case; or none.
const SegmentPreset* segment_preset(std::string_view name);

}  // namespace vml
