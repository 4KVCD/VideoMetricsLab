// How the window shows test videos' names (Settings > Window > Video names).
// Encodes of one source often share all but a few characters of a long
// name ("Show - S01E03 - Title - 1080p WEB-DL x265 10-bit CRF 20"), and a
// table column or a drop-down cut at its end shows only the part they share.
// The names shown are shortened together, as a set: "…CRF 20", "…CRF 22".
// Tooltips, exports and file names keep the whole name.
#pragma once

#include <QString>

#include <vector>

namespace vml {
struct Settings;
}

namespace vml::app {

struct VideoNameRule {
    enum class Kind {
        // What tells the names apart: the start and end they all share left
        // out, at a word's edge, with the word before the first difference
        // kept ("…CRF 20"). A single name, or names that differ in nothing
        // left after that, are shown whole.
        differs,
        // The first `start` and last `end` characters of each, with "…"
        // between (one of them may be 0); a name no longer than that whole.
        ends,
        full,
    };
    Kind kind = Kind::differs;
    int start = 10;
    int end = 16;
};

// The rule the settings choose (an unknown kind: differs; the counts kept
// within 0 to 200, and both 0: full).
VideoNameRule video_name_rule(const Settings& settings);

// `names` as shown, in the same order.
std::vector<QString> shown_video_names(const std::vector<QString>& names, const VideoNameRule& rule);

}  // namespace vml::app
