#include "vml/colour.h"

#include <map>
#include <set>
#include <string_view>
#include <utility>

#include "vml/python_text.h"

namespace vml {

namespace {

// FFmpeg's tag names (as ffprobe prints them, with their aliases), mapped to
// the values of Vship's enums in VshipColor.h, which are the H.273 numbers.
// The mapping is FFVship's (src/ffvship_utility/ffmpegToVshipColorFormat.hpp
// in Vship 5.1.1). A tag missing here has no Vship value.
const std::map<std::string, int, std::less<>> matrices{
    {"gbr", 0},      {"rgb", 0},       {"bt709", 1},      {"bt470bg", 5},   {"smpte170m", 6},
    {"ycgco", 8},    {"ycocg", 8},     {"bt2020nc", 9},   {"bt2020ncl", 9}, {"bt2020c", 10},
    {"bt2020cl", 10}, {"ictcp", 14},   {"ycgco-re", 16},  {"ycgco-ro", 17},
};
// BT.2020's 10- and 12-bit curves are BT.709's curve (H.273), and FFVship
// maps them that way.
const std::map<std::string, int, std::less<>> transfers{
    {"bt709", 1},         {"bt470m", 4},         {"gamma22", 4},      {"bt470bg", 5},
    {"gamma28", 5},       {"smpte170m", 6},      {"smpte240m", 7},    {"linear", 8},
    {"iec61966-2-1", 13}, {"srgb", 13},          {"iec61966_2_1", 13}, {"bt2020-10", 1},
    {"bt2020_10bit", 1},  {"bt2020-12", 1},      {"bt2020_12bit", 1}, {"smpte2084", 16},
    {"smpte428", 17},     {"smpte428_1", 17},    {"arib-std-b67", 18},
};
const std::map<std::string, int, std::less<>> primaries_codes{
    {"bt709", 1}, {"bt470m", 4}, {"bt470bg", 5}, {"smpte170m", 6}, {"smpte240m", 7}, {"bt2020", 9}, {"smpte432", 12},
};

bool untagged(std::string_view name) {
    return name.empty() || name == "unknown" || name == "unspecified" || name == "reserved";
}

// The transfers whose light is relative to the display's white (SDR). The
// others (PQ, ST 428 and HLG) are HDR.
const std::set<int> sdr_transfers{1, 4, 5, 6, 7, 8, 13};

}  // namespace

UnsupportedColourError::UnsupportedColourError(std::string kind, std::string value)
    : std::invalid_argument("unsupported " + kind + ": " + value), kind_(std::move(kind)), value_(std::move(value)) {}

bool VideoColour::hdr() const { return !sdr_transfers.contains(transfer); }

VideoColour video_colour(const VideoInfo& info, bool rgb, bool full_range_untagged) {
    VideoColour colour;
    const std::string matrix_name = ascii_lower(info.color_space);
    if (rgb) {
        colour.matrix = 0;
    } else if (untagged(matrix_name)) {
        // The video's own height, not the size it is compared at: otherwise
        // an untagged 1080p source scaled to a 640x360 encode would be read
        // as BT.470BG.
        colour.matrix = info.height > 650 ? 1 : 5;
    } else if (const auto found = matrices.find(matrix_name); found != matrices.end()) {
        colour.matrix = found->second;
    } else {
        throw UnsupportedColourError("matrix", info.color_space);
    }
    const bool bt2020_family = colour.matrix == 9 || colour.matrix == 10 || colour.matrix == 14;

    const std::string transfer_name = ascii_lower(info.color_transfer);
    if (untagged(transfer_name)) {
        colour.transfer = colour.matrix == 0 ? 13 : colour.matrix == 5 ? 5 : bt2020_family ? 16 : 1;
    } else if (const auto found = transfers.find(transfer_name); found != transfers.end()) {
        colour.transfer = found->second;
    } else {
        throw UnsupportedColourError("transfer", info.color_transfer);
    }

    const std::string primaries_name = ascii_lower(info.color_primaries);
    if (untagged(primaries_name)) {
        colour.primaries = colour.matrix == 5 ? 5 : bt2020_family ? 9 : 1;
    } else if (const auto found = primaries_codes.find(primaries_name); found != primaries_codes.end()) {
        colour.primaries = found->second;
    } else {
        throw UnsupportedColourError("primaries", info.color_primaries);
    }

    const std::string range_name = ascii_lower(info.color_range);
    if (range_name == "pc" || range_name == "jpeg" || range_name == "full" ||
        (range_name.empty() && full_range_untagged)) {
        colour.full_range = true;
    } else if (range_name.empty() || range_name == "unknown" || range_name == "unspecified" || range_name == "tv" ||
               range_name == "mpeg" || range_name == "limited") {
        colour.full_range = rgb;
    } else {
        throw UnsupportedColourError("range", info.color_range);
    }
    return colour;
}

bool is_rgb_format(const std::string& pix_fmt) {
    const std::string name = ascii_lower(pix_fmt);
    for (const std::string_view prefix : {"gbr", "rgb", "bgr", "argb", "abgr", "0rgb", "0bgr"}) {
        if (name.starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

std::optional<VideoColour> colour_of(const VideoInfo& info) {
    const std::string name = ascii_lower(info.pix_fmt);
    try {
        return video_colour(info, is_rgb_format(name), name.starts_with("yuvj"));
    } catch (const UnsupportedColourError&) {
        return std::nullopt;
    }
}

}  // namespace vml
