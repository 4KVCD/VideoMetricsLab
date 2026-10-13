// How a video's colours are read for the perceptual metrics: its matrix,
// transfer, primaries and range, as H.273 numbers taken from its tags.
// Values the video does not tag are guessed as FFVship (Vship's own tool)
// guesses them. A port of vmaf_app/core/colour.py.
//
// Both implementations of the perceptual metrics read colours this way:
// Vship is given these numbers on the GPU (vship.h), and the CPU tools are
// given pictures converted and described the same way. When the two read
// colours differently, they scored the same frames far apart.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include "vml/video.h"

namespace vml {

// A colour tag that has no value here.
class UnsupportedColourError : public std::invalid_argument {
public:
    // `kind` is "matrix", "transfer", "primaries" or "range"; `value` is the
    // tag.
    UnsupportedColourError(std::string kind, std::string value);

    const std::string& kind() const { return kind_; }
    const std::string& value() const { return value_; }

private:
    std::string kind_;
    std::string value_;
};

// A video's colours as H.273 numbers, and whether it is full range.
struct VideoColour {
    int matrix = 1;  // 0 for RGB
    int transfer = 1;
    int primaries = 1;
    bool full_range = false;

    // Whether the transfer is an HDR one (PQ, ST 428 or HLG).
    bool hdr() const;

    bool operator==(const VideoColour&) const = default;
};

// The video's colours. Untagged values are guessed as FFVship guesses them:
// the matrix from the video's own height (BT.709 above 650 lines, otherwise
// BT.470BG), and the transfer and primaries from the matrix (BT.470BG's for
// BT.470BG; PQ and BT.2020 for BT.2020 and ICtCp; BT.709's otherwise).
// Untagged RGB (`rgb`) is sRGB, full range. `full_range_untagged` says the
// pixel format is full range when untagged (yuvj formats). Throws
// UnsupportedColourError for a tag with no value here.
VideoColour video_colour(const VideoInfo& info, bool rgb, bool full_range_untagged = false);

// Whether FFmpeg's pixel format holds RGB rather than YUV.
bool is_rgb_format(const std::string& pix_fmt);

// The video's colours as both implementations read them, or nothing for a
// tag neither can read.
std::optional<VideoColour> colour_of(const VideoInfo& info);

}  // namespace vml
