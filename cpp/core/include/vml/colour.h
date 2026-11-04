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

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "vml/video.h"

namespace vml {

// Whether a colour range's name, in lower case, says full range. A video with
// no tag has its own rule where it is read (VideoInfo::full_range, video_colour).
inline bool is_full_range_name(std::string_view lowered) {
    return lowered == "pc" || lowered == "jpeg" || lowered == "full";
}

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

// Whether a colour tag's value says nothing about the colours: empty,
// "unknown", "unspecified" or "reserved", in any case (Python's
// colour.UNTAGGED).
bool is_untagged_tag(std::string_view value);

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

// A pair's videos with the colour matrix they are read with: a video without
// a matrix tag is read with the other's, or, neither having one, both with
// the reference's guess (its height, as video_colour guesses). An encode keeps
// its source's YUV coding whatever its size, so a 540p encode of an untagged
// 720p source, each guessed from its own height (BT.470BG and BT.709), was
// read with another matrix than its source and scored as badly broken. A
// pair with an RGB video is left as it is.
std::pair<VideoInfo, VideoInfo> with_shared_matrix(VideoInfo source, VideoInfo distorted);

// Whether with_shared_matrix reads either video of the pair with another
// matrix than its own tags or height give.
bool matrix_shared(const VideoInfo& source, const VideoInfo& distorted);

// Whether the video says its transfer or its primaries.
bool has_colour_tags(const VideoInfo& info);

// Gives `to` the colour tags of `from` when only `from` has them: its transfer
// and primaries, and its matrix unless `to` says its own. An encode keeps its
// source's colours, so an untagged encode of a tagged source is read as that
// source: left to video_colour's guess from the matrix, an HLG source's encode
// was read as PQ and scored as badly broken (SSIMULACRA2 46.68 became -64497).
// Returns whether it did.
bool take_colour_tags(VideoInfo& to, const VideoInfo& from);

// A pair's videos as they are read: one without colour tags with the other's
// (take_colour_tags), then with one colour matrix (with_shared_matrix).
std::pair<VideoInfo, VideoInfo> with_shared_colours(VideoInfo source, VideoInfo distorted);

// Whether with_shared_colours reads either video otherwise than
// with_shared_matrix alone did, as saved scores were made before it.
bool colours_lent(const VideoInfo& source, const VideoInfo& distorted);

// FFmpeg's name for an H.273 matrix (setparams' colorspace=), for the
// matrices FFmpeg can convert with; nothing for the others.
std::optional<std::string_view> ffmpeg_matrix_name(int matrix);

// The PNG chunks that make libjxl's tools read an RGB picture's values as
// Vship reads the video's. Vship takes BT.709 (and BT.2020's SDR curves) as
// a display's 2.4 gamma (BT.1886), not as the camera curve H.273 defines,
// and no H.273 code means a plain power curve, so such a curve is described
// with gAMA and cHRM; every other transfer (sRGB, linear, PQ, HLG) with
// cICP.
std::string png_colour_chunks(const VideoColour& colour);

// Replaces the colour chunks of the PNG at `path` (one FFmpeg wrote) with
// png_colour_chunks' for `colour`. Throws std::runtime_error when the file
// cannot be read or written.
void describe_png(const std::filesystem::path& path, const VideoColour& colour);

}  // namespace vml
