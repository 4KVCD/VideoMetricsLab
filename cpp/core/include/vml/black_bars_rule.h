// HandBrake's rule for black bars (libhb/scan.c), reimplemented from its
// description (HandBrake is GPL-2; no code of it is here): a line is a bar
// where it is dark and even, so a bar that holds a faint picture (a packed
// alpha matte, subtitles) is kept and a bar lifted above black (a poor
// transfer's) is still found. Each sampled picture gives its edges; the
// median edge over the pictures is the answer. Pure: crop_detect.cpp
// decodes the pictures.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vml/video.h"

namespace vml {

// A picture's luma, 8 bits a sample, row after row.
struct LumaPicture {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> luma;
};

// How many rows (top, bottom) and columns (left, right) of bars a picture has.
struct BarEdges {
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
    bool operator==(const BarEdges&) const = default;
};

// The bars of one picture: from each edge, the lines that are dark (luma
// below 16 counted as 16, the line's mean below 32 and every sample within
// 16 of it), up to a quarter of the picture; the top and bottom from a
// hundredth of the height in, the rows before deciding only where no bar
// begins past them. Nothing for a picture with a quarter or more of bars on
// an edge (a fade, credits): it says nothing of the bars.
std::optional<BarEdges> picture_edges(const LumaPicture& picture);

// How many pictures that said something HandBrake judges a video on, at
// the fewest.
inline constexpr std::size_t fewest_judged = 3;

// The box that cuts the bars `counted` found (picture_edges of the pictures
// that said something), of `decoded` pictures sampled from a video
// `width` x `height`: each edge's median, or its smallest where four or more
// pictures show ten or more lines more on an edge than its median (the
// video changes shape: no picture is cut); each edge rounded down to even.
// The whole frame where fewer than fewest_judged pictures (or than
// `decoded`, for a video too short to give them) said something.
CropBox handbrake_box(const std::vector<BarEdges>& counted, std::size_t decoded, int width, int height);

// Pictures read from FFmpeg's yuv4mpegpipe of 16-bit luma (Cmono16, as
// extractplanes=y,format=gray16le gives it), each as 8 bits (the high byte:
// exactly the luma at its own depth shifted down to 8 bits). `read` gives
// up to the asked bytes of the stream, 0 at its end.
class Y4mLumaReader {
public:
    explicit Y4mLumaReader(std::function<std::size_t(void*, std::size_t)> read);
    // The next picture, or nothing at the stream's end. Throws
    // std::runtime_error for a stream that is not 16-bit mono Y4M, or that
    // ends inside a picture.
    std::optional<LumaPicture> next();

private:
    std::string line(bool at_start);
    std::function<std::size_t(void*, std::size_t)> read_;
    int width_ = 0;
    int height_ = 0;
    bool header_read_ = false;
};

}  // namespace vml
