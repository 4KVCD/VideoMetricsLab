#include "vml/video.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

namespace vml {

std::string CropBox::as_filter() const {
    return "crop=" + std::to_string(w) + ":" + std::to_string(h) + ":" + std::to_string(x) + ":" + std::to_string(y);
}

bool CropBox::is_noop(int frame_width, int frame_height) const {
    return w == frame_width && h == frame_height && x == 0 && y == 0;
}

long long VideoInfo::estimated_frame_count() const {
    if (nb_frames > 0) {
        return nb_frames;
    }
    // Python's round(): halves go to the even neighbour.
    return std::max(1LL, static_cast<long long>(std::nearbyint(duration * fps)));
}

bool VideoInfo::is_raw_stream() const {
    // FFmpeg's demuxers of bare elementary streams, which carry no timestamps.
    constexpr std::array<std::string_view, 15> raw_stream_formats{
        "h264", "hevc", "vvc",       "evc",  "av1",       "obu",  "mpegvideo", "m4v",
        "h263", "vc1",  "dirac",     "cavsvideo", "avs2", "avs3", "mjpeg"};
    return std::find(raw_stream_formats.begin(), raw_stream_formats.end(), format_name) != raw_stream_formats.end();
}

bool VideoInfo::is_variable_frame_rate() const {
    if (nominal_fps <= 0 || fps <= 0) {
        return false;
    }
    return std::abs(nominal_fps - fps) > std::max(0.01, fps * 0.001);
}

}  // namespace vml
