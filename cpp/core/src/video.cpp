#include "vml/video.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

#include "vml/number_format.h"
#include "vml/platform/system.h"

namespace vml {

GpuVendor vendor_of_pci_id(unsigned id) {
    if (id == platform::nvidia_pci_vendor) {
        return GpuVendor::nvidia;
    }
    if (id == platform::intel_pci_vendor) {
        return GpuVendor::intel;
    }
    if (std::find(std::begin(platform::amd_pci_vendors), std::end(platform::amd_pci_vendors), id) !=
        std::end(platform::amd_pci_vendors)) {
        return GpuVendor::amd;
    }
    return GpuVendor::none;
}

std::string_view vendor_label(GpuVendor vendor) {
    switch (vendor) {
    case GpuVendor::nvidia:
        return "NVIDIA";
    case GpuVendor::intel:
        return "INTEL";
    case GpuVendor::amd:
        return "AMD";
    default:
        return "";
    }
}

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
    // FFmpeg's demuxers of bare elementary streams, which carry no timestamps
    // (a .mjpeg file is read as jpeg_pipe, a run of JPEG pictures).
    constexpr std::array<std::string_view, 18> raw_stream_formats{
        "h264", "hevc", "vvc",   "evc",       "av1",  "obu",  "mpegvideo", "m4v", "h263",
        "h261", "vc1",  "dirac", "cavsvideo", "avs2", "avs3", "mjpeg",     "jpeg_pipe", "apv"};
    return std::find(raw_stream_formats.begin(), raw_stream_formats.end(), format_name) != raw_stream_formats.end();
}

std::vector<std::string> VideoInfo::input_options() const {
    if (!timed_raw) {
        return {};
    }
    // The rate as a double's shortest exact text: FFmpeg turns it back into
    // the fraction it came from (24000/1001 for 23.976023976023978).
    return {"-framerate", python_repr(fps)};
}

bool VideoInfo::is_variable_frame_rate() const {
    if (nominal_fps <= 0 || fps <= 0) {
        return false;
    }
    return std::abs(nominal_fps - fps) > std::max(0.01, fps * 0.001);
}

}  // namespace vml
