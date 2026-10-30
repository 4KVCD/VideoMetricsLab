#include "vml/video.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

#include "vml/app_decoders.h"
#include "vml/film_grain_scoring.h"
#include "vml/geometry.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

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
    // (a .mjpeg file is read as jpeg_pipe, a run of JPEG pictures), and of
    // raw pictures (rawvideo, a .yuv file).
    constexpr std::array<std::string_view, 19> raw_stream_formats{
        "h264", "hevc", "vvc",   "evc",       "av1",  "obu",  "mpegvideo", "m4v", "h263",   "h261",
        "vc1",  "dirac", "cavsvideo", "avs2", "avs3", "mjpeg", "jpeg_pipe", "apv", "rawvideo"};
    return std::find(raw_stream_formats.begin(), raw_stream_formats.end(), format_name) != raw_stream_formats.end();
}

bool VideoInfo::full_range() const {
    const std::string range = ascii_lower(color_range);
    if (range == "pc" || range == "jpeg" || range == "full") {
        return true;
    }
    const bool untagged = range.empty() || range == "unknown" || range == "unspecified";
    return untagged && ascii_lower(pix_fmt).starts_with("yuvj");
}

std::vector<std::string> VideoInfo::input_options() const {
    std::vector<std::string> options;
    if (scaled_to) {
        // Without it, FFmpeg would rebuild its filters at the size change,
        // starting a metric's filter again, or its own scale.
        options = {"-reinit_filter", "0"};
    }
    if (!timed_raw) {
        return options;
    }
    if (format_name == "rawvideo") {
        // Pictures and nothing else, of the size and format the command line
        // gave.
        options.insert(options.end(), {"-f", "rawvideo", "-video_size",
                                       std::to_string(width) + "x" + std::to_string(height), "-pixel_format", pix_fmt});
    }
    // The rate as a double's shortest exact text: FFmpeg turns it back into
    // the fraction it came from (24000/1001 for 23.976023976023978).
    options.insert(options.end(), {"-framerate", python_repr(fps)});
    return options;
}

std::vector<std::string> VideoInfo::decode_options() const {
    std::vector<std::string> options = input_options();
    if (const AppDecodedCodec* codec = app_decoded_codec(ascii_lower(codec_name)); codec && !codec->ffmpeg_decodes) {
        throw FilmGrainRefusal("FFmpeg does not decode " + codec_label(ascii_lower(codec_name)) + ", only " +
                               std::string(codec->decoder) + " in the app does, and here FFmpeg would decode it");
    }
    const FilmGrainScoring grain = effective_film_grain(*this);
    if (grain.scoring == GrainScoring::without) {
        options.insert(options.end(), {"-export_side_data", "film_grain"});
    } else if (only_the_app_adds(grain)) {
        throw FilmGrainRefusal(
            "it is scored with its H.274 film grain, which only the app's own decoder adds as it should (FFmpeg adds "
            "it at 8 bits only, rounded down), and here FFmpeg would decode it: score it without its film grain, or "
            "calculate the metric where the app decodes the video itself (VMAF on the GPU)");
    }
    return options;
}

bool VideoInfo::is_variable_frame_rate() const {
    if (nominal_fps <= 0 || fps <= 0) {
        return false;
    }
    return std::abs(nominal_fps - fps) > std::max(0.01, fps * 0.001);
}

}  // namespace vml
