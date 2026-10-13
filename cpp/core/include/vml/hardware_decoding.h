// How FFmpeg decodes each video on this PC's GPUs (its -hwaccel), and the
// pixel formats and bit depths decoding gives. A port of the parts of
// vmaf_app/core/gpu.py the FFmpeg runs use.
//
// The choice is cautious: a -hwaccel likely to work for the video's codec,
// with the run falling back to software decoding if it fails to start. The
// two videos of a comparison are decided separately.
#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "vml/video.h"

namespace vml {

// The bit depth written in an FFmpeg pixel format's name (yuv420p10le: 10,
// p010le: 10, nv12: 8); 8 for a name without one, or not recognised.
int bit_depth(const std::string& pix_fmt);

// The common format videos of these pixel formats are converted to before
// they are compared: the deepest of them, 4:2:0 (yuv420p, yuv420p10le or
// yuv420p12le).
std::string analysis_pix_fmt(const std::vector<std::string>& pix_fmts);

// The system-memory format a GPU decoder's surface downloads to: p010le for
// video deeper than 8 bits, nv12 otherwise.
std::string hw_native_format(const std::string& pix_fmt);

// The FFmpeg -hwaccel to use for each video of a run; nothing for software
// decoding.
struct HardwareDecoding {
    std::optional<std::string> source;
    std::optional<std::string> distorted;

    bool uses_gpu() const { return source.has_value() || distorted.has_value(); }
    // The plan as a status line says it ("source cuda, distorted cpu", or
    // "off").
    std::string describe() const;
    bool operator==(const HardwareDecoding&) const = default;
};

// What decides the choice on this PC: the -hwaccel methods its FFmpeg was
// built with, and the GPU makers DirectX lists (NVIDIA first, then Intel,
// then AMD, the order "auto" tries them in).
struct DecodingCapabilities {
    std::set<std::string> hwaccels;
    std::vector<GpuVendor> vendors;

    // Asks `ffmpeg` (-hwaccels) and the system.
    static DecodingCapabilities detect(const std::string& ffmpeg);
};

// Whether FFmpeg's hardware decoding, as the app runs it, gives a video of
// this pixel format and size the same pictures its software decoding does:
// 4:2:0 at 8 or 10 bits, and an even width and height.
bool downloads_from_gpu(const std::string& pix_fmt, int width, int height);

// The -hwaccel to start a video with, or nothing for software decoding.
std::optional<std::string> pick_decode(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                       const std::string& codec, const std::string& pix_fmt, int width, int height);

// The plan for a comparison of two videos.
HardwareDecoding plan_hardware_decoding(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                        const VideoInfo& source, const VideoInfo& distorted);

// The -hwaccel options for one video (FFmpeg reads them as options of the
// next -i): NVIDIA's decoding on one thread, frames kept on the GPU.
std::vector<std::string> hwaccel_args(const std::optional<std::string>& hwaccel);

}  // namespace vml
