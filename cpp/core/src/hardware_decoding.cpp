#include "vml/hardware_decoding.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <mutex>
#include <regex>
#include <utility>

#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

#ifdef __APPLE__
#include <VideoToolbox/VideoToolbox.h>
#endif

namespace vml {

namespace {

// For each -hwaccel, the codecs it reliably decodes through FFmpeg's
// generic hardware decoding.
const std::map<std::string, std::set<std::string>>& hwaccel_codecs() {
    static const std::map<std::string, std::set<std::string>> codecs{
        {"cuda", {"h264", "hevc", "vp8", "vp9", "mpeg2video", "mpeg4", "av1", "vc1"}},
        {"qsv", {"h264", "hevc", "vp8", "vp9", "mpeg2video", "av1"}},
        {"d3d11va", {"h264", "hevc", "vp9", "mpeg2video", "vc1", "av1"}},
    };
    return codecs;
}

std::optional<std::string> preferred_hwaccel(GpuVendor vendor) {
    switch (vendor) {
        case GpuVendor::nvidia:
            return "cuda";
        case GpuVendor::intel:
            return "qsv";
        case GpuVendor::amd:
            return "d3d11va";
        default:
            return std::nullopt;
    }
}

// The codecs this Mac's GPU decodes through VideoToolbox, as FFmpeg names
// them; none elsewhere. VP9's and AV1's decoders are supplemental ones, there
// only once registered, as libavcodec registers them before it decodes
// (the same check as Video Compare's, NativeDevice::gpu_decoders).
std::set<std::string> codecs_this_mac_decodes() {
    std::set<std::string> codecs;
#ifdef __APPLE__
    VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_VP9);
    VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_AV1);
    const std::array<std::pair<CMVideoCodecType, const char*>, 5> known{{
        {kCMVideoCodecType_H264, "h264"},
        {kCMVideoCodecType_HEVC, "hevc"},
        {kCMVideoCodecType_VP9, "vp9"},
        {kCMVideoCodecType_AV1, "av1"},
        {kCMVideoCodecType_MPEG2Video, "mpeg2video"},
    }};
    for (const auto& [type, name] : known) {
        if (VTIsHardwareDecodeSupported(type)) {
            codecs.insert(name);
        }
    }
#endif
    return codecs;
}

// Whether FFmpeg on this PC has `hwaccel` and it decodes `codec`.
bool decodes(const DecodingCapabilities& capabilities, const std::optional<std::string>& hwaccel,
             const std::string& codec) {
    if (!hwaccel || !capabilities.hwaccels.contains(*hwaccel)) {
        return false;
    }
    const auto supported = hwaccel_codecs().find(*hwaccel);
    return supported != hwaccel_codecs().end() && supported->second.contains(codec);
}

// The -hwaccel to try for a codec on this GPU maker (or the first one
// found, for "auto"), or nothing for software decoding. On a Mac, for a
// caller that allows it, VideoToolbox's where the GPU decodes the codec.
std::optional<std::string> pick_hwaccel(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                        const std::string& codec_name, MacGpuDecoding mac) {
    const std::string codec = ascii_lower(codec_name);
    if (vendor == GpuVendor::none) {
        return std::nullopt;
    }
    if (mac == MacGpuDecoding::on && capabilities.hwaccels.contains("videotoolbox") &&
        capabilities.videotoolbox_codecs.contains(codec)) {
        return "videotoolbox";
    }
    if (vendor == GpuVendor::automatic) {
        for (const GpuVendor found : capabilities.vendors) {
            if (const std::optional<std::string> candidate = preferred_hwaccel(found);
                decodes(capabilities, candidate, codec)) {
                return candidate;
            }
        }
        return std::nullopt;
    }
    const std::optional<std::string> candidate = preferred_hwaccel(vendor);
    return decodes(capabilities, candidate, codec) ? candidate : std::nullopt;
}

}  // namespace

int bit_depth(const std::string& pix_fmt) {
    const std::string name = ascii_lower(pix_fmt);
    if (name.empty()) {
        return 8;
    }
    // The semi-planar hardware-surface formats: "p010" means 10 significant
    // bits stored in 16.
    constexpr std::array<std::pair<const char*, int>, 3> surfaces{{{"p016", 16}, {"p012", 12}, {"p010", 10}}};
    for (const auto& [prefix, depth] : surfaces) {
        if (name.starts_with(prefix)) {
            return depth;
        }
    }
    // The depth digits come right after the planar "p", at the end of the
    // name (yuv420p10le, gbrp12be), which keeps nv12 and rgb24 from being
    // read as deep; single-plane formats put them after the plane's name.
    static const std::regex planar(R"(p(\d{1,2})(?:le|be)?$)");
    static const std::regex gray(R"(^(?:gray|ya)(\d{1,2})(?:le|be)?$)");
    std::smatch match;
    if (std::regex_search(name, match, planar) || std::regex_search(name, match, gray)) {
        return std::stoi(match[1].str());
    }
    return 8;
}

std::string analysis_pix_fmt(const std::vector<std::string>& pix_fmts) {
    int depth = 8;
    for (const std::string& format : pix_fmts) {
        depth = std::max(depth, bit_depth(format));
    }
    if (depth <= 8) {
        return "yuv420p";
    }
    // libvmaf takes up to 12 bits, so deeper video is compared at 12.
    return depth <= 10 ? "yuv420p10le" : "yuv420p12le";
}

std::string hw_native_format(const std::string& pix_fmt) { return bit_depth(pix_fmt) > 8 ? "p010le" : "nv12"; }

std::string HardwareDecoding::describe() const {
    if (!uses_gpu()) {
        return "off";
    }
    return "source " + source.value_or("cpu") + ", distorted " + distorted.value_or("cpu");
}

DecodingCapabilities DecodingCapabilities::detect(const std::string& ffmpeg) {
    DecodingCapabilities capabilities;
    try {
        const platform::ProcessOutput output =
            platform::run_program({ffmpeg, "-hide_banner", "-hwaccels"}, std::chrono::seconds(15));
        // The names are listed one per line after "Hardware acceleration
        // methods:".
        bool listing = false;
        std::size_t start = 0;
        while (start < output.standard_output.size()) {
            std::size_t end = output.standard_output.find('\n', start);
            if (end == std::string::npos) {
                end = output.standard_output.size();
            }
            std::string line = output.standard_output.substr(start, end - start);
            line.erase(0, line.find_first_not_of(" \t\r"));
            line.erase(line.find_last_not_of(" \t\r") + 1);
            if (ascii_lower(line).starts_with("hardware acceleration methods")) {
                listing = true;
            } else if (listing && !line.empty()) {
                capabilities.hwaccels.insert(line);
            }
            start = end + 1;
        }
    } catch (const platform::ProcessError&) {
        // No FFmpeg to ask: no hardware decoding.
    }
    capabilities.vendors = detected_gpu_vendors();
    capabilities.videotoolbox_codecs = codecs_this_mac_decodes();
    return capabilities;
}

DecodingCapabilities DecodingCapabilities::remembered(const std::string& ffmpeg) {
    static std::mutex mutex;
    static std::map<std::string, DecodingCapabilities> known;
    {
        const std::lock_guard lock(mutex);
        if (const auto found = known.find(ffmpeg); found != known.end()) {
            return found->second;
        }
    }
    // Asked outside the lock: two threads may both ask the first time, and
    // give the same answer.
    DecodingCapabilities capabilities = detect(ffmpeg);
    const std::lock_guard lock(mutex);
    return known.try_emplace(ffmpeg, std::move(capabilities)).first->second;
}

std::vector<GpuVendor> detected_gpu_vendors() {
    std::vector<GpuVendor> present;
    for (const unsigned id : platform::gpu_vendor_ids()) {
        present.push_back(vendor_of_pci_id(id));
    }
    std::vector<GpuVendor> vendors;
    for (const GpuVendor vendor : {GpuVendor::nvidia, GpuVendor::intel, GpuVendor::amd}) {
        if (std::find(present.begin(), present.end(), vendor) != present.end()) {
            vendors.push_back(vendor);
        }
    }
    return vendors;
}

bool downloads_from_gpu(const std::string& pix_fmt, int width, int height) {
    // An odd-sized video decodes on NVIDIA's GPUs to the decoder's own,
    // larger picture with its chroma a row out; the software decode is the
    // video's own picture.
    if ((width & 1) != 0 || (height & 1) != 0) {
        return false;
    }
    const std::string name = ascii_lower(pix_fmt);
    if (name.empty()) {
        return true;  // unknown: left to try
    }
    // 4:2:0 at 8 or 10 bits (the NV12 and P010 surfaces); anything else
    // decodes to a surface the download fails on.
    return name == "nv12" || name == "p010le" || name == "p010be" ||
           ((name.starts_with("yuv420p") || name.starts_with("yuvj420p")) && bit_depth(name) <= 10);
}

std::optional<std::string> pick_decode(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                       const std::string& codec, const std::string& pix_fmt, int width, int height,
                                       MacGpuDecoding mac) {
    if (!downloads_from_gpu(pix_fmt, width, height)) {
        return std::nullopt;
    }
    std::optional<std::string> hwaccel = pick_hwaccel(capabilities, vendor, codec, mac);
    // Intel's, AMD's and Apple's GPUs do not decode H.264 deeper than 8 bits
    // (FFmpeg then decodes it in software and the download fails); NVIDIA's
    // RTX 50 series does.
    if (hwaccel != "cuda" && ascii_lower(codec) == "h264" && !pix_fmt.empty() && bit_depth(pix_fmt) > 8) {
        return std::nullopt;
    }
    return hwaccel;
}

HardwareDecoding plan_hardware_decoding(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                        const VideoInfo& source, const VideoInfo& distorted, MacGpuDecoding mac) {
    return HardwareDecoding{
        pick_decode(capabilities, vendor, source.codec_name, source.pix_fmt, source.width, source.height, mac),
        pick_decode(capabilities, vendor, distorted.codec_name, distorted.pix_fmt, distorted.width, distorted.height,
                    mac),
    };
}

std::vector<std::string> hwaccel_args(const std::optional<std::string>& hwaccel) {
    if (!hwaccel) {
        return {};
    }
    std::vector<std::string> args;
    if (*hwaccel == "cuda") {
        // NVIDIA's decoding gains nothing from FFmpeg's frame threads, which
        // each hold pictures of their own: 260 MB less memory for 4K.
        args = {"-threads", "1"};
    }
    // The frames stay on the GPU for a graph that downloads them. The
    // format's name is the hwaccel's own, except D3D11VA's ("d3d11") and
    // VideoToolbox's ("videotoolbox_vld").
    const std::string output_format = *hwaccel == "d3d11va"        ? "d3d11"
                                      : *hwaccel == "videotoolbox" ? "videotoolbox_vld"
                                                                   : *hwaccel;
    args.insert(args.end(), {"-hwaccel", *hwaccel, "-hwaccel_output_format", output_format});
    return args;
}

}  // namespace vml
