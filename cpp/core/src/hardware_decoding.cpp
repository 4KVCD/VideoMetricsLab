#include "vml/hardware_decoding.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <system_error>
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
// generic hardware decoding. MPEG-2 and MPEG-4 Part 2 are decoded in
// software all the same, VideoToolbox's too (gpu_decodes_exactly).
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

// Whether FFmpeg on this PC has `hwaccel` and it decodes `codec`; D3D11VA's
// only where the GPU offers the codec's decoder profile, when that is known.
bool decodes(const DecodingCapabilities& capabilities, const std::optional<std::string>& hwaccel,
             const std::string& codec) {
    if (!hwaccel || !capabilities.hwaccels.contains(*hwaccel)) {
        return false;
    }
    if (*hwaccel == "d3d11va" && capabilities.d3d11va_codecs && !capabilities.d3d11va_codecs->contains(codec)) {
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
    capabilities.vendors = detected_gpu_vendors();
    // D3D11VA is AMD's route only. Its GPU's decoder profiles are asked while
    // FFmpeg lists its -hwaccel methods, which takes longer: on a Ryzen AI
    // Max+ 395, FFmpeg's list 0.21-0.24 s, the profiles 0.09-0.13 s. A check
    // that cannot start or fails leaves them unknown, as on other systems.
    std::future<std::optional<std::set<std::string>>> d3d11va;
    if (std::find(capabilities.vendors.begin(), capabilities.vendors.end(), GpuVendor::amd) !=
        capabilities.vendors.end()) {
        try {
            d3d11va = std::async(std::launch::async, platform::d3d11va_codecs);
        } catch (const std::system_error&) {
            // No thread for it: the profiles stay unknown.
        }
    }
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
    if (d3d11va.valid()) {
        std::optional<std::set<std::string>> codecs;
        try {
            codecs = d3d11va.get();
        } catch (const std::exception&) {
            // The profiles stay unknown.
        }
        if (capabilities.hwaccels.contains("d3d11va")) {
            capabilities.d3d11va_codecs = std::move(codecs);
        }
    }
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

bool gpu_decodes_exactly(const std::string& codec) {
    // Against FFmpeg's decode of a 3 Mb/s 1080p MPEG-2 encode with one
    // keyframe in 243 frames, NVIDIA's decoder (RTX 5090) gave 63.8 dB PSNR
    // at the keyframe and 37.2 dB at the last frame, Intel's (Core Ultra 9
    // 285K) 64.4 dB at worst; NVIDIA's decode of an MPEG-4 Part 2 encode of
    // the same video, 56.3 dB at worst. The other codecs' transforms are
    // exact integer ones.
    const std::string name = ascii_lower(codec);
    return name != "mpeg2video" && name != "mpeg4";
}

std::optional<std::string> pick_decode(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                       const std::string& codec, const std::string& pix_fmt, int width, int height,
                                       MacGpuDecoding mac) {
    if (!downloads_from_gpu(pix_fmt, width, height) || !gpu_decodes_exactly(codec)) {
        return std::nullopt;
    }
    if (mac == MacGpuDecoding::deep) {
        mac = bit_depth(pix_fmt) > 8 ? MacGpuDecoding::on : MacGpuDecoding::off;
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

bool h264_444_profile(const VideoInfo& video) {
    return ascii_lower(video.codec_name) == "h264" && video.profile.find("4:4:4") != std::string::npos;
}

std::optional<std::string> pick_decode(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                       const VideoInfo& video, MacGpuDecoding mac) {
    std::optional<std::string> hwaccel =
        pick_decode(capabilities, vendor, video.codec_name, video.pix_fmt, video.width, video.height, mac);
    // No profile is a video not probed (a saved result's, which Video Compare
    // can play): its H.264 may be lossless too.
    const bool unknown_h264 = ascii_lower(video.codec_name) == "h264" && video.profile.empty();
    if (hwaccel && (h264_444_profile(video) || unknown_h264) && hwaccel != "cuda" && hwaccel != "videotoolbox") {
        return std::nullopt;
    }
    return hwaccel;
}

std::string software_decoding_reason(const VideoInfo& info, bool gpu_decode) {
    if (!gpu_decode) {
        return "GPU decoding is off";
    }
    if (info.scaled_to) {
        return "its picture size changes partway through";
    }
    if (!downloads_from_gpu(info.pix_fmt, info.width, info.height)) {
        return "GPU decoding takes 4:2:0 at 8 or 10 bits and an even size, not " + info.pix_fmt + " at " +
               std::to_string(info.width) + "x" + std::to_string(info.height);
    }
    if (!gpu_decodes_exactly(info.codec_name)) {
        return "the GPUs decode " + info.codec_name + " to other pictures than FFmpeg's";
    }
    if (ascii_lower(info.codec_name) == "h264" && bit_depth(info.pix_fmt) > 8) {
        return "this GPU does not decode H.264 deeper than 8 bits";
    }
    if (h264_444_profile(info)) {
        return "this GPU does not decode H.264's " + info.profile + " profile exactly";
    }
    if (ascii_lower(info.codec_name) == "h264" && info.profile.empty()) {
        return "its H.264 profile is not known, and this GPU does not decode every one exactly";
    }
    return "no GPU decoder here for " + info.codec_name;
}

HardwareDecoding plan_hardware_decoding(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                        const VideoInfo& source, const VideoInfo& distorted, MacGpuDecoding mac) {
    // A video whose size changes (VideoInfo::scaled_to) is decoded in
    // software: FFmpeg rebuilds its filters for a GPU decoder's surfaces of a
    // new size whatever it is told (-reinit_filter 0).
    const auto pick = [&](const VideoInfo& video) {
        return video.scaled_to ? std::nullopt : pick_decode(capabilities, vendor, video, mac);
    };
    return HardwareDecoding{pick(source), pick(distorted)};
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
