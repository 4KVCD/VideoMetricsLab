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
// built with, the GPU makers DirectX lists (NVIDIA first, then Intel, then
// AMD, the order "auto" tries them in), on a Mac the codecs its GPU decodes
// through VideoToolbox, as FFmpeg names them (h264, hevc, ...), and on
// Windows the codecs D3D11VA can decode on its GPU.
struct DecodingCapabilities {
    std::set<std::string> hwaccels;
    std::vector<GpuVendor> vendors;
    std::set<std::string> videotoolbox_codecs;
    // The codecs for which the GPU FFmpeg's D3D11VA decoding uses offers a
    // decoder profile (platform::d3d11va_codecs; a Radeon 8060S has none for
    // VC-1 or MPEG-2). Without one, FFmpeg's D3D11VA decoding fails to start,
    // and the run goes on only after its software retries. Nothing where it is
    // not known: D3D11VA is then tried for every codec it decodes.
    std::optional<std::set<std::string>> d3d11va_codecs;

    // Asks `ffmpeg` (-hwaccels) and the system.
    static DecodingCapabilities detect(const std::string& ffmpeg);
    // The same, asked once per FFmpeg in this process (another FFmpeg chosen
    // in Settings is asked again): playback plans each video it starts, and
    // asking at each seek would start a process each time. Any thread.
    static DecodingCapabilities remembered(const std::string& ffmpeg);
};

// The makers of the GPUs DirectX lists (none on other systems so far): NVIDIA
// first, then Intel, then AMD (the Python app's detected_gpu_vendors).
std::vector<GpuVendor> detected_gpu_vendors();

// Whether FFmpeg's hardware decoding, as the app runs it, gives a video of
// this pixel format and size the same pictures its software decoding does:
// 4:2:0 at 8 or 10 bits, and an even width and height.
bool downloads_from_gpu(const std::string& pix_fmt, int width, int height);

// Whether the GPUs' decoders decode `codec` (FFmpeg's name) to the pictures
// FFmpeg's software decoder gives. Not MPEG-2 or MPEG-4 Part 2: their inverse
// DCT is specified only to within an error (IEEE 1180), so each GPU maker's
// decoder rounds it its own way, and the difference grows from one predicted
// picture to the next until the next keyframe. Their scores would depend on
// the GPU, so they are decoded in software.
bool gpu_decodes_exactly(const std::string& codec);

// Which videos a caller decodes on a Mac's GPU (FFmpeg's -hwaccel
// videotoolbox), where VideoToolbox decodes the video; scores and crops were
// checked identical to software decoding. Through FFmpeg, VideoToolbox
// decodes one picture after another: on an M1 Pro an 8-bit video comes at
// 160-250 pictures a second at 1080p whatever the codec, where the CPU's
// cores decode H.264 at about 1000, while 10-bit HEVC comes faster than the
// CPU decodes it (about 460 against 280).
// - on: every video the GPU decodes. The Vship pass, whose metrics are
//   slower than that anyway, and which so saves the CPU.
// - deep: only a video deeper than 8 bits. The libvmaf routes (VMAF, VMAF
//   NEG, VMAF v1, PSNR, SSIM, XPSNR), the resolution test and black-bar
//   detection, whose pace the decoding sets.
// - off: none. Playback's FFmpeg fallback: on a Mac FFmpeg has no Vulkan
//   device for its libplacebo modes, so only its CPU mode runs, which decodes
//   in software, and VideoToolbox would only add an attempt that fails.
//   Video Compare's own player decodes on the GPU.
enum class MacGpuDecoding { off, deep, on };

// The -hwaccel to start a video with, or nothing for software decoding
// (always for a codec gpu_decodes_exactly refuses).
std::optional<std::string> pick_decode(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                       const std::string& codec, const std::string& pix_fmt, int width, int height,
                                       MacGpuDecoding mac = MacGpuDecoding::deep);

// The plan for a comparison of two videos.
HardwareDecoding plan_hardware_decoding(const DecodingCapabilities& capabilities, GpuVendor vendor,
                                        const VideoInfo& source, const VideoInfo& distorted,
                                        MacGpuDecoding mac = MacGpuDecoding::deep);

// The -hwaccel options for one video (FFmpeg reads them as options of the
// next -i): NVIDIA's decoding on one thread, frames kept on the GPU (in
// VideoToolbox's own surfaces on a Mac, videotoolbox_vld).
std::vector<std::string> hwaccel_args(const std::optional<std::string>& hwaccel);

}  // namespace vml
