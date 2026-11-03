#include "vml/gpu_decoders.h"

#include <algorithm>
#include <array>

namespace vml {

namespace {

// The codecs each -hwaccel reliably decodes through FFmpeg's generic hardware
// decoding. MPEG-2 and MPEG-4 Part 2 are decoded in software all the same,
// VideoToolbox's too (gpu_decodes_exactly).
constexpr std::array<std::string_view, 8> cuda_codecs{"h264", "hevc", "vp8", "vp9", "mpeg2video", "mpeg4", "av1", "vc1"};
constexpr std::array<std::string_view, 6> qsv_codecs{"h264", "hevc", "vp8", "vp9", "mpeg2video", "av1"};
constexpr std::array<std::string_view, 6> d3d11va_codecs{"h264", "hevc", "vp9", "mpeg2video", "vc1", "av1"};

constexpr std::array<GpuDecoder, 6> decoders{{
    {.name = nvidia_decoder,
     .vendor = GpuVendor::nvidia,
     .hwaccel = "cuda",
     .output_format = "cuda",
     .hwaccel_codecs = cuda_codecs,
     .ffmpeg_downloads = true,
     .library = "nvdec_frames",
     .hand_over = HandOver::always,
     .cuda_copies = true,
     .streams_share_no_memory = true,
     .scales_in_two_steps = true},
    {.name = intel_decoder,
     .vendor = GpuVendor::intel,
     .hwaccel = "qsv",
     .output_format = "qsv",
     .hwaccel_codecs = qsv_codecs,
     .library = "vpl_frames"},
    {.name = amd_decoder,
     .vendor = GpuVendor::amd,
     .hwaccel = "d3d11va",
     .output_format = "d3d11",
     .hwaccel_codecs = d3d11va_codecs,
     .ffmpeg_downloads = true,
     .library = "amf_frames",
     .hand_over = HandOver::when_unscaled},
    {.name = apple_decoder, .hwaccel = "videotoolbox", .output_format = "videotoolbox_vld", .ffmpeg_downloads = true},
    {.name = software_decoder, .library = "software_frames", .streams_share_no_memory = true},
    {.name = vvdec_decoder,
     .library = "vvdecvk_frames",
     .hand_over = HandOver::when_unscaled,
     .streams_share_no_memory = true},
}};

constexpr GpuDecoder unknown_decoder{};

template <typename Matches>
const GpuDecoder* find_decoder(Matches matches) {
    const auto found = std::find_if(decoders.begin(), decoders.end(), matches);
    return found == decoders.end() ? nullptr : &*found;
}

}  // namespace

const GpuDecoder& gpu_decoder(std::string_view name) {
    const GpuDecoder* found = find_decoder([name](const GpuDecoder& decoder) { return decoder.name == name; });
    return found ? *found : unknown_decoder;
}

const GpuDecoder* decoder_of_vendor(GpuVendor vendor) {
    // Every row without a maker of its own says none: never the answer to it.
    if (vendor == GpuVendor::none) {
        return nullptr;
    }
    return find_decoder([vendor](const GpuDecoder& decoder) { return decoder.vendor == vendor; });
}

const GpuDecoder* decoder_of_hwaccel(std::string_view hwaccel) {
    if (hwaccel.empty()) {
        return nullptr;
    }
    return find_decoder([hwaccel](const GpuDecoder& decoder) { return decoder.hwaccel == hwaccel; });
}

std::optional<GpuVendor> gpu_vendor_named(std::string_view word) {
    const GpuDecoder& decoder = gpu_decoder(word);
    return decoder.vendor == GpuVendor::none ? std::nullopt : std::optional(decoder.vendor);
}

std::vector<std::string> gpu_maker_words() {
    std::vector<std::string> words;
    for (const GpuDecoder& decoder : decoders) {
        if (decoder.vendor != GpuVendor::none) {
            words.emplace_back(decoder.name);
        }
    }
    return words;
}

bool libvmaf_cuda_takes(std::string_view decoder) {
    return gpu_decoder(decoder).cuda_copies || decoder == software_decoder;
}

}  // namespace vml
