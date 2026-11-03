#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vml/video.h"

namespace vml {

// The decoders' names, as GpuFrameStream::backend and GpuDecodeUnavailableError
// give them.
inline constexpr std::string_view nvidia_decoder = "nvidia";
inline constexpr std::string_view intel_decoder = "intel";
inline constexpr std::string_view amd_decoder = "amd";
inline constexpr std::string_view apple_decoder = "apple";
// The decoder for the videos FFmpeg would decode in software.
inline constexpr std::string_view software_decoder = "software";
// VVC decoded on the GPU by vvdec-vulkan (native/vvdecvk_frames.cpp), where it is wanted (vvdec_wanted), its
// pictures handed over on the GPU: to Vulkan VMAF (the test video of a pair, beside a reference the CPU writes into
// the slot: VulkanScorer::add_shared_mixed), or copied by the GPU into host memory the decoder imported: Vship's
// frame buffers (pinned, NativeFrameStream), and GpuFrameStream::importable_buffer for the CPU's metrics, the engines
// reading host planes (HostFrames) and the pictures fed to FFmpeg (grain_feed.h).
inline constexpr std::string_view vvdec_decoder = "vvdec";

// When a decoder's pictures can stay on the GPU for the GPU's metrics
// (can_hand_over): AMD's and vvdec-vulkan's only as they were decoded.
enum class HandOver { never, when_unscaled, always };

// What one decoder can do. Features ask this table rather than compare a
// decoder's name, so that a maker's new ability is one value here. A rule that
// takes more than one fact stays a function beside its feature (can_hand_over,
// pass_feeds_cpu_metrics); a quirk of one driver stays with the code it changes,
// with its reason.
struct GpuDecoder {
    std::string_view name;

    // How FFmpeg decodes with this maker's GPU. Empty for the app's own decoders
    // (software, vvdec), which are only reached by name.
    GpuVendor vendor = GpuVendor::none;  // none for Apple's too: a Mac's route is chosen by its hwaccel (pick_hwaccel)
    std::string_view hwaccel;
    std::string_view output_format;                    // -hwaccel_output_format
    std::span<const std::string_view> hwaccel_codecs;  // those it reliably decodes; Apple's are asked of the Mac
    bool ffmpeg_downloads = false;                     // FFmpeg's decoder hands frames over in system memory itself

    // The video decoded in this process.
    std::string_view library;  // empty: none yet, so FFmpeg decodes for it, through its pipes (Apple's)
    HandOver hand_over = HandOver::never;
    bool cuda_copies = false;              // CUDA reads its pictures on the GPU: page-locks memory, copies luma there
    bool streams_share_no_memory = false;  // two of its streams may download at the same time (downloads_at_once)
    bool scales_in_two_steps = false;      // a resolution test's scaling down and back up
};

// The row of the decoder with this name: every answer no for a name not known.
const GpuDecoder& gpu_decoder(std::string_view name);

// The row of a maker with a GpuVendor (not Apple's), or of a -hwaccel; nullptr if none.
const GpuDecoder* decoder_of_vendor(GpuVendor vendor);
const GpuDecoder* decoder_of_hwaccel(std::string_view hwaccel);

// The maker a word of --gpu-decode names, and those words, in the table's order.
std::optional<GpuVendor> gpu_vendor_named(std::string_view word);
std::vector<std::string> gpu_maker_words();

// Whether libvmaf's CUDA code takes this decoder's pictures: on the GPU
// (cuda_copies), or uploads the software decoder's.
bool libvmaf_cuda_takes(std::string_view decoder);

}  // namespace vml
