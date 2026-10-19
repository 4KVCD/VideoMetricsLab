// The texts a saved score is identified by: its implementation's
// compatibility id, and the colour-tag mapping it was made with. A saved
// score is reused only where these match what a run would record now, so
// each text is written once, here, for both the code that saves scores and
// the code that reads them. Changing one makes every score saved under the
// old text be calculated again.
#pragma once

#include <string>
#include <string_view>

namespace vml::score_identity {

// FFmpeg's libvmaf filter, or libvmaf in the app: VMAF v0.6.1, VMAF NEG,
// VMAF v1, PSNR and SSIM.
inline constexpr std::string_view libvmaf = "ffmpeg-libvmaf-v1";

// XPSNR weighted by the source's activity, as XPSNR is defined. (Version 1
// was weighted by the test video's.)
inline constexpr std::string_view xpsnr = "ffmpeg-xpsnr-v2";

// CVVDP, which only Vship calculates, on the GPU.
inline constexpr std::string_view cvvdp = "cvvdp-vship-gpu-v1";

// SSIMULACRA2 or Butteraugli (`key`): calculated by Vship on the GPU, by
// libjxl's tools on the CPU, and the request that takes a score from either.
inline std::string vship_gpu(std::string_view key) { return std::string(key) + "-vship-gpu-v1"; }
inline std::string libjxl_cpu(std::string_view key) { return std::string(key) + "-libjxl-cpu-v1"; }
inline constexpr std::string_view gpu_or_cpu_suffix = "-auto-or-libjxl-cpu-v1";
inline std::string gpu_or_cpu(std::string_view key) { return std::string(key) + std::string(gpu_or_cpu_suffix); }

// How the videos' colour tags were read, as a score records it. A Vship GPU
// score: as Vship 5.1.1's FFVship reads them; "/2" marks the version that
// reads a video without chroma siting with the other video's
// (shared_chroma_location in vship.h). A CPU SSIMULACRA2 or Butteraugli
// score: as Vship reads them, with Vship's Butteraugli 3-norm; "/2" marks
// the version that converts a scaled video to RGB after its tags are set.
// "/3", both: the version that reads a video without a colour matrix with
// the other's, or both untagged with the reference's guess
// (with_shared_matrix in colour.h). A score without its mapping is
// calculated again where it would differ (stale_vship_score in
// metric_cache.cpp).
inline constexpr std::string_view vship_colour_tags = "ffvship-5.1.1/3";
inline constexpr std::string_view cpu_colour_tags = "vship-5.1.1/3";
// The mappings before "/3", whose scores differ only for a pair whose
// videos' matrices with_shared_matrix reads otherwise (matrix_shared).
inline constexpr std::string_view vship_colour_tags_unshared_matrix = "ffvship-5.1.1/2";
inline constexpr std::string_view cpu_colour_tags_unshared_matrix = "vship-5.1.1/2";
// The Vship mapping before "/2", whose scores differ only for a pair with
// one video's chroma siting given and the other's not.
inline constexpr std::string_view vship_colour_tags_unshared_siting = "ffvship-5.1.1";

}  // namespace vml::score_identity
