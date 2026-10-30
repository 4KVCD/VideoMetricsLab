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

// The Vship scores' ids are version 2 for Vship-fast's, which computes as
// libjxl and pycvvdp do, and version 1 for Vship 5.1's (stock Vship 5.1.2's
// now, and the app's own before Vship-fast). A video's row is set to one of
// the two builds (VshipBuild) and reads only that build's saved scores: they
// differ slightly. A version 1 score is calculated again for a pair whose
// colours a later fix reads otherwise (stale_score in metric_cache.cpp).

// CVVDP, which only Vship calculates, on the GPU: Vship-fast's, Vship 5.1's.
inline constexpr std::string_view cvvdp = "cvvdp-vship-gpu-v2";
inline constexpr std::string_view vship51_cvvdp = "cvvdp-vship-gpu-v1";

// SSIMULACRA2 or Butteraugli (`key`): calculated by Vship on the GPU, by
// libjxl's tools on the CPU, and the requests that take a score from either,
// one per Vship build. Vship 5.1's GPU ids are "<key>-vship-gpu-v1", and
// before that had its version inside ("<key>-vship-5.1.0-gpu-v1").
inline std::string vship_gpu(std::string_view key) { return std::string(key) + "-vship-gpu-v2"; }
inline std::string vship51_gpu(std::string_view key) { return std::string(key) + "-vship-gpu-v1"; }
inline bool is_vship51_gpu(std::string_view key, std::string_view id) {
    const std::string prefix = std::string(key) + "-vship-";
    return id.starts_with(prefix) && id.ends_with("-gpu-v1");
}
inline std::string libjxl_cpu(std::string_view key) { return std::string(key) + "-libjxl-cpu-v1"; }
inline constexpr std::string_view gpu_or_cpu_suffix = "-auto-or-libjxl-cpu-v1";
inline std::string gpu_or_cpu(std::string_view key) { return std::string(key) + std::string(gpu_or_cpu_suffix); }
inline constexpr std::string_view vship51_gpu_or_cpu_suffix = "-vship51-or-libjxl-cpu-v1";
inline std::string vship51_gpu_or_cpu(std::string_view key) {
    return std::string(key) + std::string(vship51_gpu_or_cpu_suffix);
}

// Whether a score's id is one of Vship 5.1's on the GPU.
inline bool is_vship51_id(std::string_view id) {
    return id == vship51_cvvdp || is_vship51_gpu("ssimulacra2", id) || is_vship51_gpu("butteraugli", id);
}
// Whether a score's id is one of Vship-fast's.
inline bool is_vship_fast_id(std::string_view id) {
    return id == cvvdp || id == vship_gpu("ssimulacra2") || id == vship_gpu("butteraugli");
}

// A video's film grain as a score was made with it, where it records that
// (film_grain_identity in film_grain_scoring.h): H.274's added by VVdeC,
// AFGS1 by FFmpeg's decoders, AV1's by the decoders, which all add it alike;
// or the grain left out.
inline constexpr std::string_view film_grain_h274 = "h274-vvdec-v1";
inline constexpr std::string_view film_grain_afgs1 = "afgs1-ffmpeg-v1";
inline constexpr std::string_view film_grain_av1 = "av1-v1";
inline constexpr std::string_view film_grain_left_out = "without";

// A VVC video decoded by VVdeC, as every one is where its pictures can reach
// the route: before, FFmpeg's own decoder decoded those not scored with
// H.274's grain, and some wrongly. Where they cannot, FFmpeg's (vvc_ffmpeg).
inline constexpr std::string_view vvc_vvdec = "vvdec-v1";
inline constexpr std::string_view vvc_ffmpeg = "vvc-ffmpeg-v1";

// How the videos' colour tags were read, as a score records it. A Vship GPU
// score: as Vship 5.1.1's FFVship reads them; "/2" marks the version that
// reads a video without chroma siting with the other video's
// (shared_chroma_location in vship.h). A CPU SSIMULACRA2 or Butteraugli
// score: as Vship reads them, with Vship's Butteraugli 3-norm; "/2" marks
// the version that converts a scaled video to RGB after its tags are set.
// "/3", both: the version that reads a video without a colour matrix with
// the other's, or both untagged with the reference's guess
// (with_shared_matrix in colour.h). "/4", CPU: the version that converts an
// 8-bit video to RGB as precisely as a deeper one (to_rgb in
// perceptual_cpu.cpp). A score of an older mapping is calculated again
// where it would differ (stale_score in metric_cache.cpp).
inline constexpr std::string_view vship_colour_tags = "ffvship-5.1.1/3";
inline constexpr std::string_view cpu_colour_tags = "vship-5.1.1/4";
// The CPU mapping before "/4", whose scores differ only for a pair with an
// 8-bit video: FFmpeg converted 4:2:0 and 4:2:2 ones to RGB in 8 bits, about
// 2 steps of 255 too dark.
inline constexpr std::string_view cpu_colour_tags_8_bit_rgb = "vship-5.1.1/3";
// The mappings before "/3", whose scores differ only for a pair whose
// videos' matrices with_shared_matrix reads otherwise (matrix_shared).
inline constexpr std::string_view vship_colour_tags_unshared_matrix = "ffvship-5.1.1/2";
inline constexpr std::string_view cpu_colour_tags_unshared_matrix = "vship-5.1.1/2";
// The Vship mapping before "/2", whose scores differ only for a pair with
// one video's chroma siting given and the other's not.
inline constexpr std::string_view vship_colour_tags_unshared_siting = "ffvship-5.1.1";

}  // namespace vml::score_identity
