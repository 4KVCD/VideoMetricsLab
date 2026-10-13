// What scores frame pairs outside FFmpeg, whichever library does it: the
// Vulkan engine for VMAF and NEG (vmaf_vulkan.h), libvmaf's own CPU
// extractors for PSNR, SSIM and XPSNR (libvmaf_cpu.h). The pipe route
// (gpu_vmaf.h's GpuAttempt) feeds either the frames each video's FFmpeg
// writes, as the Python app's vmaf_gpu.GpuAttempt does.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vml {

// The numbers of the frames scored, and each metric's scores for them, in
// the order the metrics were given.
struct PairScores {
    std::vector<unsigned> frames;
    std::vector<std::pair<std::string, std::vector<double>>> by_metric;
};

class PairScorer {
public:
    virtual ~PairScorer() = default;

    // The bytes of one packed frame: 4:2:0, as FFmpeg's rawvideo writes it,
    // 16-bit little-endian samples above 8 bits.
    virtual std::size_t frame_bytes() const = 0;

    // Scores one more pair. Its frame number is the number of pairs given
    // before it.
    virtual void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) = 0;

    // The scores, once every pair has been given.
    virtual PairScores finish() = 0;
};

}  // namespace vml
