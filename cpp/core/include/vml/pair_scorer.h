// What scores frame pairs outside FFmpeg, whichever library does it: the
// Vulkan engine for VMAF and NEG (vmaf_vulkan.h), libvmaf's own CPU
// extractors for PSNR, SSIM and XPSNR (libvmaf_cpu.h). The pipe route
// (gpu_vmaf.h's GpuAttempt) feeds either the frames each video's FFmpeg
// writes, as the Python app's vmaf_gpu.GpuAttempt does.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

// A plane of a picture where it is: its first sample, and the bytes from one
// row to the next.
struct Plane {
    const std::uint8_t* data = nullptr;
    std::ptrdiff_t stride = 0;
};

// A 4:2:0 frame's Y, U and V planes, wherever each one is.
using FramePlanes = std::array<Plane, 3>;

// Where the planes of a packed frame are: 4:2:0, as FFmpeg's rawvideo writes
// it, 16-bit little-endian samples above 8 bits, each plane's rows packed and
// the planes one after the other.
class PackedFrame {
public:
    PackedFrame() = default;
    PackedFrame(int width, int height, int bit_depth) {
        const std::size_t sample = bit_depth <= 8 ? 1 : 2;
        const auto columns = static_cast<std::size_t>(width), rows = static_cast<std::size_t>(height);
        const std::size_t chroma_bytes = ((columns + 1) / 2) * ((rows + 1) / 2) * sample;
        offsets_ = {0, columns * rows * sample, columns * rows * sample + chroma_bytes};
        strides_ = {static_cast<std::ptrdiff_t>(columns * sample),
                    static_cast<std::ptrdiff_t>((columns + 1) / 2 * sample),
                    static_cast<std::ptrdiff_t>((columns + 1) / 2 * sample)};
        bytes_ = offsets_[2] + chroma_bytes;
    }

    // The bytes of the whole frame.
    std::size_t bytes() const { return bytes_; }

    // The planes of such a frame at `frame`.
    FramePlanes planes(const std::uint8_t* frame) const {
        return {Plane{frame + offsets_[0], strides_[0]}, Plane{frame + offsets_[1], strides_[1]},
                Plane{frame + offsets_[2], strides_[2]}};
    }

private:
    std::array<std::size_t, 3> offsets_{};
    std::array<std::ptrdiff_t, 3> strides_{};
    std::size_t bytes_ = 0;
};

// Copies `rows` rows of `row_bytes` from `plane` to `target`, rows packed.
inline void pack_plane(Plane plane, std::size_t rows, std::size_t row_bytes, std::uint8_t* target) {
    if (plane.stride == static_cast<std::ptrdiff_t>(row_bytes)) {
        std::memcpy(target, plane.data, rows * row_bytes);
        return;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        std::memcpy(target + row * row_bytes, plane.data + static_cast<std::ptrdiff_t>(row) * plane.stride,
                    row_bytes);
    }
}

class PairScorer {
public:
    virtual ~PairScorer() = default;

    // The bytes of one packed frame (PackedFrame).
    virtual std::size_t frame_bytes() const = 0;

    // Scores one more pair of packed frames. Its frame number is the number
    // of pairs given before it.
    virtual void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) = 0;

    // As add, for frames whose planes are wherever a decoder left them. Only
    // the planes the metrics read are read (the luma, but VMAF v1's chroma
    // too), and they are copied before this returns, so they can then be
    // given back.
    virtual void add_planes(const FramePlanes& reference, const FramePlanes& distorted) = 0;

    // The scores, once every pair has been given.
    virtual PairScores finish() = 0;
};

}  // namespace vml
