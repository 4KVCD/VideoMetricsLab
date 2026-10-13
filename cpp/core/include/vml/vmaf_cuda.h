// VMAF and VMAF NEG with libvmaf's own CUDA code, on an NVIDIA GPU: what the
// GPU backend setting "CUDA" asks for, and what scores VMAF on an NVIDIA GPU
// where Vulkan cannot (gpu_vmaf.h chooses). The CUDA feature extractors
// (VIF, ADM and motion) are the bundled libvmaf's, libvmaf-fast, a fork of
// libvmaf with the fixes to its CUDA code merged; the Vulkan engine
// (vmaf_vulkan.h) is their port and gives the same scores. A port of the
// CUDA parts of vmaf_app/core/vmaf_gpu.py.
//
// libvmaf works in the primary CUDA context of CUDA's first GPU, the
// fastest, on which NVIDIA's decoder (gpu_frames.h) runs too, so that the
// decoder's pictures are copied into libvmaf's on the GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vml/libvmaf.h"
#include "vml/pair_scorer.h"

namespace vml {

// What a score calculated this way records as having calculated it (its
// provenance): "libvmaf-fast v1 (CUDA)".
std::string cuda_library_build();

class CudaUpload;  // vmaf_cuda.cpp: the copies of frames from system memory to the GPU

// One libvmaf context on the GPU, scoring frame pairs given in order (the
// Python app's vmaf_gpu.GpuScorer). add takes 4:2:0 frames in system memory,
// packed as FFmpeg's rawvideo writes them, at `bit_depth` bits (16-bit
// little-endian samples above 8 bits); only their luma is uploaded, because
// VMAF reads nothing else. add_decoded takes the pictures of the app's own
// decoders.
class CudaScorer : public PairScorer {
public:
    // `models` are libvmaf models to score, each a name and the version of a
    // model libvmaf has built in ({"vmaf", "vmaf_v0.6.1"}). Every
    // `n_subsample`th frame is scored. With `on_device`, every picture is
    // copied on the GPU (add_decoded's `copy`), and nothing is allocated for
    // uploads. Throws LibvmafError, also when the bundled libvmaf has no
    // CUDA code or CUDA cannot start.
    CudaScorer(int width, int height, int bit_depth, const std::vector<std::pair<std::string, std::string>>& models,
               int n_subsample = 1, bool on_device = false);
    ~CudaScorer() override;
    CudaScorer(const CudaScorer&) = delete;
    CudaScorer& operator=(const CudaScorer&) = delete;

    std::size_t frame_bytes() const override { return frame_bytes_; }

    void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) override;

    // One picture of a pair from a decoder in this process, given one of two
    // ways: copied on the GPU (`copy` is called with the GPU address of the
    // libvmaf picture's luma plane and the bytes from one of its rows to
    // the next), or written into the system memory it is uploaded from
    // (`write` is called with that memory's address, to write the luma
    // plane to with its rows packed). Each returns once the plane is there.
    struct Decoded {
        std::function<void(std::uint64_t, std::int64_t)> copy;
        std::function<void(std::uint8_t*)> write;
    };

    // Scores one more pair from the decoders: each picture is copied on the
    // GPU when its `copy` is set, and written and uploaded otherwise (which
    // a scorer made `on_device` cannot do).
    void add_decoded(const Decoded& reference, const Decoded& distorted);

    // The numbers of the frames scored (every n_subsample-th), and each
    // model's scores for them, kept to 6 decimals as libvmaf's JSON log
    // keeps them, in the order the models were given. Both are empty when
    // no pair was given (the caller treats that as a failure).
    PairScores finish() override;

private:
    // Takes two pictures from libvmaf's pool, has `fill` fill them (the
    // source's, then the test video's), and hands them to libvmaf.
    void add_pictures(const std::function<void(VmafPicture&, VmafPicture&)>& fill);
    void close();

    const Libvmaf& library_;
    unsigned step_ = 1;
    unsigned count_ = 0;
    VmafContext* context_ = nullptr;
    std::vector<std::pair<std::string, VmafModel*>> models_;
    std::unique_ptr<CudaUpload> upload_;
    std::size_t frame_bytes_ = 0;
};

// Whether libvmaf's CUDA code scores VMAF on this PC: libvmaf's version
// ("libvmaf 3.0.0"), or why not.
struct CudaProbe {
    bool available = false;
    std::string text;
};

// Starts CUDA and scores three small frame pairs, because the CUDA kernels
// load only then (a GPU older than the build supports fails here). Run it
// in a process of its own, where a driver that crashes ends only that
// process (the Python app's vmaf_gpu.probe).
CudaProbe probe_cuda();

}  // namespace vml
