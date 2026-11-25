// VMAF of a frame's chroma planes, for YUV VMAF (MSU's YUV-VMAF 6:1:1): the
// run's VMAF model applied to the U plane and to the V plane as if each were
// a luma plane (its VIF, ADM and motion), so that YUV VMAF is
// (6 VMAF + VMAF U + VMAF V) / 8 (JobRun combines them with the run's VMAF).
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vml/libvmaf.h"

namespace vml {

class GpuFrameStream;  // gpu_frames.h
class VulkanScorer;    // vmaf_vulkan.h

// The keys of the two planes' scores.
inline constexpr const char* vmaf_u_key = "vmaf_yuv_u";
inline constexpr const char* vmaf_v_key = "vmaf_yuv_v";

// Decoders whose pictures stay on the GPU, which the planes' scorers may take
// them from: the test video's decoder, which imports the scorers' memory
// (the source's decoder copies into it too: GpuFrameStream::
// shares_gpu_memory_with), and the Vulkan device of the pass's VMAF.
struct ChromaFromDecoders {
    GpuFrameStream* test = nullptr;
    int device = 0;
};

// Whether the Vulkan engine can score the U and V planes of 4:2:0 pictures `width` x `height` at `bit_depth`
// with `model` (the run's model= value): one of its two built-in models, 10 bits at most, and even-sized chroma
// planes. Otherwise ChromaVmaf scores them with libvmaf on the CPU.
bool chroma_vmaf_fits_gpu(int width, int height, int bit_depth, const std::string& model);

// Scores the U and V planes (or one of them) of the libvmaf pictures given,
// 4:2:0 at `width` x `height`, whose chroma planes are half the size rounded
// down. On the GPU (Vulkan's number `gpu`, -1 for the one Vulkan picks) with
// the Vulkan engine VMAF runs on, where the model is one of the two it has
// built in and the chroma planes are even-sized (the engine takes no odd
// size); otherwise with libvmaf's own VMAF, one context for each plane, on
// `threads` threads in all. On the GPU, with `from_decoders` whose decoders
// copy their planes on the GPU (GpuFrameStream::copies_planes) and an even
// frame size, both planes are copied there from the decoders' memory
// (add_decoded) rather than given in system memory (add). Every pair is given
// (motion reads the frame before); every `n_subsample`th is scored. Throws
// LibvmafError (VmafVulkanError among them).
class ChromaVmaf {
public:
    // `model`: the run's model= value ("version=vmaf_v0.6.1", "path=<file>").
    // `keys`: vmaf_u_key and / or vmaf_v_key.
    ChromaVmaf(int width, int height, int bit_depth, const std::string& model, const std::vector<std::string>& keys,
               int n_subsample, int threads, std::optional<int> gpu,
               std::optional<ChromaFromDecoders> from_decoders = std::nullopt);
    ~ChromaVmaf();
    ChromaVmaf(const ChromaVmaf&) = delete;
    ChromaVmaf& operator=(const ChromaVmaf&) = delete;

    // The keys of the planes scored on the GPU.
    std::vector<std::string> keys_on_gpu() const;

    // Whether the pairs are given with add_decoded (all planes copied from
    // the decoders' memory), else with add.
    bool takes_decoded() const { return takes_decoded_; }

    // Scores the next pair. The planes are read (copied) before it returns.
    void add(const VmafPicture& reference, const VmafPicture& distorted);

    // Scores the next pair, in the decoders' slots, which the GPU copies into
    // each plane's scorer (where takes_decoded). Throws GpuDecodeFailedError
    // when a decoder fails to copy.
    void add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot);

    // Each plane's scores, by its key, for the frames scored (every
    // n_subsample-th), kept to 6 decimals as the run's VMAF is.
    std::map<std::string, std::vector<double>> finish();

private:
    struct Plane;
    std::vector<std::unique_ptr<Plane>> planes_;
    bool takes_decoded_ = false;
};

}  // namespace vml
