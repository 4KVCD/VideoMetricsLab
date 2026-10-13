// VMAF and VMAF NEG on any GPU with Vulkan. The features VMAF is predicted
// from (VIF, ADM and motion) are calculated by the bundled Vulkan engine
// (tools/vmaf_vulkan, built from libvmaf-fast's fast/vulkan folder, from
// the same release as the bundled libvmaf), and libvmaf predicts the scores
// from them (predict_scores). A port of vmaf_app/core/vmaf_vulkan.py.
//
// The engine's shaders use integer arithmetic only, so its features are
// those of libvmaf's CUDA code to the last bit, on any GPU whose driver
// compiles the shaders right; probe_vulkan() checks that a GPU's does.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vml/libvmaf.h"
#include "vml/platform/library.h"

namespace vml {

// VMAF could not be calculated with Vulkan; the CPU then calculates it.
class VmafVulkanError : public LibvmafError {
public:
    using LibvmafError::LibvmafError;
};

// A GPU as Vulkan lists it.
struct VulkanDevice {
    int index = 0;  // Vulkan's number for it, which the engine takes
    std::string name;
    std::uint32_t vendor = 0;  // its maker's PCI ID: 0x10DE NVIDIA, 0x8086 Intel, 0x1002 AMD
    std::uint32_t kind = 0;    // Vulkan's device type: 1 integrated, 2 discrete
    bool usable = false;       // whether it has what the shaders need (64-bit integers)
    bool has_double = false;   // whether it has 64-bit floating point
};

// The engine's state for one comparison; only the engine knows its layout.
struct VulkanContext;

class GpuFrameStream;  // gpu_frames.h

// The bundled engine and the functions of it the app calls. The engine has
// no header: these are its functions as fast/vulkan/vmaf_vulkan.cpp
// defines them.
struct VulkanEngine {
    explicit VulkanEngine(const std::filesystem::path& path);

    platform::DynamicLibrary library;
    int (*device)(int index, char* name, int name_bytes, std::uint32_t* vendor, std::uint32_t* type,
                  std::uint32_t* flags);
    int (*create)(VulkanContext** out, int device, int width, int height, int bit_depth, int flags);
    void (*destroy)(VulkanContext* context);
    const char* (*error)();
    int (*submit)(VulkanContext* context, const std::uint8_t* reference, std::ptrdiff_t reference_stride,
                  const std::uint8_t* distorted, std::ptrdiff_t distorted_stride, int score);
    int (*flush)(VulkanContext* context);
    int (*features)(VulkanContext* context, unsigned index, double* out);
    int (*sums)(VulkanContext* context, unsigned index, std::uint64_t* out, int count);
    // The memory to write the next pair's luma planes to, rows `stride`
    // bytes apart, and the call that scores them once they are written:
    // submit without its copy.
    int (*staging)(VulkanContext* context, std::uint8_t** reference, std::uint8_t** distorted, std::uint32_t* stride);
    int (*commit)(VulkanContext* context, int score);
    // For a context whose frames come from GPU memory: each slot's buffer,
    // exported for a decoder on the same GPU to import, and where the next
    // pair's planes go in it.
    int (*export_memory)(VulkanContext* context, int slot, void** handle, std::uint64_t* bytes);
    int (*shared_next)(VulkanContext* context, int* slot, std::uint32_t* stride, std::uint32_t* distorted_offset);
};

// Where the bundled engine is.
std::filesystem::path vulkan_engine_path();

// The bundled engine, loaded the first time it is asked for. Throws
// VmafVulkanError when it cannot be loaded.
const VulkanEngine& vulkan_engine();

// What a score calculated this way records as having calculated it (its
// provenance): "libvmaf-fast v1 (Vulkan)".
std::string vulkan_library_build();

// The GPUs Vulkan lists, in its order. Throws VmafVulkanError.
std::vector<VulkanDevice> vulkan_devices();

// The GPU to score on: a discrete one before an integrated one, each the
// first Vulkan lists; nothing when no GPU can.
std::optional<VulkanDevice> best_vulkan_device(const std::vector<VulkanDevice>& found);

// How many features the engine gives for each frame.
inline constexpr std::size_t vulkan_feature_count = 12;

// The features of the frames scored: their numbers, and each feature's
// value for every one of them, a column per feature in the engine's order
// (VIF's 4 scales, ADM2, motion2, NEG's VIF and ADM2, motion).
struct VulkanFeatures {
    std::vector<unsigned> frames;
    std::array<std::vector<double>, vulkan_feature_count> columns;
};

// Calculates VMAF's features with Vulkan for frame pairs given in order:
// 4:2:0 frames with luma and chroma planes, packed as FFmpeg's rawvideo
// writes them, at `bit_depth` bits (16-bit little-endian samples above 8
// bits). Only the luma is used, as in VMAF. Frames a decoder in this
// process writes itself go straight into the engine's memory instead
// (add_decoded), or, from NVIDIA's decoder, stay on the GPU (add_shared).
class VulkanScorer {
public:
    // `models` are libvmaf models to score, each a name and the version of
    // a model libvmaf has built in ({"vmaf", "vmaf_v0.6.1"}). Every
    // `n_subsample`th frame is scored. `device` is Vulkan's number for the
    // GPU; nothing picks best_vulkan_device(). With `shared`, the frames
    // come from GPU memory that decoder imports and writes (add_shared).
    // Throws VmafVulkanError, also when the decoder and Vulkan cannot share
    // memory (another GPU, an old driver).
    VulkanScorer(int width, int height, int bit_depth, std::vector<std::pair<std::string, std::string>> models,
                 int n_subsample = 1, std::optional<int> device = std::nullopt, GpuFrameStream* shared = nullptr);
    ~VulkanScorer();
    VulkanScorer(const VulkanScorer&) = delete;
    VulkanScorer& operator=(const VulkanScorer&) = delete;

    // The bytes of one packed frame.
    std::size_t frame_bytes() const { return frame_bytes_; }

    // Scores one more pair. Its frame number is the number of pairs given
    // before it.
    void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted);

    // Scores one more pair that a decoder writes in itself: `reference` and
    // `distorted` are each called with the address to write a luma plane to,
    // its rows packed, and return once it is written. A frame that is not
    // scored (n_subsample) needs only the source's plane, for motion.
    void add_decoded(const std::function<void(std::uint8_t*)>& reference,
                     const std::function<void(std::uint8_t*)>& distorted);

    // Scores one more pair that the decoder given as `shared` copies on the
    // GPU: `reference` and `distorted` are each called with the GPU address
    // to copy a luma plane to and the bytes from one row to the next, and
    // return once the plane is there.
    void add_shared(const std::function<void(std::uint64_t, std::int64_t)>& reference,
                    const std::function<void(std::uint64_t, std::int64_t)>& distorted);

    // The frames scored and their features. Call it once every frame has
    // been given.
    VulkanFeatures features();

    // A frame's raw sums (at most 128), for the probe and tests. The engine
    // keeps them for the first 256 frames only; a later frame's throws
    // VmafVulkanError.
    std::vector<std::uint64_t> sums(unsigned frame) const;

    // The numbers of the frames scored, and each model's scores for them,
    // in the order the models were given.
    struct Scores {
        std::vector<unsigned> frames;
        std::vector<std::pair<std::string, std::vector<double>>> by_model;
    };
    Scores finish();

private:
    void check(int error, const std::string& what) const;
    // Gives the decoder's imports back and the engine's context.
    void release();

    const VulkanEngine& engine_;
    std::vector<std::pair<std::string, std::string>> models_;
    unsigned step_ = 1;
    unsigned count_ = 0;
    VulkanContext* context_ = nullptr;
    std::ptrdiff_t luma_stride_ = 0;
    std::size_t luma_bytes_ = 0;
    std::size_t frame_bytes_ = 0;
    // With frames from GPU memory: the decoder, and each slot's memory as it
    // imported it (its address there, and what it takes to give it back).
    GpuFrameStream* shared_ = nullptr;
    std::vector<std::pair<std::uint64_t, void*>> imports_;
    // add_decoded's plane, written packed for a width whose rows the engine
    // pads.
    std::vector<std::uint8_t> packed_;
};

// The probe's frame pairs (reference frames, distorted frames): flat,
// smooth and textured areas, blurred, noisy and sharpened in turn, so that
// every branch of the shaders is taken. Made with integer arithmetic only,
// they are the same bytes on every PC, and the Python app's own.
struct VulkanProbeFrames {
    std::vector<std::vector<std::uint8_t>> reference;
    std::vector<std::vector<std::uint8_t>> distorted;
};
VulkanProbeFrames vulkan_probe_frames(int bits, int count = 3, int width = 320, int height = 192);

// The SHA-256 of the sums a GPU gives for the probe's frames.
std::string vulkan_probe_sums(int device, int bits);

// Whether Vulkan scores VMAF on this PC, on which GPU, and what the GPU is
// (or why not).
struct VulkanProbe {
    bool available = false;
    std::optional<int> device;
    std::string text;
};

// Scores the probe's frames at 8 and 10 bits on the GPU to score on (the
// one VML_VMAF_VULKAN_DEVICE names, or the best), and accepts it only if
// its sums are exactly the known ones: a driver that compiles a shader
// wrongly gives wrong scores, not an error (Intel's and AMD's did).
VulkanProbe probe_vulkan();

}  // namespace vml
