// VMAF v1 with the GPU: the score FFmpeg's libvmaf gives for a VMAF v1 model
// (vmaf_v1.0.16 and its variants), several times faster. A port of
// vmaf_app/core/vmaf_v1_gpu.py.
//
// VMAF v1 is predicted from four features:
// - ADM3 and motion3, about two thirds of the CPU's work, are calculated on
//   the GPU by the Vulkan engine's VMAF v1 mode, which follows libvmaf's CPU
//   code (integer_adm.c, integer_motion.c) in integer arithmetic and gives
//   the same values bit for bit on any GPU.
// - CAMBI and SpEED chroma are calculated on the GPU too when the engine
//   supports the model's options for them, again bit for bit as libvmaf's
//   CPU code; otherwise by the bundled libvmaf's own CPU extractors, on
//   threads beside the GPU: the same code on the same frames.
// libvmaf then predicts the score from the four features, so the score is
// libvmaf's CPU score.
//
// The frames decoded in the app stay on the GPU with NVIDIA's decoder, and
// with AMD's, which hands them over to Vulkan (gpu_frames.h); Windows' own
// decoder's pictures are even read where it left them, as textures.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/gpu_vmaf.h"
#include "vml/libvmaf.h"
#include "vml/pair_scorer.h"
#include "vml/vmaf_cuda.h"
#include "vml/vmaf_vulkan.h"
#include "vml/video.h"

namespace vml {

// VMAF v1 could not be calculated with the GPU; FFmpeg's libvmaf then
// calculates it on the CPU.
class VmafV1Error : public LibvmafError {
public:
    using LibvmafError::LibvmafError;
};

// libvmaf's names for VMAF v1's four features, as its models list them.
inline constexpr const char* v1_adm3 = "VMAF_integer_feature_adm3_score";
inline constexpr const char* v1_motion3 = "VMAF_integer_feature_motion3_score";
inline constexpr const char* v1_cambi = "Cambi_feature_cambi_score";
inline constexpr const char* v1_speed = "Speed_chroma_feature_speed_chroma_uv_score";

// What a score calculated this way records as having calculated it (its
// provenance): "libvmaf-fast v1 (Vulkan: ADM3, motion3, CAMBI, SpEED)".
std::string v1_library_build();

// The frames scored, and each of the four features' values for them (by
// the names above).
struct V1Features {
    std::vector<unsigned> frames;
    std::map<std::string, std::vector<double>> values;
};

// Calculates VMAF v1 of frame pairs given in order: 4:2:0 frames packed as
// FFmpeg's rawvideo writes them, at `bit_depth` bits (16-bit little-endian
// samples above 8 bits), with their chroma, which SpEED reads.
class V1Scorer : public PairScorer {
public:
    // `model` is the model's file. Every `n_subsample`th frame is scored.
    // `device` is Vulkan's number for the GPU (nothing: the best);
    // `threads`, libvmaf's for CAMBI and SpEED on the CPU (nothing: as many
    // as the cores allow). With `shared`, the frames come from that decoder
    // on the GPU (add_decoded): the lumas, and the chroma SpEED reads, are
    // copied on the GPU into memory the decoder and Vulkan share, and the
    // planes libvmaf's extractors read are downloaded straight into their
    // pictures. `gpu_cambi` and `gpu_speed` false calculate those two with
    // libvmaf's CPU code even where the engine could. `pictures` means the
    // decoders give their pictures as textures (GpuFrameStream::
    // gives_textures): the engine then reads them where they are, without
    // copying them, if it can, and pictures() says whether it does; each
    // pair's slots are then given back through release_later. Throws
    // VmafV1Error (or VmafVulkanError, LibvmafError).
    V1Scorer(int width, int height, int bit_depth, const std::filesystem::path& model, int n_subsample = 1,
             std::optional<int> device = std::nullopt, std::optional<int> threads = std::nullopt,
             std::string name = "vmaf_v1", GpuFrameStream* shared = nullptr, bool gpu_cambi = true,
             bool gpu_speed = true, bool pictures = false);
    ~V1Scorer() override;
    V1Scorer(const V1Scorer&) = delete;
    V1Scorer& operator=(const V1Scorer&) = delete;

    std::size_t frame_bytes() const override { return frame_bytes_; }

    void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) override;

    // Scores one more pair held by the decoders, for a scorer made with
    // `shared`. The CPU copies none of it.
    void add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot);

    // Whether the engine reads the decoders' pictures where they are.
    bool pictures() const;

    // In pictures mode, gives `stream`'s `slot`, which the pairs given so far
    // no longer need, back once the engine has read it: after the last frame
    // given, or for a source picture (`reference`) after the next frame too,
    // because the next frame's motion feature reads it.
    void release_later(GpuFrameStream& stream, int slot, bool reference);

    // The features of the frames scored. Call it once every frame has been
    // given.
    V1Features features();

    // The frames scored, and the model's scores for them (as `name`),
    // predicted by libvmaf and rounded as its JSON log rounds them.
    PairScores finish() override;

private:
    struct State;
    std::unique_ptr<State> state_;
    std::size_t frame_bytes_ = 0;
};

// The model's scores for its four features' values, predicted by libvmaf
// itself. Throws VmafV1Error, LibvmafError.
std::vector<double> predict_v1(const std::filesystem::path& model, const V1Features& features);

// VMAF v1 of frames in memory with libvmaf's CPU code alone: the frames
// scored and the features' values, and the scores rounded as V1Scorer
// rounds them. The probe and the tests hold V1Scorer to it.
struct V1Reference {
    V1Features features;
    std::vector<double> scores;
};
V1Reference v1_cpu_reference(const std::filesystem::path& model, int width, int height, int bit_depth,
                             const std::vector<std::vector<std::uint8_t>>& reference,
                             const std::vector<std::vector<std::uint8_t>>& distorted, int n_subsample = 1,
                             int threads = 0);

// The probe's frame pairs (reference frames, distorted frames), at 960x720:
// textured and moving, the test frames blurred and noisy, made with integer
// arithmetic only, so they are the same bytes on every PC, and the Python
// app's own.
VulkanProbeFrames v1_probe_frames(int bits, int count = 4);
inline constexpr int v1_probe_width = 960, v1_probe_height = 720;

// Feeds the same frames to a run's GPU scorers: the Vulkan engine's or
// libvmaf's CUDA code for VMAF v0.6.1 and NEG, and V1Scorer for VMAF v1
// (the Python app's MultiScorer).
class MultiScorer : public PairScorer {
public:
    // `models` as gpu_models gives them, VMAF v1's entry ("vmaf_v1") with
    // its model's "path=<file>". VMAF v0.6.1 and NEG are scored by
    // `backend`: "vulkan" on Vulkan's GPU `device` (nothing: the best), or
    // "cuda" by libvmaf's CUDA code, with VMAF v1 on the best GPU. `shared`
    // and `pictures` as V1Scorer's.
    MultiScorer(int width, int height, int bit_depth, const GpuModels& models, int n_subsample,
                const std::string& backend, std::optional<int> device, GpuFrameStream* shared = nullptr,
                bool pictures = false);
    ~MultiScorer() override;

    std::size_t frame_bytes() const override;
    void add(std::span<const std::uint8_t> reference, std::span<const std::uint8_t> distorted) override;

    // Scores one more pair held by the decoders, without a copy through the
    // CPU (for scorers made with `shared`).
    void add_decoded(GpuFrameStream& reference, int reference_slot, GpuFrameStream& test, int test_slot);

    // Whether VMAF v1's scorer reads the decoders' pictures where they are;
    // the decoders' slots are then given back through release_later.
    bool pictures() const;

    // In pictures mode, V1Scorer::release_later.
    void release_later(GpuFrameStream& stream, int slot, bool reference);

    // The frames scored, and every scorer's scores for them, VMAF v0.6.1's
    // and NEG's before VMAF v1's.
    PairScores finish() override;

private:
    // VMAF v0.6.1's and NEG's, one or neither.
    std::unique_ptr<VulkanScorer> vulkan_;
    std::unique_ptr<CudaScorer> cuda_;
    std::unique_ptr<V1Scorer> v1_;
};

// score_decoded (gpu_vmaf.h) for a run with VMAF v1: both videos decoded in
// this process as whole frames, because SpEED reads the chroma; with
// NVIDIA's decoder for both, or AMD's with Vulkan, the planes stay on the
// GPU. Throws as score_decoded does, and VmafV1Error.
PairScores score_decoded_v1(const VideoInfo& source, const VideoInfo& distorted,
                            const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                            const GpuPlan& plan, int n_subsample, const std::optional<std::string>& duration_limit,
                            long long total_frames, const std::string& scale_algorithm,
                            const std::pair<std::string, std::string>& decoders, const std::string& ffmpeg,
                            const std::string& ffprobe,
                            const std::function<void(long long, long long, double)>& on_progress,
                            const CancelToken& cancel);

// Whether VMAF v1 is calculated with the GPU on this PC, and what the GPU
// is or why not.
struct V1Probe {
    bool available = false;
    std::string text;
};

// The check of the GPU, in this process: the Vulkan check of the shaders
// VMAF v0.6.1 shares with it, then the four features of a few frames at 8
// and at 10 bits, with the five-frame motion window, held to libvmaf's CPU
// values to the last bit. A driver that compiles a shader wrongly gives
// wrong scores, not an error.
V1Probe probe_v1();

// probe_v1's answer, found once, in a process of its own where a program to
// run it is set (isolated.h).
V1Probe v1_available();

// Whether a run's VMAF v1 is calculated with the GPU: `model` (its
// "path=<file>", or empty while Auto is not yet resolved) is a VMAF v1 model,
// `size`, where known, leaves SpEED blocks to score, and the GPU passed its
// check.
bool v1_on_gpu(const std::string& model, std::optional<std::pair<int, int>> size);

}  // namespace vml
