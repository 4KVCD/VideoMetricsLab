// VMAF and VMAF NEG on the GPU: which runs score them there (VMAF v1 with
// them, vmaf_v1_gpu.h), the check of the GPU, and the frames' way to the
// Vulkan scorer, from one FFmpeg per video or from the app's own decoders,
// paired frame N with frame N. A port of the parts of
// vmaf_app/core/vmaf_gpu.py and vmaf_vulkan.py the C++ app has so far. Not
// yet: libvmaf's CUDA code, frames paired by timestamp (VML_FRAME_PAIRING)
// and FFmpeg pairing them (VML_VMAF_PIPES=canvas).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vml/cancel.h"
#include "vml/video.h"
#include "vml/vmaf_vulkan.h"

namespace vml {

// The models a run scores on the GPU: each metric's key ("vmaf",
// "vmaf_neg") with the libvmaf model it is scored with ("vmaf_v0.6.1"), and
// VMAF v1's ("vmaf_v1") with its model's "path=<file>".
using GpuModels = std::vector<std::pair<std::string, std::string>>;

// The models of VMAF, NEG and VMAF v1 a run scores on the GPU, or none.
// VMAF with a custom model file is calculated on the CPU, and so is NEG
// beside it. `model` is the run's model= value ("version=vmaf_v0.6.1");
// `model_v1`, VMAF v1's "path=<file>".
GpuModels gpu_models(bool compute_vmaf, bool compute_vmaf_neg, const std::string& model,
                     bool compute_vmaf_v1 = false, const std::string& model_v1 = "");

// The GPU backend setting ("auto", "vulkan" or "cuda"): which backend is
// tried first. The GPU is checked again when it changes that.
void set_gpu_backend(const std::string& preference);

// What scores VMAF on this PC's GPU, as the Python app's probe finds it.
struct GpuVmafBackend {
    // "vulkan" when Vulkan scores it here; "cuda" when the Python app would
    // try libvmaf's CUDA code (an NVIDIA GPU, with CUDA first or Vulkan
    // failing), which this app does not have yet; "" when the CPU scores it.
    std::string backend;
    std::optional<int> device;  // Vulkan's number for the GPU
    std::string text;           // what scores it ("Vulkan on ..."), or why the GPU does not
};

// The GPU's answer, found once (in this process: the Python app checks in a
// process of its own) and kept.
GpuVmafBackend gpu_vmaf_backend();

// The models a run scores on the GPU (gpu_models), or none when its VMAF is
// calculated on the CPU: VMAF set to the CPU (`enabled`), a custom model, a
// comparison deeper than 10 bits, an odd width or height (`size`, where
// known), or no GPU that scores VMAF here (gpu_vmaf_backend). VMAF v1 is
// included where the GPU calculates it (vmaf_v1_gpu.h's v1_on_gpu). What
// the GPU does not score is left out, and FFmpeg's libvmaf calculates it.
GpuModels scores_on_gpu(bool compute_vmaf, bool compute_vmaf_neg, const std::string& model, bool enabled,
                        int bit_depth, std::optional<std::pair<int, int>> size, bool compute_vmaf_v1 = false,
                        const std::string& model_v1 = "");

// What a run scores outside FFmpeg, and how (the Python app's _GpuPlan):
// VMAF, NEG and VMAF v1 on the GPU, or PSNR, SSIM and XPSNR with libvmaf's
// CPU extractors.
struct GpuPlan {
    GpuModels models;  // with backend "cpu": each metric's key with itself
    int width = 0;     // the size and depth frames are compared at
    int height = 0;
    int bit_depth = 8;
    // Vulkan's number for the GPU; -1 for the best one, for VMAF v1 alone
    // where VMAF v0.6.1 would not be scored on the GPU.
    int device = 0;
    // "vulkan", or "cpu" for PSNR, SSIM and XPSNR with libvmaf's CPU
    // extractors (libvmaf_cpu.h), which take their frames the same way.
    std::string backend = "vulkan";
    int threads = 0;     // the CPU extractors' libvmaf threads
    int frame_rate = 0;  // XPSNR's, in whole frames per second
};

// The GPU's turn: one GPU pass at a time in this process (the Python app's
// gpu.GPU_PASS), so that no video's GPU metrics run beside another's.
std::timed_mutex& gpu_pass();

// FFmpeg's output options for VMAF on the GPU, the test video's FFmpeg's and
// the source's: the video's frames ([vmaf_dist] or [vmaf_ref]) as they come,
// cut at `duration_limit` seconds (0: none), as YUV4MPEG into the pipe at
// `test_pipe` or `source_pipe` (GpuAttempt's).
std::pair<std::vector<std::string>, std::vector<std::string>> gpu_output_args(double duration_limit,
                                                                              const std::string& test_pipe,
                                                                              const std::string& source_pipe);

// The status a run waiting for the GPU's turn sends.
inline constexpr std::string_view gpu_wait_message = "Waiting for the GPU: another video's GPU pass is running…";

// One video as FFmpeg's filter chain gives it: its frames as a YUV4MPEG
// stream (FFmpeg's yuv4mpegpipe output) on a pipe this process serves. A
// thread reads the frames ahead into a few buffers; pull() gives them in
// order, and each is given back with release().
class StreamReader {
public:
    // A reader of frames of `frame_bytes` each, its pipe's path ending in
    // `name`. Throws platform::PipeError.
    StreamReader(std::string_view name, std::size_t frame_bytes);
    ~StreamReader();
    StreamReader(const StreamReader&) = delete;
    StreamReader& operator=(const StreamReader&) = delete;

    // The path FFmpeg writes to.
    const std::string& path() const;

    // Starts the reading thread.
    void start();

    // The next frame, or nullptr at the video's end. Throws LibvmafError
    // when the stream is not what FFmpeg writes.
    std::vector<std::uint8_t>* pull();

    // Gives a frame's buffer back for the next frame.
    void release(std::vector<std::uint8_t>* frame);

    // Says the comparison is over: whatever FFmpeg still writes is read and
    // dropped.
    void drop_rest();

    // Says FFmpeg has gone, so the reader stops waiting for more. FFmpeg
    // opens no output that got no frame, so a pipe it never opened is
    // opened here to end the wait for it.
    void ffmpeg_ended();

    // Stops reading, even while waiting for FFmpeg.
    void stop();

    // Waits for the reading thread to end, and throws its error, if any.
    void join();

private:
    void read_frames();

    struct State;
    std::unique_ptr<State> state_;
};

// The app's side of one FFmpeg attempt: the pipes FFmpeg writes the two
// videos' frames to, and the thread that pairs them, frame N with frame N
// until either video ends, and has the plan's scorer score each pair: the
// Vulkan engine, or libvmaf's CPU extractors.
class GpuAttempt {
public:
    // Throws LibvmafError (VmafVulkanError) when the scorer cannot start,
    // and platform::PipeError.
    GpuAttempt(const GpuPlan& plan, int n_subsample);
    ~GpuAttempt();
    GpuAttempt(const GpuAttempt&) = delete;
    GpuAttempt& operator=(const GpuAttempt&) = delete;

    // FFmpeg's output options for the test video's FFmpeg and the source's:
    // the video's frames ([vmaf_dist] or [vmaf_ref]) as they come, cut at
    // `duration_limit` seconds (0: none), as YUV4MPEG into its pipe.
    std::pair<std::vector<std::string>, std::vector<std::string>> output_args(double duration_limit) const;

    // Call once FFmpeg has ended: the scores, or nothing when FFmpeg failed
    // (the caller then retries or gives up). Throws LibvmafError when the
    // scoring failed.
    std::optional<PairScores> finish(bool ffmpeg_succeeded);

private:
    void feed();
    void stop();

    struct State;
    std::unique_ptr<State> state_;
};

// VMAF and NEG with both videos decoded in this process (gpu_frames.h)
// instead of by FFmpeg: the frames GpuAttempt's FFmpegs would give, cropped,
// scaled and widened by the decoders, and paired frame N with frame N as
// far as `duration_limit` (-t's text, or nothing) keeps them. `decoders`
// are the source's and the test video's (decoders_here); `ffmpeg` copies
// their packets out of their containers. With NVIDIA's decoder for both,
// each luma plane stays on the GPU, copied into the memory the engine reads
// it from; otherwise the decoder writes it into the memory the engine
// copies to the GPU. Throws GpuDecodeUnavailableError before the first
// picture when the app cannot decode a video, GpuDecodeFailedError after
// it, VmafVulkanError, and Cancelled.
PairScores score_decoded(const VideoInfo& source, const VideoInfo& distorted,
                         const std::optional<CropBox>& source_crop,
                         const std::optional<CropBox>& distorted_crop, const GpuPlan& plan,
                         int n_subsample, const std::optional<std::string>& duration_limit,
                         long long total_frames, const std::string& scale_algorithm,
                         const std::pair<std::string, std::string>& decoders, const std::string& ffmpeg,
                         const std::string& ffprobe,
                         const std::function<void(long long, long long, double)>& on_progress,
                         const CancelToken& cancel);

}  // namespace vml
