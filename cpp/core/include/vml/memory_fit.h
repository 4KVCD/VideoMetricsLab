// A pass whose metrics do not fit in the GPU's memory or the PC's at once is
// stopped, and the run scores them in two passes of half of them each, one
// after the other, halving again a half that still does not fit
// (ffmpeg_run.cpp's score_fitting). Each metric's scores are the same as in
// one pass: the same frames, scored by the same code.
//
// Windows rarely refuses memory when it runs out: it pages the GPU's out to
// the PC's memory, and the PC's to the disk, and the pass goes on 3 to 5
// times slower (4K, nine GPU metrics, an RTX 5090 left 512 MB to 2 GB:
// 21-35 s against 7.1 s). So a pass is watched for that state as it starts
// (MemoryWatch), not only for an allocation that fails.
#pragma once

#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vml {

// The metrics a split keeps in one pass, as `keys` (requested_metrics'
// order) list them: a YUV metric with its planes, VMAF YUV's U and V with
// VMAF (its Y), every other metric alone.
std::vector<std::vector<std::string>> memory_groups(const std::vector<std::string>& keys);

// Whether a group (memory_groups') keeps memory on the GPU: VMAF, NEG, VMAF
// v1, HDR VMAF, MS-SSIM, MS-SSIM YUV, CAMBI and CIEDE2000. PSNR, SSIM, XPSNR
// and PSNR-HVS-M are scored on the CPU only.
bool group_uses_gpu_memory(const std::vector<std::string>& group);

// How many of `keys`' groups a split for the GPU's memory (`gpu`) or the
// PC's can put in different passes: those that keep GPU memory, or all.
std::size_t memory_split_groups(const std::vector<std::string>& keys, bool gpu);

// `keys` in two halves of their groups, in order, the first with the extra
// group of an odd count. For the GPU's memory, the groups that keep none
// stay in the first half. Nothing when fewer than two groups can be split.
std::optional<std::pair<std::vector<std::string>, std::vector<std::string>>> halve_metrics(
    const std::vector<std::string>& keys, bool gpu);

// The GPU's memory as its driver reports it (vulkan_memory), in bytes.
struct GpuMemory {
    std::uint64_t used = 0;     // by this process
    std::uint64_t budget = 0;   // what this process may use
    std::uint64_t spilled = 0;  // meant for the GPU's own memory, in the PC's instead
};

// Why a pass does not fit, once MemoryWatch has found it.
struct MemoryShortage {
    bool gpu = false;  // the GPU's memory, else the PC's
    std::string text;
};

// Whether `error` is memory refused to a pass, as Linux and macOS refuse it
// where Windows pages it out, and Windows at its limit: the GPU's (true; the
// Vulkan engine's "out of GPU memory") or the PC's (false; std::bad_alloc,
// libvmaf's ENOMEM). Read from its text too, as the scorers' threads and
// processes hand errors on as text. Nothing for any other error.
std::optional<bool> memory_refused(const std::exception& error);

// A pass stopped because its metrics did not fit (MemoryShortage): the run
// scores them in halves (score_fitting). Ends the pass whatever route
// scores it (ends_the_run), so that no part tries another route for it.
class MemoryShort : public std::runtime_error {
public:
    explicit MemoryShort(MemoryShortage shortage)
        : std::runtime_error(shortage.text), gpu_(shortage.gpu) {}
    // The GPU's memory, else the PC's.
    bool gpu() const { return gpu_; }

private:
    bool gpu_;
};

// Watches a pass of `keys` as it starts: at its first progress reports once
// frames are scored, whether the GPU's memory (`gpu_memory`, where the pass
// uses a GPU that reports it) and the PC's (`headroom`: bytes left) still
// hold it. Only a pass that halve_metrics could split is watched. The
// readers are the pass's GPU's and the system's; tests give their own.
class MemoryWatch {
public:
    using GpuReader = std::function<std::optional<GpuMemory>()>;
    using HeadroomReader = std::function<std::optional<unsigned long long>()>;

    MemoryWatch(const std::vector<std::string>& keys, GpuReader gpu_memory, HeadroomReader headroom);

    // Whether the pass is watched at all.
    bool active() const { return watch_gpu_ || watch_ram_; }
    // Checks at a progress report (`done` frames, as the pass counts them).
    // True the first time the pass is found not to fit; shortage() says why.
    bool check(long long done);
    std::optional<MemoryShortage> shortage() const;

    // How many reports after the first scored frames are checked: the pass's
    // buffers are all made by then.
    static constexpr int checks = 3;
    // The PC's memory left under which a pass no longer fits.
    static constexpr unsigned long long headroom_floor = 1ull << 30;

private:
    std::optional<MemoryShortage> look();

    bool watch_gpu_;
    bool watch_ram_;
    GpuReader gpu_memory_;
    HeadroomReader headroom_;
    mutable std::mutex mutex_;
    std::optional<long long> first_done_;
    int checked_ = 0;
    std::optional<MemoryShortage> shortage_;
};

// The bytes a test variable caps the watch's reading to, from its value in
// MB; nothing when it is not set. VML_GPU_MEMORY_BUDGET_MB caps the GPU's
// budget and VML_MEMORY_HEADROOM_MB the PC's memory left, to drive the split
// on any PC.
std::optional<std::uint64_t> memory_test_cap(const char* variable);

}  // namespace vml
