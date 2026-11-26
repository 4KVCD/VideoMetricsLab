#include "vml/memory_fit.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace vml {
namespace {

using Keys = std::vector<std::string>;

// VMAF with its U and V, PSNR, PSNR YUV with a plane, and three GPU metrics.
const Keys run_keys{"vmaf", "vmaf_yuv_u", "vmaf_yuv_v", "psnr", "psnr_yuv", "psnr_yuv_y", "cambi", "ms_ssim", "ciede2000"};

TEST(MemoryFitTest, HalvesKeepGroupsTogetherAndCpuMetricsFirstForTheGpu) {
    // For the GPU's memory: its four groups halved; PSNR and PSNR YUV keep none, so they stay in the first pass.
    const auto gpu = halve_metrics(run_keys, true);
    ASSERT_TRUE(gpu);
    EXPECT_EQ(gpu->first, (Keys{"vmaf", "vmaf_yuv_u", "vmaf_yuv_v", "psnr", "psnr_yuv", "psnr_yuv_y", "cambi"}));
    EXPECT_EQ(gpu->second, (Keys{"ms_ssim", "ciede2000"}));
    // For the PC's: all six groups halved, the planes with their metric.
    const auto ram = halve_metrics(run_keys, false);
    ASSERT_TRUE(ram);
    EXPECT_EQ(ram->first, (Keys{"vmaf", "vmaf_yuv_u", "vmaf_yuv_v", "psnr", "psnr_yuv", "psnr_yuv_y"}));
    EXPECT_EQ(ram->second, (Keys{"cambi", "ms_ssim", "ciede2000"}));
    // One group, or one that keeps GPU memory, is not split.
    EXPECT_FALSE(halve_metrics({"psnr_yuv", "psnr_yuv_y"}, false));
    EXPECT_FALSE(halve_metrics({"vmaf", "psnr", "ssim"}, true));
}

TEST(MemoryWatchTest, ChecksTheFirstReportsAfterFramesAreScored) {
    std::optional<GpuMemory> gpu = GpuMemory{1000, 2000, 0};
    std::optional<unsigned long long> headroom = 8ull << 30;
    MemoryWatch watch(run_keys, [&gpu] { return gpu; }, [&headroom] { return headroom; });
    ASSERT_TRUE(watch.active());
    gpu->used = 3000;                // more than the budget...
    EXPECT_FALSE(watch.check(0));    // ...but the pass's first report is not checked
    EXPECT_FALSE(watch.check(0));    // nor one before a frame more is scored
    gpu->used = 1500;
    EXPECT_FALSE(watch.check(5));    // fits: checks 1 and 2
    EXPECT_FALSE(watch.check(10));
    gpu->spilled = 1;                // spilled into the PC's memory
    EXPECT_TRUE(watch.check(15));
    ASSERT_TRUE(watch.shortage());
    EXPECT_TRUE(watch.shortage()->gpu);
    EXPECT_FALSE(watch.check(20));  // found once
}

TEST(MemoryWatchTest, StopsCheckingAfterItsChecksAndWatchesThePcsMemory) {
    std::optional<unsigned long long> headroom = 8ull << 30;
    MemoryWatch watch(run_keys, nullptr, [&headroom] { return headroom; });
    ASSERT_TRUE(watch.active());
    EXPECT_FALSE(watch.check(1));
    headroom = MemoryWatch::headroom_floor - 1;
    EXPECT_TRUE(watch.check(2));
    EXPECT_FALSE(watch.shortage()->gpu);

    MemoryWatch late(run_keys, nullptr, [&headroom] { return headroom; });
    headroom = 8ull << 30;
    for (long long done = 1; done <= MemoryWatch::checks + 1; ++done) {
        EXPECT_FALSE(late.check(done));
    }
    headroom = 0;
    EXPECT_FALSE(late.check(100));  // its buffers were all made by its last check

    // A pass of one group is not watched: it could not be split.
    EXPECT_FALSE(MemoryWatch({"psnr_yuv", "psnr_yuv_y"}, nullptr, [&headroom] { return headroom; }).active());
}

}  // namespace
}  // namespace vml
