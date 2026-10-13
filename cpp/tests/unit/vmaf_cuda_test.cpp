// libvmaf's CUDA code, on a PC where it runs (an NVIDIA GPU): its scores are
// the Python app's (vulkan.json: the Vulkan engine's, which the Python app's
// tests hold to CUDA's bit for bit), and the probe accepts this PC's GPU.
#include "vml/vmaf_cuda.h"

#include <gtest/gtest.h>

#include <string>

#include "reference.h"
#include "vml/vmaf_vulkan.h"

namespace vml {
namespace {

using Json = nlohmann::json;

// Whether libvmaf's CUDA code runs here: GitHub's runners have no GPU, and a
// system without NVIDIA's driver no CUDA.
bool cuda_runs() {
    static const bool runs = probe_cuda().available;
    return runs;
}

// The Python app's scores for the probe's frames, at each size, depth and
// n_subsample, to the last digit.
TEST(VmafCuda, ScoresAsThePythonApp) {
    if (!cuda_runs()) {
        GTEST_SKIP() << "libvmaf's CUDA code does not run here";
    }
    const Json reference = test::load_reference("vulkan.json");
    for (const Json& expected : reference["cases"]) {
        const int width = expected["width"].get<int>(), height = expected["height"].get<int>();
        const int bits = expected["bits"].get<int>();
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + ", " + std::to_string(bits) + "-bit");
        const VulkanProbeFrames frames = vulkan_probe_frames(bits, expected["count"].get<int>(), width, height);
        std::vector<std::pair<std::string, std::string>> models;
        for (const auto& [name, version] : expected["models"].items()) {
            models.emplace_back(name, version.get<std::string>());
        }
        CudaScorer scorer(width, height, bits, models, expected["n_subsample"].get<int>());
        for (std::size_t i = 0; i < frames.reference.size(); ++i) {
            scorer.add(frames.reference[i], frames.distorted[i]);
        }
        const PairScores scores = scorer.finish();
        EXPECT_EQ(scores.frames, expected["frames"].get<std::vector<unsigned>>());
        ASSERT_EQ(scores.by_metric.size(), expected["scores"].size());
        for (const auto& [name, values] : scores.by_metric) {
            const std::vector<double> wanted = test::reference_numbers(expected["scores"].at(name));
            ASSERT_EQ(values.size(), wanted.size()) << name;
            for (std::size_t i = 0; i < values.size(); ++i) {
                EXPECT_TRUE(test::same_number(wanted[i], values[i]))
                    << name << ", frame " << scores.frames[i] << ": " << values[i] << " for " << wanted[i];
            }
        }
    }
}

// The probe accepts this PC's GPU, with libvmaf's version. A scorer given no
// frames gives each model no scores, and a frame shorter than its luma
// plane is an error, not a read past its end.
TEST(VmafCuda, ProbeAcceptsThisPcsGpuAndEdgesAreErrors) {
    if (!cuda_runs()) {
        GTEST_SKIP() << "libvmaf's CUDA code does not run here";
    }
    const CudaProbe probe = probe_cuda();
    EXPECT_TRUE(probe.available) << probe.text;
    EXPECT_EQ(probe.text, "libvmaf " + std::string(libvmaf().version()));

    CudaScorer scorer(64, 32, 8, {{"vmaf", "vmaf_v0.6.1"}, {"vmaf_neg", "vmaf_v0.6.1neg"}});
    const std::vector<std::uint8_t> frame(64 * 32 - 1);
    EXPECT_THROW(scorer.add(frame, frame), LibvmafError);
    const PairScores scores = scorer.finish();
    EXPECT_TRUE(scores.frames.empty());
    ASSERT_EQ(scores.by_metric.size(), 2u);
    EXPECT_EQ(scores.by_metric[0].first, "vmaf");
    EXPECT_EQ(scores.by_metric[1].first, "vmaf_neg");
    EXPECT_TRUE(scores.by_metric[0].second.empty());
    EXPECT_TRUE(scores.by_metric[1].second.empty());

    // Pictures copied on the GPU only: frames in system memory are refused.
    CudaScorer on_device(64, 32, 8, {{"vmaf", "vmaf_v0.6.1"}}, 1, true);
    const std::vector<std::uint8_t> whole(on_device.frame_bytes());
    EXPECT_THROW(on_device.add(whole, whole), LibvmafError);
    EXPECT_THROW(on_device.add_decoded({.copy = {}, .write = [](std::uint8_t*) {}},
                                       {.copy = {}, .write = [](std::uint8_t*) {}}),
                 LibvmafError);
}

}  // namespace
}  // namespace vml
