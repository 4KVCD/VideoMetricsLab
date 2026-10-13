// VMAF v1 with the GPU (vmaf_v1_gpu.h), on a PC with a GPU Vulkan lists:
// every GPU's features and scores are libvmaf's CPU code's to the last bit,
// with CAMBI and SpEED on the GPU or on libvmaf's CPU extractors, and the
// probe accepts this PC's GPU.
#include "vml/vmaf_v1_gpu.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <system_error>

#include "vml/model_select.h"

namespace vml {
namespace {

// The GPUs the engine can score on; none where the engine is not bundled
// for this system or Vulkan lists none it can use (GitHub's runners). The
// tests are skipped without one.
std::vector<VulkanDevice> usable_gpus() {
    std::error_code error;
    if (!std::filesystem::is_regular_file(vulkan_engine_path(), error)) {
        return {};
    }
    try {
        std::vector<VulkanDevice> found = vulkan_devices();
        std::erase_if(found, [](const VulkanDevice& device) {
            return !device.usable || (device.kind != 1 && device.kind != 2);
        });
        return found;
    } catch (const VmafVulkanError&) {
        return {};
    }
}

std::filesystem::path bundled_model(const char* file) { return resource_folder() / "models" / file; }

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

TEST(VmafV1Gpu, EveryGpuGivesLibvmafsCpuFeaturesAndScoresBitForBit) {
    const std::vector<VulkanDevice> devices = usable_gpus();
    if (devices.empty()) {
        GTEST_SKIP() << "no GPU that Vulkan can calculate VMAF on";
    }
    const struct {
        const char* model;
        int bits;
        int step;
    } cases[] = {{"vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", 8, 1},
                 {"vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", 10, 2},
                 {"vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json", 10, 1},
                 {"vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json", 8, 3}};
    for (const auto& checked : cases) {
        SCOPED_TRACE(std::string(checked.model) + ", " + std::to_string(checked.bits) + "-bit, every " +
                     std::to_string(checked.step));
        const std::filesystem::path model = bundled_model(checked.model);
        const VulkanProbeFrames frames = v1_probe_frames(checked.bits, 7);
        const V1Reference expected = v1_cpu_reference(model, v1_probe_width, v1_probe_height, checked.bits,
                                                      frames.reference, frames.distorted, checked.step);
        std::vector<unsigned> wanted;
        for (unsigned frame = 0; frame < 7; frame += static_cast<unsigned>(checked.step)) {
            wanted.push_back(frame);
        }
        ASSERT_EQ(expected.features.frames, wanted);
        ASSERT_TRUE(std::all_of(expected.scores.begin(), expected.scores.end(),
                                [](double score) { return score > 0 && score < 100; }));
        ASSERT_GT(std::set<double>(expected.scores.begin(), expected.scores.end()).size(), 1u);
        for (const VulkanDevice& device : devices) {
            // CAMBI and SpEED on the GPU, then on libvmaf's CPU extractors.
            for (const bool on_gpu : {true, false}) {
                SCOPED_TRACE(device.name + (on_gpu ? ", CAMBI and SpEED on the GPU" : ", CAMBI and SpEED on the CPU"));
                V1Scorer scorer(v1_probe_width, v1_probe_height, checked.bits, model, checked.step, device.index, 2,
                                "vmaf_v1", nullptr, on_gpu, on_gpu);
                for (std::size_t i = 0; i < frames.reference.size(); ++i) {
                    scorer.add(frames.reference[i], frames.distorted[i]);
                }
                const V1Features got = scorer.features();
                EXPECT_EQ(got.frames, expected.features.frames);
                for (const auto& [feature, values] : expected.features.values) {
                    ASSERT_TRUE(got.values.contains(feature)) << feature;
                    EXPECT_TRUE(same_bits(got.values.at(feature), values)) << feature;
                }
                EXPECT_EQ(predict_v1(model, got), expected.scores);
            }
        }
    }
}

TEST(VmafV1Gpu, TheProbeAcceptsThisPcsGpu) {
    if (usable_gpus().empty()) {
        GTEST_SKIP() << "no GPU that Vulkan can calculate VMAF on";
    }
    const V1Probe probe = probe_v1();
    EXPECT_TRUE(probe.available) << probe.text;
    EXPECT_TRUE(probe.text.starts_with("Vulkan on ")) << probe.text;
}

}  // namespace
}  // namespace vml
