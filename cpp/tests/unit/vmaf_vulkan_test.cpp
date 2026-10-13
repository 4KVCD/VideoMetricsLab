// The Vulkan engine's binding held to the Python app's (vulkan.json, from
// vmaf_vulkan.py): the probe's frames on every system, and, on a PC with a
// GPU Vulkan lists, the features, the scores and the probe.
#include "vml/vmaf_vulkan.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <system_error>

#include "reference.h"
#include "vml/sha256.h"

namespace vml {
namespace {

using Json = nlohmann::json;

std::string sha256_of(const std::vector<std::uint8_t>& bytes) {
    Sha256 digest;
    digest.update(bytes.data(), bytes.size());
    return digest.hexdigest();
}

// The GPU the engine scores on, or nothing where the engine is not bundled
// for this system or Vulkan lists no GPU it can use (GitHub's runners have
// none). The GPU tests are skipped without one.
std::optional<VulkanDevice> gpu() {
    std::error_code error;
    if (!std::filesystem::is_regular_file(vulkan_engine_path(), error)) {
        return std::nullopt;
    }
    try {
        return best_vulkan_device(vulkan_devices());
    } catch (const VmafVulkanError&) {
        return std::nullopt;
    }
}

void set_device_variable(const std::string& value) {
#ifdef _WIN32
    _putenv_s("VML_VMAF_VULKAN_DEVICE", value.c_str());
#else
    setenv("VML_VMAF_VULKAN_DEVICE", value.c_str(), 1);
#endif
}

// The probe's frames are the Python app's, byte for byte, at each size and
// depth.
TEST(VmafVulkan, ProbeFramesAreThePythonApps) {
    const Json reference = test::load_reference("vulkan.json");
    ASSERT_FALSE(reference["cases"].empty());
    for (const Json& expected : reference["cases"]) {
        const int width = expected["width"].get<int>(), height = expected["height"].get<int>();
        const int bits = expected["bits"].get<int>();
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + ", " + std::to_string(bits) + "-bit");
        const VulkanProbeFrames frames = vulkan_probe_frames(bits, expected["count"].get<int>(), width, height);
        ASSERT_EQ(frames.reference.size(), expected["reference"].size());
        for (std::size_t i = 0; i < frames.reference.size(); ++i) {
            EXPECT_EQ(sha256_of(frames.reference[i]), expected["reference"][i].get<std::string>()) << "frame " << i;
            EXPECT_EQ(sha256_of(frames.distorted[i]), expected["distorted"][i].get<std::string>()) << "frame " << i;
        }
    }
}

// On this PC's GPU the engine gives the Python app's features for those
// frames, and libvmaf predicts its scores from them, to the last digit.
TEST(VmafVulkan, ScoresAsThePythonApp) {
    if (!gpu()) {
        GTEST_SKIP() << "no GPU that Vulkan can calculate VMAF on";
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
        VulkanScorer scorer(width, height, bits, models, expected["n_subsample"].get<int>());
        for (std::size_t i = 0; i < frames.reference.size(); ++i) {
            scorer.add(frames.reference[i], frames.distorted[i]);
        }
        const PairScores scores = scorer.finish();
        EXPECT_EQ(scores.frames, expected["frames"].get<std::vector<unsigned>>());
        const VulkanFeatures measured = scorer.features();
        ASSERT_EQ(measured.frames.size(), expected["rows"].size());
        for (std::size_t i = 0; i < measured.frames.size(); ++i) {
            const std::vector<double> wanted = test::reference_numbers(expected["rows"][i]);
            ASSERT_EQ(wanted.size(), vulkan_feature_count);
            for (std::size_t feature = 0; feature < vulkan_feature_count; ++feature) {
                const double value = measured.columns[feature][i];
                EXPECT_TRUE(test::same_number(wanted[feature], value))
                    << "frame " << measured.frames[i] << ", feature " << feature << ": " << value << " for "
                    << wanted[feature];
            }
        }
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

// The probe gives the Python app's sums on this PC's GPU and accepts it.
// VML_VMAF_VULKAN_DEVICE names the GPU instead, with whitespace around its
// number taken off as Python's strip() takes it off; a number Vulkan does
// not list is no GPU.
TEST(VmafVulkan, ProbeAcceptsThisPcsGpu) {
    const std::optional<VulkanDevice> device = gpu();
    if (!device) {
        GTEST_SKIP() << "no GPU that Vulkan can calculate VMAF on";
    }
    const Json reference = test::load_reference("vulkan.json");
    EXPECT_EQ(vulkan_probe_sums(device->index, 8), reference["probe_sums"][0].get<std::string>());
    EXPECT_EQ(vulkan_probe_sums(device->index, 10), reference["probe_sums"][1].get<std::string>());

    set_device_variable("");
    VulkanProbe probe = probe_vulkan();
    EXPECT_TRUE(probe.available) << probe.text;
    EXPECT_EQ(probe.device, device->index);
    EXPECT_EQ(probe.text, "Vulkan on " + device->name);

    set_device_variable("\x1c " + std::to_string(device->index) + "\t");
    probe = probe_vulkan();
    EXPECT_TRUE(probe.available) << probe.text;
    EXPECT_EQ(probe.device, device->index);

    set_device_variable("99");
    probe = probe_vulkan();
    EXPECT_FALSE(probe.available);
    EXPECT_EQ(probe.text, "no GPU that Vulkan can calculate VMAF on");
    set_device_variable("");
}

// A scorer given no frames gives each of its models no scores, and a frame
// shorter than its luma plane is an error, not a read past its end.
TEST(VmafVulkan, NoFramesGiveNoScoresAndAShortFrameIsAnError) {
    if (!gpu()) {
        GTEST_SKIP() << "no GPU that Vulkan can calculate VMAF on";
    }
    VulkanScorer scorer(64, 32, 8, {{"vmaf", "vmaf_v0.6.1"}, {"vmaf_neg", "vmaf_v0.6.1neg"}});
    const std::vector<std::uint8_t> frame(64 * 32 - 1);
    EXPECT_THROW(scorer.add(frame, frame), VmafVulkanError);
    const PairScores scores = scorer.finish();
    EXPECT_TRUE(scores.frames.empty());
    ASSERT_EQ(scores.by_metric.size(), 2u);
    EXPECT_EQ(scores.by_metric[0].first, "vmaf");
    EXPECT_EQ(scores.by_metric[1].first, "vmaf_neg");
    EXPECT_TRUE(scores.by_metric[0].second.empty());
    EXPECT_TRUE(scores.by_metric[1].second.empty());
}

}  // namespace
}  // namespace vml
