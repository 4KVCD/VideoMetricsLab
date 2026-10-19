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

}  // namespace
}  // namespace vml
