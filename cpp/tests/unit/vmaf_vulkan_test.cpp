// The Vulkan engine's binding held to the Python app's (vulkan.json, from
// vmaf_vulkan.py): the probe's frames on every system, and, on a PC with a
// GPU Vulkan lists, the features, the scores and the probe.
#include "vml/vmaf_vulkan.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <system_error>

#include "reference.h"
#include "vml/gpu_choice.h"
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

// The GPU chosen (gpu_choice.h) is found by its whole name or a part of one,
// in any case, a whole name first; VMAF runs on it, and on no GPU (the CPU)
// when Vulkan lists none of that name or cannot use it: not on another GPU,
// which the choice may keep free.
TEST(VmafVulkan, VmafRunsOnTheGpuChosen) {
    EXPECT_EQ(chosen_among({"AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 5090"}, "nvidia geforce rtx 5090"), 1u);
    EXPECT_EQ(chosen_among({"AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 5090"}, " radeon "), 0u);
    EXPECT_EQ(chosen_among({"Intel(R) Arc(TM) Graphics 2", "Intel(R) Arc(TM) Graphics"}, "Intel(R) Arc(TM) Graphics"),
              1u);
    EXPECT_EQ(chosen_among({"AMD Radeon(TM) Graphics"}, "titan"), std::nullopt);
    EXPECT_EQ(chosen_among({"AMD Radeon(TM) Graphics"}, ""), std::nullopt);

    const std::vector<VulkanDevice> found{{0, "NVIDIA GeForce RTX 5090", 0x10DE, 2, true, true},
                                          {1, "AMD Radeon(TM) Graphics", 0x1002, 1, true, true},
                                          {2, "llvmpipe", 0x10005, 4, true, true},
                                          {3, "Intel(R) UHD Graphics 770", 0x8086, 1, false, true}};
    const auto index_of = [&found] {
        const std::optional<VulkanDevice> device = chosen_vulkan_device(found);
        return device ? device->index : -1;
    };
    const std::optional<std::string> before = chosen_gpu();
    choose_gpu(std::nullopt);
    EXPECT_EQ(index_of(), 0);  // a discrete one first
    choose_gpu("Radeon");
    EXPECT_EQ(index_of(), 1);
    choose_gpu("titan");
    EXPECT_EQ(index_of(), -1);  // none of that name: the CPU
    choose_gpu("Intel(R) UHD Graphics 770");
    EXPECT_EQ(index_of(), -1);  // one Vulkan cannot use (no 64-bit integers): the CPU
    choose_gpu(before);
}

// A part of a name that two GPUs have names neither: each API lists the GPUs
// in its own order (DirectX's, Vulkan's, Vship's discrete-first), and the
// first of each that has it could be a different GPU (D3D11VA decoding on the
// Radeon iGPU while VMAF scores on the Radeon card). A whole name still names
// its GPU.
TEST(VmafVulkan, APartOfTwoGpusNamesNamesNeither) {
    const std::vector<std::string> directx{"AMD Radeon RX 7900 XTX", "AMD Radeon(TM) Graphics"};
    EXPECT_EQ(gpus_named(directx, "radeon"), (std::vector<std::size_t>{0, 1}));
    EXPECT_EQ(chosen_among(directx, "radeon"), std::nullopt);
    EXPECT_EQ(chosen_among(directx, "amd radeon(tm) graphics"), 1u);
    EXPECT_EQ(chosen_among(directx, "7900"), 0u);

    const std::vector<VulkanDevice> found{{0, "AMD Radeon(TM) Graphics", 0x1002, 1, true, true},
                                          {1, "AMD Radeon RX 7900 XTX", 0x1002, 2, true, true}};
    const std::optional<std::string> before = chosen_gpu();
    choose_gpu("radeon");
    const std::optional<VulkanDevice> device = chosen_vulkan_device(found);
    choose_gpu(before);
    EXPECT_FALSE(device.has_value()) << device->name;
}

// NVIDIA's decoder and libvmaf's CUDA code use the NVIDIA GPU chosen, by
// CUDA's number for it, where there are two: they took CUDA's first GPU
// whatever was chosen. With one NVIDIA GPU, or none chosen, or another
// maker's GPU chosen, CUDA's first, as before, without asking CUDA; an
// NVIDIA GPU CUDA does not list (CUDA_VISIBLE_DEVICES) is no CUDA GPU at all.
TEST(VmafVulkan, NvidiasDecoderAndCudaUseTheNvidiaGpuChosen) {
    const std::vector<platform::GpuAdapter> two{{0, 0x10DE, "NVIDIA GeForce RTX 5090"},
                                                {1, 0x1002, "AMD Radeon(TM) Graphics"},
                                                {2, 0x10DE, "NVIDIA GeForce RTX 3070"}};
    int asked = 0;
    const auto cuda = [&asked] {
        ++asked;
        return std::vector<std::string>{"NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3070"};
    };
    EXPECT_EQ(chosen_cuda_device_among("NVIDIA GeForce RTX 3070", two, cuda), 1);
    EXPECT_EQ(chosen_cuda_device_among("NVIDIA GeForce RTX 5090", two, cuda), 0);
    EXPECT_EQ(asked, 2);
    EXPECT_EQ(chosen_cuda_device_among(std::nullopt, two, cuda), 0);
    EXPECT_EQ(chosen_cuda_device_among("AMD Radeon(TM) Graphics", two, cuda), 0);
    const std::vector<platform::GpuAdapter> one{{0, 0x1002, "AMD Radeon(TM) Graphics"},
                                                {1, 0x10DE, "NVIDIA GeForce RTX 3070"}};
    EXPECT_EQ(chosen_cuda_device_among("NVIDIA GeForce RTX 3070", one, cuda), 0);
    EXPECT_EQ(asked, 2);  // CUDA not started for these
    // Two of one name: the place among them, in CUDA's order
    const std::vector<platform::GpuAdapter> same{{0, 0x10DE, "NVIDIA GeForce RTX 3070"},
                                                 {1, 0x10DE, "NVIDIA GeForce RTX 3070"}};
    const auto both = [] { return std::vector<std::string>{"NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3070"}; };
    EXPECT_EQ(chosen_cuda_device_among("NVIDIA GeForce RTX 3070 #2", same, both), 1);
    EXPECT_EQ(chosen_cuda_device_among("NVIDIA GeForce RTX 3070 #1", same, both), 0);
    // Hidden from CUDA: none, not CUDA's first
    const auto hidden = [] { return std::vector<std::string>{"NVIDIA GeForce RTX 5090"}; };
    EXPECT_EQ(chosen_cuda_device_among("NVIDIA GeForce RTX 3070", two, hidden), std::nullopt);
    // No system list (Linux): CUDA's own, else its first
    EXPECT_EQ(chosen_cuda_device_among("rtx 3070", {}, cuda), 1);
    EXPECT_EQ(chosen_cuda_device_among("radeon", {}, cuda), 0);
}

// Of two GPUs of one name (two RTX 3070s), the second can be chosen: the
// choice saved is the name and its place among them, which every API matches
// in its own list. A name alone is the first of that name, as it was.
TEST(VmafVulkan, TheSecondOfTwoGpusOfOneNameCanBeChosen) {
    const std::vector<std::string> directx{"NVIDIA GeForce RTX 3070", "AMD Radeon(TM) Graphics",
                                           "NVIDIA GeForce RTX 3070"};
    EXPECT_EQ(gpu_choice_texts(directx), (std::vector<std::string>{"NVIDIA GeForce RTX 3070 #1",
                                                                   "AMD Radeon(TM) Graphics",
                                                                   "NVIDIA GeForce RTX 3070 #2"}));
    EXPECT_EQ(chosen_among(directx, "nvidia geforce rtx 3070 #2"), 2u);
    EXPECT_EQ(chosen_among(directx, "NVIDIA GeForce RTX 3070 #1"), 0u);
    EXPECT_EQ(chosen_among(directx, "NVIDIA GeForce RTX 3070"), 0u);
    EXPECT_EQ(chosen_among(directx, "NVIDIA GeForce RTX 3070 #3"), std::nullopt);
    EXPECT_EQ(chosen_among(directx, "rtx 3070"), std::nullopt);  // a part of two names
    EXPECT_EQ(chosen_among({"NVIDIA GeForce RTX 3070"}, "NVIDIA GeForce RTX 3070 #1"), 0u);

    const std::vector<VulkanDevice> found{{0, "NVIDIA GeForce RTX 3070", 0x10DE, 2, true, true},
                                          {1, "NVIDIA GeForce RTX 3070", 0x10DE, 2, true, true}};
    const std::optional<std::string> before = chosen_gpu();
    choose_gpu("NVIDIA GeForce RTX 3070 #2");
    const std::optional<VulkanDevice> device = chosen_vulkan_device(found);
    choose_gpu(before);
    ASSERT_TRUE(device.has_value());
    EXPECT_EQ(device->index, 1);
}

}  // namespace
}  // namespace vml
