// The GPU checks' answers kept between starts (gpu_checks.h).
#include "vml/gpu_checks.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "test_videos.h"

namespace vml {
namespace {

void write_text(const std::filesystem::path& file, const std::string& text) {
    std::ofstream(file, std::ios::binary) << text;
}

// A check's pass is used at a later start only on the PC as it was (its
// key); a GPU that failed is checked again, so a failure is never saved; a
// damaged file holds nothing; the oldest answers make room for new ones.
TEST(GpuChecks, OnlyPassesAreSavedAndEachIsFoundOnlyUnderItsKey) {
    const test::Folder folder;
    const std::filesystem::path file = folder / "gpu_checks.json";
    const nlohmann::json pass = {{"available", true}, {"device", 0}, {"text", "RTX"}, {"failed", false}};
    EXPECT_TRUE(gpu_check_passed("vulkan-probe", pass));
    EXPECT_FALSE(gpu_check_passed("vulkan-probe", {{"available", true}, {"failed", true}}));  // wrong sums
    EXPECT_FALSE(gpu_check_passed("cuda-probe", {{"available", false}, {"text", "no CUDA"}}));
    EXPECT_TRUE(gpu_check_passed("vship-probe", {{"device", {{"name", "RTX"}}}, {"reason", ""}}));
    EXPECT_FALSE(gpu_check_passed("vship-probe", {{"device", nullptr}, {"reason", "no GPU"}}));
    EXPECT_TRUE(gpu_check_passed("nvdec-probe", {{"h264_10bit", true}}));
    EXPECT_FALSE(gpu_check_passed("nvdec-probe", {{"h264_10bit", false}}));  // may be a failure to ask: asked again

    const nlohmann::json key = {{"gpus", {"RTX 10DE:2B85 driver 32.0.15.9692"}}, {"check", "vulkan-probe"}};
    EXPECT_FALSE(saved_gpu_check(file, key));  // no file yet
    save_gpu_check(file, key, pass);
    EXPECT_EQ(saved_gpu_check(file, key), pass);
    EXPECT_FALSE(saved_gpu_check(file, {{"gpus", {"RTX 10DE:2B85 driver 32.0.15.9700"}}, {"check", "vulkan-probe"}}));

    for (int other = 0; other < 16; ++other) {
        save_gpu_check(file, {{"other", other}}, pass);
    }
    EXPECT_FALSE(saved_gpu_check(file, key));  // the oldest went
    EXPECT_TRUE(saved_gpu_check(file, {{"other", 15}}));

    write_text(file, "{\"answers\": [");
    EXPECT_FALSE(saved_gpu_check(file, {{"other", 15}}));
}

// Vship's probe that crashed on one GPU's check runs again without it (its
// request's "skip"); that answer is saved under the first request too, which
// would otherwise never be answered from the file (its process crashes), so
// that every later start crashed a check process before the retry found its
// saved answer.
TEST(GpuChecks, AnAnswerFoundWithoutACrashingGpuAnswersTheFirstRequest) {
    const test::Folder folder;
    const std::filesystem::path file = folder / "gpu_checks.json";
    const nlohmann::json pc = {{"gpus", {"Radeon 1002:13C0 driver 32.0.21045.29003"}}};
    const nlohmann::json first = {{"vship_build", "vship-fast"}};
    nlohmann::json retry = first;
    retry["skip"] = {{1, "Vulkan GPU 1 (Radeon): its check crashed"}};
    const nlohmann::json pass = {{"device", {{"name", "RTX"}}}, {"reason", ""}};
    save_gpu_check(file, gpu_check_key(pc, "vship-probe", retry), pass);
    EXPECT_EQ(saved_gpu_check(file, gpu_check_key(pc, "vship-probe", first)), pass);
    EXPECT_FALSE(saved_gpu_check(file, gpu_check_key(pc, "vship-probe", {{"vship_build", "vship-hip"}})));
    EXPECT_FALSE(saved_gpu_check(file, gpu_check_key(pc, "vulkan-probe", first)));
}

// A Vulkan driver updated without a new driver version (its library
// replaced) changes the key; a library named without a folder, which the
// system finds, is known by its name.
TEST(GpuChecks, AVulkanManifestsLibraryIsPartOfTheKey) {
    const test::Folder folder;
    const std::filesystem::path manifest = folder / "icd.json";
    write_text(manifest, R"({"file_format_version": "1.0.1", "ICD": {"library_path": "./driver.dll"}})");
    write_text(folder / "driver.dll", "one");
    const nlohmann::json before = vulkan_manifest_stamps({manifest});
    write_text(folder / "driver.dll", "two, longer");
    EXPECT_NE(vulkan_manifest_stamps({manifest}), before);

    const std::filesystem::path layer = folder / "layer.json";
    write_text(layer, R"({"layers": [{"library_path": "overlay.dll"}]})");
    EXPECT_EQ(vulkan_manifest_stamps({layer})[0]["libraries"], nlohmann::json::array({"overlay.dll"}));
}

}  // namespace
}  // namespace vml
