#include "vml/libvmaf.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "reference.h"
#include "vml/libvmaf_cpu.h"
#include "vml/pair_scorer.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

// The bundled libvmaf predicts the Python app's scores from the same
// features (reference/predict.json, from vmaf_vulkan.predict), to the
// last digit: they are the same library's numbers, kept to 6 decimals the
// same way.
TEST(Libvmaf, PredictsVmafFromFeaturesAsThePythonApp) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    // The headers in third_party/libvmaf-fast are release v1's, which v1.1
    // and v1.2 left as they were (they changed only the Vulkan engine). A
    // local build after one ("v1.2-2-g...") has the same while the fork keeps
    // libvmaf's headers as they are; another release needs its own.
    const std::string bundled = libvmaf_fast_version();
    EXPECT_TRUE(bundled == "v1" || bundled == "v1.1" || bundled == "v1.2" || bundled.starts_with("v1-") ||
                bundled.starts_with("v1.1-") || bundled.starts_with("v1.2-"))
        << bundled << ": the headers in third_party/libvmaf-fast are release v1's; copy this library's with it";
    const nlohmann::json reference = test::load_reference("predict.json");
    ASSERT_FALSE(reference.empty());
    for (const nlohmann::json& expected : reference) {
        const std::string version = expected["version"];
        SCOPED_TRACE(version);
        std::vector<std::pair<std::string, std::vector<double>>> columns;
        for (const auto& [name, values] : expected["features"].items()) {
            columns.emplace_back(name, test::reference_numbers(values));
        }
        std::vector<FeatureValues> features;
        for (const auto& [name, values] : columns) {
            features.emplace_back(name, values);
        }
        const std::vector<double> scores =
            predict_scores(version, expected["frames"].get<std::vector<unsigned>>(), features);
        const std::vector<double> wanted = test::reference_numbers(expected["scores"]);
        ASSERT_EQ(scores.size(), wanted.size());
        for (std::size_t i = 0; i < scores.size(); ++i) {
            EXPECT_EQ(scores[i], wanted[i]) << "frame " << expected["frames"][i];
        }
    }
}

// The CPU's metrics score planes read where a decoder holds them, rows
// padded with whatever its memory holds, as they score the same frames
// packed: only each row's samples are read.
TEST(Libvmaf, CpuMetricsReadPaddedPlanesAsPackedOnes) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    constexpr int width = 66, height = 40, bit_depth = 10, frames = 3;
    const PackedFrame packed(width, height, bit_depth);
    constexpr std::size_t rows = height, row_bytes = width * 2, padded = row_bytes + 30;
    const auto frame = [&](int index, int test) {
        std::vector<std::uint8_t> bytes(packed.bytes());
        for (std::size_t i = 0; i < bytes.size() / 2; ++i) {
            const auto sample = static_cast<std::uint16_t>((i * 7 + index * 13 + test * (i % 5)) % 1024);
            bytes[2 * i] = static_cast<std::uint8_t>(sample);
            bytes[2 * i + 1] = static_cast<std::uint8_t>(sample >> 8);
        }
        return bytes;
    };
    const auto scored = [&](bool as_planes) {
        CpuScorer scorer(width, height, bit_depth, {"psnr", "ssim"}, 1, 1);
        for (int index = 0; index < frames; ++index) {
            const std::vector<std::uint8_t> reference = frame(index, 0), test = frame(index, 1);
            if (!as_planes) {
                scorer.add(reference, test);
                continue;
            }
            // The lumas, their rows spaced out, the gaps filled with noise.
            std::vector<std::uint8_t> reference_luma(padded * rows, 0xa5), test_luma(padded * rows, 0x5a);
            for (std::size_t row = 0; row < rows; ++row) {
                std::copy_n(reference.data() + row * row_bytes, row_bytes, reference_luma.data() + row * padded);
                std::copy_n(test.data() + row * row_bytes, row_bytes, test_luma.data() + row * padded);
            }
            const auto stride = static_cast<std::ptrdiff_t>(padded);
            scorer.add_planes(FramePlanes{Plane{reference_luma.data(), stride}},
                              FramePlanes{Plane{test_luma.data(), stride}});
        }
        return scorer.finish();
    };
    const PairScores from_frames = scored(false), from_planes = scored(true);
    ASSERT_EQ(from_frames.frames.size(), static_cast<std::size_t>(frames));
    EXPECT_EQ(from_planes.frames, from_frames.frames);
    EXPECT_EQ(from_planes.by_metric, from_frames.by_metric);
}

// A model libvmaf does not have is an error that says what failed.
TEST(Libvmaf, AnUnknownModelIsAnError) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    try {
        predict_scores("vmaf_no_such_model", {0}, {});
        FAIL() << "no error";
    } catch (const LibvmafError& error) {
        EXPECT_EQ(std::string(error.what()).rfind("Loading the vmaf_no_such_model model failed (libvmaf error ", 0), 0u);
    }
}

}  // namespace
}  // namespace vml
