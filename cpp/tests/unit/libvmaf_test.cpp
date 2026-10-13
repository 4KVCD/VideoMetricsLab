#include "vml/libvmaf.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "reference.h"
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
    // The headers in third_party/libvmaf-fast are release v1's. A local build
    // after it ("v1-4-g994b1814") has the same while the fork keeps
    // libvmaf's headers as they are; another release needs its own.
    const std::string bundled = libvmaf_fast_version();
    EXPECT_TRUE(bundled == "v1" || bundled.starts_with("v1-"))
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
