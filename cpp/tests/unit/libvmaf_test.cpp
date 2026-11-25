#include "vml/libvmaf.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "reference.h"
#include "vml/gpu_cpu_metrics.h"
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
    // The headers in third_party/libvmaf-fast are release v1's, which v1.1,
    // v1.2 and v1.3 left as they were (they changed only the Vulkan engine). A
    // local build after one ("v1.3-2-g...") has the same while the fork keeps
    // libvmaf's headers as they are; another release needs its own.
    const std::string bundled = libvmaf_fast_version();
    EXPECT_TRUE(bundled == "v1" || bundled == "v1.1" || bundled == "v1.2" || bundled == "v1.3" ||
                bundled.starts_with("v1-") || bundled.starts_with("v1.1-") || bundled.starts_with("v1.2-") ||
                bundled.starts_with("v1.3-"))
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

// A 10-bit 4:2:0 pair of an odd size, packed as FFmpeg writes it: chroma
// rounded up. `same_chroma` gives the test the reference's chroma.
constexpr int yuv_width = 65, yuv_height = 39;

std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>> yuv_pair(int index, bool same_chroma) {
    const PackedFrame packed(yuv_width, yuv_height, 10);
    std::vector<std::uint8_t> reference(packed.bytes()), test(packed.bytes());
    const std::size_t luma_samples = static_cast<std::size_t>(yuv_width) * yuv_height;
    for (std::size_t i = 0; i < reference.size() / 2; ++i) {
        const auto sample = static_cast<std::uint16_t>((i * 7 + index * 13) % 1024);
        const bool keep = same_chroma && i >= luma_samples;
        const auto changed = static_cast<std::uint16_t>(keep ? sample : (sample + i % 5 + (i % 3) * 9) % 1024);
        reference[2 * i] = static_cast<std::uint8_t>(sample);
        reference[2 * i + 1] = static_cast<std::uint8_t>(sample >> 8);
        test[2 * i] = static_cast<std::uint8_t>(changed);
        test[2 * i + 1] = static_cast<std::uint8_t>(changed >> 8);
    }
    return {reference, test};
}

PairScores score_yuv_pairs(const std::vector<std::string>& metrics, bool same_chroma) {
    CpuScorer scorer(yuv_width, yuv_height, 10, metrics, 1, 1);
    for (int index = 0; index < 2; ++index) {
        const auto [reference, test] = yuv_pair(index, same_chroma);
        scorer.add(reference, test);
    }
    return scorer.finish();
}

const std::vector<double>& scores_of(const PairScores& scores, const std::string& key) {
    const auto found = std::ranges::find(scores.by_metric, key, &std::pair<std::string, std::vector<double>>::first);
    EXPECT_NE(found, scores.by_metric.end()) << key;
    return found->second;
}

// PSNR YUV is (6 Y + Cb + Cr) / 8 of the three planes' PSNRs, and PSNR YUV
// avg. MSE the PSNR of (6 Y + Cb + Cr) / 8 of their MSEs, each worked out
// here from the samples: the chroma's whole samples only (32 x 19 of the 33 x
// 20 an odd 65 x 39 frame has), at the 10-bit peak of 1023.
TEST(Libvmaf, PsnrYuvWeighsTheLumaSixTimesTheChroma) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    const PackedFrame packed(yuv_width, yuv_height, 10);
    const PairScores scores = score_yuv_pairs({"psnr_yuv", "psnr_yuv_mse"}, false);
    const std::vector<double>&yuv = scores_of(scores, "psnr_yuv"), &yuv_mse = scores_of(scores, "psnr_yuv_mse");
    ASSERT_EQ(yuv.size(), 2U);
    ASSERT_EQ(yuv_mse.size(), 2U);
    for (int index = 0; index < 2; ++index) {
        const auto [reference, test] = yuv_pair(index, false);
        const FramePlanes reference_planes = packed.planes(reference.data()), test_planes = packed.planes(test.data());
        const auto plane_mse = [&](int plane, int columns, int rows) {
            double sse = 0;
            for (int row = 0; row < rows; ++row) {
                const auto* a = reinterpret_cast<const std::uint16_t*>(reference_planes[plane].data +
                                                                       row * reference_planes[plane].stride);
                const auto* b = reinterpret_cast<const std::uint16_t*>(test_planes[plane].data +
                                                                       row * test_planes[plane].stride);
                for (int column = 0; column < columns; ++column) {
                    const double difference = static_cast<double>(a[column]) - b[column];
                    sse += difference * difference;
                }
            }
            return sse / (columns * rows);
        };
        const auto psnr = [](double mse) { return 10.0 * std::log10(1023.0 * 1023.0 / mse); };
        const double y = plane_mse(0, yuv_width, yuv_height), cb = plane_mse(1, yuv_width / 2, yuv_height / 2),
                     cr = plane_mse(2, yuv_width / 2, yuv_height / 2);
        const auto frame = static_cast<std::size_t>(index);
        EXPECT_NEAR(yuv[frame], (6 * psnr(y) + psnr(cb) + psnr(cr)) / 8, 1e-6) << "frame " << index;
        EXPECT_NEAR(yuv_mse[frame], psnr((6 * y + cb + cr) / 8), 1e-6) << "frame " << index;
        EXPECT_LT(yuv_mse[frame], yuv[frame]) << "frame " << index;
    }
}

// The 6:1:1 metrics leave PSNR and SSIM as they were (XPSNR's own PSNR
// takes PSNR's place with XPSNR on), and chroma that matches counts as SSIM 1
// and as PSNR's 72 dB ceiling at 10 bits, in PSNR YUV avg. MSE as that
// ceiling's error.
TEST(Libvmaf, YuvMetricsLeaveTheLumaScoresAsTheyWere) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    const PairScores luma = score_yuv_pairs({"psnr", "ssim"}, true);
    const PairScores all = score_yuv_pairs({"psnr", "ssim", "xpsnr", "psnr_yuv", "psnr_yuv_mse", "ssim_yuv"}, true);
    EXPECT_EQ(scores_of(all, "psnr"), scores_of(luma, "psnr"));
    EXPECT_EQ(scores_of(all, "ssim"), scores_of(luma, "ssim"));
    const std::vector<double>&psnr = scores_of(luma, "psnr"), &ssim = scores_of(luma, "ssim");
    const std::vector<double>&psnr_yuv = scores_of(all, "psnr_yuv"), &ssim_yuv = scores_of(all, "ssim_yuv");
    ASSERT_EQ(psnr_yuv.size(), psnr.size());
    for (std::size_t i = 0; i < psnr.size(); ++i) {
        EXPECT_NEAR(psnr_yuv[i], (6 * psnr[i] + 2 * 72.0) / 8, 1e-6);
        const auto error = [](double db) { return std::pow(10.0, -db / 10); };
        EXPECT_NEAR(scores_of(all, "psnr_yuv_mse")[i], -10 * std::log10((6 * error(psnr[i]) + 2 * error(72.0)) / 8),
                    1e-6);
        EXPECT_NEAR(ssim_yuv[i], (6 * ssim[i] + 2.0) / 8, 1e-6);
    }
}

// A metric's planes (MetricDefinition::plane_of) are its own planes' scores,
// read in its pass: they leave the metric as it was, Y is the luma's score,
// and (6 Y + U + V) / 8 gives the 6:1:1 metric back. A plane is refused
// where its metric is.
TEST(Libvmaf, PlanesAreTheirMetricsPlanes) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    const PairScores alone = score_yuv_pairs({"psnr", "ssim", "psnr_yuv", "ssim_yuv", "psnr_hvs"}, false);
    const PairScores planes = score_yuv_pairs({"psnr", "ssim", "psnr_yuv", "psnr_yuv_y", "psnr_yuv_u", "psnr_yuv_v",
                                               "ssim_yuv", "ssim_yuv_y", "ssim_yuv_u", "ssim_yuv_v", "psnr_hvs",
                                               "psnr_hvs_y", "psnr_hvs_u", "psnr_hvs_v"},
                                              false);
    for (const char* key : {"psnr", "ssim", "psnr_yuv", "ssim_yuv", "psnr_hvs"}) {
        EXPECT_EQ(scores_of(planes, key), scores_of(alone, key)) << key;
    }
    EXPECT_EQ(scores_of(planes, "psnr_yuv_y"), scores_of(alone, "psnr"));
    EXPECT_EQ(scores_of(planes, "ssim_yuv_y"), scores_of(alone, "ssim"));
    for (const std::string metric : {"psnr_yuv", "ssim_yuv"}) {
        const std::vector<double>&y = scores_of(planes, metric + "_y"), &u = scores_of(planes, metric + "_u"),
                            &v = scores_of(planes, metric + "_v"), &yuv = scores_of(planes, metric);
        ASSERT_EQ(y.size(), yuv.size());
        for (std::size_t i = 0; i < yuv.size(); ++i) {
            // libvmaf gives each score to six decimals.
            EXPECT_NEAR(yuv[i], (6 * y[i] + u[i] + v[i]) / 8, 1e-6) << metric << " frame " << i;
            EXPECT_NE(u[i], y[i]) << metric << ": the chroma differs here, so its score must too";
        }
    }
    EXPECT_EQ(scores_of(planes, "psnr_hvs_v").size(), 2U);
    EXPECT_EQ(cpu_metrics_unfit({"ms_ssim_yuv", "ms_ssim_yuv_u"}, 320, 240, 8, {}).size(), 2U);
}

// VMAF YUV's U and V are the run's VMAF model applied to each chroma plane:
// with only the test's U changed, V scores as identical planes do and U lower.
// Chroma below VMAF's 32 x 32 is refused.
TEST(Libvmaf, VmafOfEachChromaPlaneReadsThatPlane) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    constexpr int width = 128, height = 96, frames = 2;
    const PackedFrame packed(width, height, 10);
    const std::size_t luma = static_cast<std::size_t>(width) * height, chroma = luma / 4;
    const auto picture = [&](int index, bool change_u) {
        std::vector<std::uint8_t> bytes(packed.bytes());
        for (std::size_t i = 0; i < bytes.size() / 2; ++i) {
            auto sample = static_cast<std::uint16_t>(64 + (i * 7 + index * 13 + (i / width) * 3) % 800);
            if (change_u && i >= luma && i < luma + chroma) {
                sample = 512;  // a flat U: all its detail lost
            }
            bytes[2 * i] = static_cast<std::uint8_t>(sample);
            bytes[2 * i + 1] = static_cast<std::uint8_t>(sample >> 8);
        }
        return bytes;
    };
    const auto scored = [&](bool change_u) {
        CpuScorer scorer(width, height, 10, {"vmaf_yuv_u", "vmaf_yuv_v"}, 1, 1, 0, {}, std::nullopt,
                         "version=vmaf_v0.6.1");
        for (int index = 0; index < frames; ++index) {
            scorer.add(picture(index, false), picture(index, change_u));
        }
        return scorer.finish();
    };
    const PairScores same = scored(false), u_changed = scored(true);
    ASSERT_EQ(scores_of(u_changed, "vmaf_yuv_u").size(), static_cast<std::size_t>(frames));
    EXPECT_EQ(scores_of(u_changed, "vmaf_yuv_v"), scores_of(same, "vmaf_yuv_v"));
    for (std::size_t i = 0; i < frames; ++i) {
        EXPECT_LT(scores_of(u_changed, "vmaf_yuv_u")[i], scores_of(same, "vmaf_yuv_u")[i]) << "frame " << i;
    }
    EXPECT_EQ(cpu_metrics_unfit({"vmaf_yuv_u", "vmaf_yuv_v"}, 62, 96, 8, {}).size(), 2U);
    EXPECT_TRUE(cpu_metrics_unfit({"vmaf_yuv_u", "vmaf_yuv_v"}, 64, 64, 8, {}).empty());
}

// The app's metrics a comparison cannot score are refused up front, each with
// its reason, so they fail alone: libvmaf would fail every metric scored with
// them. 160x90 is below MS-SSIM's 176 and CAMBI's 216; DCI 4K is past CAMBI's
// largest window; 14 bits is past PSNR-HVS-M's 12 and not one of CIEDE2000's.
TEST(Libvmaf, MetricsThatCannotScoreAComparisonAreRefusedAlone) {
    const std::vector<std::string> all{"psnr", "ms_ssim", "ms_ssim_yuv", "psnr_hvs", "cambi", "ciede2000"};
    const auto refused = [&](int width, int height, int bit_depth, CambiInput cambi = {}) {
        std::vector<std::string> keys;
        for (const auto& [key, reason] : cpu_metrics_unfit(all, width, height, bit_depth, cambi)) {
            EXPECT_FALSE(reason.empty()) << key;
            keys.push_back(key);
        }
        return keys;
    };
    EXPECT_EQ(refused(160, 90, 8), (std::vector<std::string>{"cambi", "ms_ssim", "ms_ssim_yuv"}));
    EXPECT_EQ(refused(320, 240, 8), (std::vector<std::string>{"ms_ssim_yuv"}));  // chroma 160x120
    EXPECT_TRUE(refused(1920, 1080, 10).empty());
    EXPECT_EQ(refused(4096, 2160, 10), (std::vector<std::string>{"cambi"}));
    EXPECT_EQ(refused(1920, 1080, 14), (std::vector<std::string>{"ciede2000", "psnr_hvs"}));
}

// CAMBI's score is found under the name libvmaf gives it for its options (an
// 8-bit test video of a smaller size, PQ), so a wrong name fails the scoring;
// and CIEDE2000 of identical pictures (libvmaf's +inf) is a difference of 0.
TEST(Libvmaf, CambiWithItsOptionsAndCiedeOfIdenticalPicturesAreScored) {
    if (!std::filesystem::exists(libvmaf_folder() / platform::shared_library_name("libvmaf"))) {
        GTEST_SKIP() << "the app ships no libvmaf-fast for this system yet";
    }
    constexpr int width = 320, height = 240;
    const PackedFrame packed(width, height, 10);
    std::vector<std::uint8_t> picture(packed.bytes());
    for (std::size_t i = 0; i < picture.size() / 2; ++i) {
        const auto sample = static_cast<std::uint16_t>(64 + (i / 7) % 512);  // a slow ramp: bands
        picture[2 * i] = static_cast<std::uint8_t>(sample);
        picture[2 * i + 1] = static_cast<std::uint8_t>(sample >> 8);
    }
    for (const CambiInput& cambi : {CambiInput{256, 192, 8, true}, CambiInput{width, height, 10, true}}) {
        CpuScorer scorer(width, height, 10, {"cambi", "ciede2000"}, 1, 1, 0, cambi);
        scorer.add(picture, picture);
        const PairScores scores = scorer.finish();
        ASSERT_EQ(scores_of(scores, "cambi").size(), 1U) << cambi.width;
        EXPECT_GE(scores_of(scores, "cambi")[0], 0.0);
        EXPECT_EQ(scores_of(scores, "ciede2000"), std::vector<double>{0.0});
    }
}

// CIEDE2000 goes to the GPU unless the size is odd (the engine's chroma
// planes are larger than libvmaf's), CAMBI unless its test video is scored at
// another size or bit depth (options the GPU does not take); PQ goes.
TEST(Libvmaf, OnlyTheCpuMetricsTheGpuCalculatesAsLibvmafGoToIt) {
    using Options = std::vector<std::pair<std::string, std::string>>;
    const std::vector<std::string> metrics{"psnr", "ciede2000", "ms_ssim", "cambi"};
    EXPECT_EQ(gpu_metrics_of(metrics, false, Options{{"eotf", "pq"}}),
              (std::vector<std::string>{"ciede2000", "cambi"}));
    EXPECT_EQ(gpu_metrics_of(metrics, true, {}), std::vector<std::string>{"cambi"});
    EXPECT_EQ(gpu_metrics_of(metrics, false, Options{{"enc_bitdepth", "8"}}), std::vector<std::string>{"ciede2000"});
    EXPECT_EQ(gpu_metrics_of(metrics, false, Options{{"enc_height", "1080"}, {"enc_width", "1920"}}),
              std::vector<std::string>{"ciede2000"});
    EXPECT_TRUE(gpu_metrics_of({"psnr", "ms_ssim"}, false, {}).empty());
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
