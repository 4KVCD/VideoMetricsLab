// The app's SSIMULACRA2 and Butteraugli engines (cpu_pair_scorer.h) give
// the scores of libjxl 0.12.0's official tools to the last bit, on a small
// pair of odd size read as BT.709 SDR, as PQ, and with the source and the
// test read with different colours; and the same scores whatever number of
// threads Butteraugli's work is split between.
#include "vml/cpu_pair_scorer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "vml/colour.h"

namespace vml {
namespace {

constexpr int width = 67;
constexpr int height = 45;

// A picture with gradients and detail, and a copy of it with noise added,
// as 16-bit RGB samples. The expected scores below are the official tools'
// for these pictures written as 16-bit PNG files, each tagged with its own
// colours as the app tagged them for the tools.
void write_synthetic_pair(std::uint16_t* source, std::uint16_t* test) {
    for (long long y = 0; y < height; ++y) {
        for (long long x = 0; x < width; ++x) {
            for (long long c = 0; c < 3; ++c) {
                const long long value = (x * 977 + y * 613 + c * 20011 + ((x * y) % 97) * 331) % 65536;
                const long long noise = (((x * 31 + y * 17 + c * 7 + 1) * 2654435761LL) >> 20) & 0xFFF;
                const std::size_t at = static_cast<std::size_t>((y * width + x) * 3 + c);
                source[at] = static_cast<std::uint16_t>(value);
                test[at] = static_cast<std::uint16_t>(std::clamp(value + noise - 2048, 0LL, 65535LL));
            }
        }
    }
}

std::vector<double> scores(const VideoColour& source, const VideoColour& test, std::size_t butteraugli_threads) {
    CpuPairScorer scorer({"ssimulacra2", "butteraugli"}, source, test, width, height, butteraugli_threads);
    write_synthetic_pair(scorer.source_samples(), scorer.test_samples());
    return scorer.score();
}

const VideoColour bt709{1, 1, 1, false};

TEST(CpuPairScorer, GivesTheOfficialToolsScores) {
#if defined(_WIN32)
    const VideoColour pq_bt2020{9, 16, 9, false};
    const VideoColour hlg_bt2020{9, 18, 9, false};
    // ssimulacra2.exe and butteraugli_main.exe of libjxl 0.12.0's Windows
    // release, run with the source first (Butteraugli with --intensity_target
    // 203 where neither picture is HDR), the map's 3-norm taken as
    // butteraugli_norm3_top_row_first takes it.
    EXPECT_EQ(scores(bt709, bt709, 1), (std::vector<double>{72.45142825, 0.6687956082571304}));
    EXPECT_EQ(scores(pq_bt2020, pq_bt2020, 1), (std::vector<double>{51.91229247, 2.6565961682898442}));
    // Each picture read with its own colours; Butteraugli at the source's
    // brightness (PQ's, HLG's), or the tool's 80 nits for an SDR source, with
    // the test brought to the source's scale.
    EXPECT_EQ(scores(pq_bt2020, hlg_bt2020, 1), (std::vector<double>{-101.84809807, 95.89014463433348}));
    EXPECT_EQ(scores(hlg_bt2020, pq_bt2020, 1), (std::vector<double>{-117.84570985, 97.92656861863976}));
    EXPECT_EQ(scores(bt709, pq_bt2020, 1), (std::vector<double>{-523.52227903, 130.4644224855373}));
#else
    // The Mac's reference is its own build of the tools (without fused
    // multiply-adds), whose scores of this pair are not recorded here yet.
    GTEST_SKIP() << "the tools' scores of this pair are recorded for Windows only";
#endif
}

TEST(CpuPairScorer, ButteraugliThreadsDoNotChangeTheScores) {
    const std::vector<double> one_thread = scores(bt709, bt709, 1);
    EXPECT_EQ(scores(bt709, bt709, 3), one_thread);
    // More threads than rows of the half-size level.
    EXPECT_EQ(scores(bt709, bt709, 32), one_thread);
}

}  // namespace
}  // namespace vml
