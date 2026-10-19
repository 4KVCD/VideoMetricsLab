// The Bitrate tab's numbers held to the Python app's (reference/bitrate.json,
// from its bitrate.py): ffprobe's lines of packets read, and each stream's
// graphs, one-second bins and summary; and, with FFmpeg, a real clip's
// packets read with ffprobe.
#include "vml/bitrate.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "reference.h"
#include "test_videos.h"
#include "vml/ffprobe.h"
#include "vml/platform/system.h"

namespace vml {
namespace {

using Json = nlohmann::json;

// Whether `actual` holds the Python app's numbers, `expected`.
template <typename Number>
void expect_numbers(const Json& expected, const std::vector<Number>& actual, const std::string& what) {
    const std::vector<double> numbers = test::reference_numbers(expected);
    ASSERT_EQ(actual.size(), numbers.size()) << what;
    for (std::size_t index = 0; index < numbers.size(); ++index) {
        EXPECT_TRUE(test::same_number(numbers[index], static_cast<double>(actual[index])))
            << what << " [" << index << "]: " << numbers[index] << " vs " << actual[index];
    }
}

void expect_plot(const Json& expected, const BitratePlot& plot, const std::string& what) {
    expect_numbers(expected.at("times"), plot.times, what + " times");
    expect_numbers(expected.at("values"), plot.values, what + " values");
    EXPECT_EQ(plot.axis_label, expected.at("axis_label").get<std::string>()) << what;
    EXPECT_EQ(plot.value_label, expected.at("value_label").get<std::string>()) << what;
}

TEST(Bitrate, ReadsThePacketsAndDrawsTheGraphsAsThePythonAppDoes) {
    const Json reference = test::load_reference("bitrate.json");
    const Json& reads = reference.at("reads");
    ASSERT_FALSE(reads.empty());
    for (const Json& item : reads) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        PacketList packets(*test::reference_number(item.at("fps")));
        for (const Json& line : item.at("lines")) {
            packets.add(line.get<std::string>());
        }
        const Json& expected = item.at("expected");
        if (expected.contains("error")) {
            try {
                packets.finish("clip.mkv");
                ADD_FAILURE() << "no error";
            } catch (const BitrateError& error) {
                EXPECT_EQ(std::string(error.what()), expected.at("error").get<std::string>());
            }
            continue;
        }
        const BitrateData data = packets.finish("clip.mkv");
        expect_numbers(expected.at("times"), data.times, "times");
        expect_numbers(expected.at("durations"), data.durations, "durations");
        EXPECT_EQ(data.sizes, expected.at("sizes").get<std::vector<long long>>());
        EXPECT_EQ(data.keyframes, expected.at("keyframes").get<std::vector<bool>>());
    }

    const Json& streams = reference.at("streams");
    ASSERT_FALSE(streams.empty());
    for (const Json& item : streams) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const BitrateData data = bitrate_data(
            "stream.mkv", test::reference_numbers(item.at("times")), test::reference_numbers(item.at("durations")),
            item.at("sizes").get<std::vector<long long>>(), item.at("keyframes").get<std::vector<bool>>());
        EXPECT_TRUE(test::same_number(*test::reference_number(item.at("start_time")), data.start_time()));
        EXPECT_TRUE(test::same_number(*test::reference_number(item.at("end_time")), data.end_time()));
        EXPECT_TRUE(test::same_number(*test::reference_number(item.at("duration")), data.duration()));
        EXPECT_EQ(data.total_bytes(), item.at("total_bytes").get<long long>());
        const Json& summary = item.at("summary");
        EXPECT_TRUE(test::same_number(*test::reference_number(summary.at("average_kbps")), data.summary.average_kbps));
        EXPECT_TRUE(test::same_number(*test::reference_number(summary.at("minimum_kbps")), data.summary.minimum_kbps));
        EXPECT_TRUE(test::same_number(*test::reference_number(summary.at("maximum_kbps")), data.summary.maximum_kbps));
        for (const bool adjust : {true, false}) {
            const std::string variant = adjust ? "true" : "false";
            const SecondBins bins = second_bins(data, adjust);
            const Json& expected = item.at("bins").at(variant);
            expect_numbers(expected.at("starts"), bins.starts, "bin starts " + variant);
            expect_numbers(expected.at("ends"), bins.ends, "bin ends " + variant);
            expect_numbers(expected.at("rates"), bins.rates, "bin rates " + variant);
            const Json& plots = item.at("plots");
            expect_plot(plots.at("frame|" + variant), frame_plot(data, adjust), "frame " + variant);
            expect_plot(plots.at("second|" + variant), second_plot(data, adjust), "second " + variant);
            expect_plot(plots.at("gop|" + variant), gop_plot(data, adjust), "gop " + variant);
        }
    }
}

}  // namespace
}  // namespace vml
