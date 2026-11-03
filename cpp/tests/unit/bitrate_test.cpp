// The Bitrate tab's numbers held to the Python app's (reference/bitrate.json,
// from its bitrate.py): ffprobe's lines of packets read, and each stream's
// graphs, one-second bins and summary; and, with FFmpeg, a real clip's
// packets read with ffprobe.
#include "vml/bitrate.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
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

// The graphs' figures as the command line's CSV files write them: each
// frame, second and group of pictures, from the stream's start or, without
// adjust_start, on the file's timeline in all three (a stream starting 10 s
// into its file).
TEST(Bitrate, WritesEachGraphsFiguresAsCsv) {
    const BitrateData data = bitrate_data("clip.mkv", {10.0, 10.5, 11.0, 11.5}, {0.5, 0.5, 0.5, 0.5},
                                          {1000, 500, 2000, 500}, {true, false, true, false});
    const GopBins gops = gop_bins(data, true);
    EXPECT_EQ(gops.frames, (std::vector<long long>{2, 2}));
    EXPECT_EQ(gops.bytes, (std::vector<long long>{1500, 2500}));
    EXPECT_EQ(gops.rates, (std::vector<double>{12.0, 20.0}));
    EXPECT_EQ(frames_csv(data, true),
              "frame,time_s,size_bytes,keyframe\r\n0,0.000000,1000,1\r\n1,0.500000,500,0\r\n2,1.000000,2000,1\r\n"
              "3,1.500000,500,0\r\n");
    EXPECT_EQ(seconds_csv(data, true), "start_s,end_s,kbps\r\n0.000000,1.000000,12.0\r\n1.000000,2.000000,20.0\r\n");
    EXPECT_EQ(seconds_csv(data, false), "start_s,end_s,kbps\r\n10.000000,11.000000,12.0\r\n11.000000,12.000000,20.0\r\n");
    EXPECT_EQ(gops_csv(data, false),
              "start_s,end_s,frames,bytes,kbps\r\n10.000000,11.000000,2,1500,12.0\r\n11.000000,12.000000,2,2500,20.0\r\n");
    EXPECT_EQ(frames_csv(data, false).substr(34, 23), "0,10.000000,1000,1\r\n1,1");
}

// A raw stream timed by --fps is read at that rate: ffprobe would take a raw
// 120 fps HEVC stream as 25 frames a second (its packets 0.04 s long, the
// bitrate 4.8 times too low). A video scaled to one size reads as any other:
// ffprobe has no -reinit_filter.
TEST(Bitrate, ReadsARawStreamAtItsFrameRate) {
    VideoInfo raw;
    raw.path = "clip.hevc";
    raw.format_name = "hevc";
    raw.timed_raw = true;
    raw.fps = 120;
    const std::vector<std::string> timed = bitrate_command("ffprobe", raw);
    const auto rate = std::find(timed.begin(), timed.end(), "-framerate");
    ASSERT_NE(rate, timed.end());
    EXPECT_EQ(*(rate + 1), "120.0");
    EXPECT_EQ(timed.back(), "clip.hevc");

    // Its packets' times are frames at that rate: ffprobe's raw HEVC parser
    // gives most of them 1 microsecond (HoneyBee's 4K stream: 550 of 600).
    PacketList packets(120, true);
    for (const char* duration : {"0.008333", "0.000001", "0.000001"}) {
        packets.add(std::string("pts_time=N/A|dts_time=N/A|duration_time=") + duration + "|size=1000|pos=0|flags=___");
    }
    const BitrateData data = packets.finish("clip.hevc");
    EXPECT_DOUBLE_EQ(data.duration(), 3.0 / 120);
    EXPECT_DOUBLE_EQ(data.times.back(), 2.0 / 120);
    // 5 s of it are 5 seconds, not 6 (the durations summed came to
    // 5.000000000000038).
    PacketList five_seconds(120, true);
    for (int frame = 0; frame < 600; ++frame) {
        five_seconds.add("pts_time=N/A|dts_time=N/A|duration_time=0.000001|size=1000|pos=0|flags=___");
    }
    EXPECT_EQ(second_bins(five_seconds.finish("clip.hevc"), true).starts.size(), 5U);

    VideoInfo scaled;
    scaled.path = "clip.mkv";
    scaled.scaled_to = std::pair(1920, 1080);
    const std::vector<std::string> plain = bitrate_command("ffprobe", scaled);
    EXPECT_EQ(std::find(plain.begin(), plain.end(), "-reinit_filter"), plain.end());
    EXPECT_EQ(std::find(plain.begin(), plain.end(), "-framerate"), plain.end());
}

struct Step {
    double start;
    double end;
    float value;
};

// A stepped graph's points from its steps, as step_points makes them.
BitratePlot stepped(const std::vector<Step>& steps) {
    BitratePlot plot{{}, {}, "Video bitrate (kb/s)", "kb/s"};
    for (const auto& [start, end, value] : steps) {
        plot.times.insert(plot.times.end(), {start, end});
        plot.values.insert(plot.values.end(), {value, value});
    }
    return plot;
}

// The values as text, NaN as "nan", for comparing lines with gaps.
std::vector<std::string> texts(const std::vector<float>& values) {
    std::vector<std::string> result;
    for (const float value : values) {
        result.push_back(std::isnan(value) ? "nan" : std::to_string(value));
    }
    return result;
}

// The Bitrate tab's Difference band: frame graphs pair frame i with frame i
// at the first's times, NaN past the second's last; stepped graphs subtract
// over time, NaN before or after either's steps, each step lasting until the
// next one starts (a group of pictures' end is a little off it either way).
TEST(Bitrate, SubtractsTwoGraphsFrameByFrameOrOverTime) {
    const float nan = std::nanf("");
    const BitratePlot first{{0.0, 0.5, 1.0}, {10, 20, 30}, "Frame size (kbit)", "kbit"};
    const BitratePlot second{{0.1, 0.6}, {4, 25}, "", ""};
    const BitratePlot frames = frame_difference_plot(first, second);
    EXPECT_EQ(frames.times, first.times);
    EXPECT_EQ(texts(frames.values), texts({6, -5, nan}));
    EXPECT_EQ(frames.axis_label, "Frame size (kbit)");

    // a's second step overlaps its third by 1 ms; b's first ends 1 ms before its second
    const BitratePlot a = stepped({{0, 1, 10}, {1, 2.001, 20}, {2, 3, 30}});
    const BitratePlot b = stepped({{0.5, 1.999, 4}, {2, 3, 5}});
    const BitratePlot difference = step_difference_plot(a, b);
    EXPECT_EQ(difference.times, (std::vector<double>{0, 0.5, 0.5, 1, 1, 2, 2, 3}));
    EXPECT_EQ(texts(difference.values), texts({nan, nan, 6, 6, 16, 16, 25, 25}));
    EXPECT_EQ(difference.value_label, "kb/s");

    // No time in common: no difference anywhere
    const BitratePlot later = stepped({{5, 6, 1}});
    EXPECT_TRUE(std::ranges::all_of(step_difference_plot(a, later).values, [](float v) { return std::isnan(v); }));
}

}  // namespace
}  // namespace vml
