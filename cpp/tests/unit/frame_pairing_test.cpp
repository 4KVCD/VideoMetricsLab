// A results JSON's frame_sync (frame_pairing.h): every result says how its
// frames were paired, in a status and a note in plain words.
#include "vml/frame_pairing.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace vml {
namespace {

// Two 10 s videos at 24 fps (240 frames each by their length), with PSNR
// scored on their first `pairs` frames.
ComparisonResult scored(int pairs) {
    ComparisonResult result;
    for (VideoInfo* video : {&result.source_info, &result.distorted_info}) {
        video->duration = 10.0;
        video->fps = 24.0;
    }
    FrameMetricResult psnr{"psnr", std::vector<int>(static_cast<std::size_t>(pairs)), {}, {}, {}};
    std::iota(psnr.frame.begin(), psnr.frame.end(), 0);
    psnr.time.assign(psnr.frame.size(), 0.0);
    psnr.values.assign(psnr.frame.size(), 40.0F);
    result.metric_results.add(std::move(psnr));
    return result;
}

TEST(FramePairing, EveryResultSaysHowItsFramesWerePairedAndWhatThatMeans) {
    const auto status = [](const ComparisonResult& result) {
        return frame_pairing_json(result).at("status").get<std::string>();
    };
    const auto note = [](const ComparisonResult& result) {
        return frame_pairing_json(result).at("note").get<std::string>();
    };
    EXPECT_EQ(status(scored(240)), "no_mismatch_seen");
    EXPECT_EQ(status(scored(239)), "no_mismatch_seen");  // a duration's rounding
    EXPECT_EQ(status(scored(120)), "not_checked");       // a range: no FramesShort was made
    ComparisonResult resolution_test = scored(240);
    resolution_test.resample_target = ResampleTarget{};
    EXPECT_EQ(status(resolution_test), "not_checked");

    ComparisonResult short_of = scored(235);
    short_of.frames_short = FramesShort{235, 240};
    EXPECT_EQ(status(short_of), "may_be_out_of_sync");
    EXPECT_NE(note(short_of).find("--frame-counts check"), std::string::npos);  // how to tell
    short_of.frames_short = FramesShort{235, 240, 235, 240};
    EXPECT_EQ(status(short_of), "out_of_sync");
    EXPECT_EQ(frame_pairing_json(short_of).at("test_frames"), 240);
    short_of.frames_short = FramesShort{235, 240, 235, 235};
    EXPECT_EQ(status(short_of), "in_sync");
    EXPECT_EQ(frames_short_text(*short_of.frames_short), "");  // no command-line note
}

// A range keeps whole-video frame numbers, so a range from the middle to
// the end reaches the videos' length: it is still only part of them, and
// its pairs are those from the range's start.
TEST(FramePairing, ARangeToTheEndIsNotCheckedAndCountsOnlyItsOwnPairs) {
    ComparisonResult range = scored(120);
    FrameMetricResult psnr = *range.metric_results.frame("psnr");
    for (int& frame : psnr.frame) {
        frame += 120;  // frames 120 to 239 of 240
    }
    psnr.provenance.parameters["range_frames"] = "[120, null]";
    range.metric_results = MetricResultSet();
    range.metric_results.add(std::move(psnr));
    const nlohmann::ordered_json sync = frame_pairing_json(range);
    EXPECT_EQ(sync.at("status"), "not_checked");
    EXPECT_EQ(sync.at("compared_pairs"), 120);
}

// The pairs always end with the video that has fewer frames, so of two
// videos of different lengths, in step or not depends on which lacks more of
// its length's frames. A reference with a remux's gaps (2,390 frames, length
// 2,398) and a test that keeps them but ends 2 frames early (2,388, length
// 2,396: whole runs allow 0.1 s) are in step; a test at a constant rate that
// fills the gaps (2,396) is not, nor one that lacks a frame more.
TEST(FramePairing, VideosOfDifferentLengthsAreInStepWhenTheOneWithFewerFramesLacksNoMore) {
    const auto short_of = [](long long source, long long source_length, long long test, long long test_length) {
        const long long pairs = std::min(source, test);
        return FramesShort{pairs, std::min(source_length, test_length), source, test, source_length, test_length};
    };
    const FramesShort keeps_the_gaps = short_of(2390, 2398, 2388, 2396);
    EXPECT_TRUE(keeps_the_gaps.in_step());
    EXPECT_EQ(frames_short_note(keeps_the_gaps).find("same length"), std::string::npos);
    EXPECT_TRUE(short_of(2388, 2396, 2390, 2398).in_step());   // the other way round
    EXPECT_FALSE(short_of(2390, 2398, 2396, 2396).in_step());  // the gaps filled
    const FramesShort lacks_more = short_of(2390, 2398, 2387, 2396);
    EXPECT_FALSE(lacks_more.in_step());
    EXPECT_EQ(frames_short_note(lacks_more).find("same length"), std::string::npos);
    // A verdict saved before the lengths were kept: judged as of one length.
    EXPECT_FALSE((FramesShort{2388, 2396, 2390, 2388}).in_step());
}

}  // namespace
}  // namespace vml
