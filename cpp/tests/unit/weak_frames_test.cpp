#include "vml/weak_frames.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace vml {
namespace {

constexpr double fps = 10.0;

// A frame metric scored on `frames`, `score` everywhere but where `at` says.
FrameMetricResult frame_metric(const std::string& key, const std::vector<int>& frames, float score,
                               const std::vector<std::pair<int, float>>& at = {}) {
    FrameMetricResult metric;
    metric.key = key;
    for (const int frame : frames) {
        metric.frame.push_back(frame);
        metric.time.push_back(frame / fps);
        float value = score;
        for (const auto& [where, given] : at) {
            if (where == frame) {
                value = given;
            }
        }
        metric.values.push_back(value);
    }
    return metric;
}

std::vector<int> every(int step, int last) {
    std::vector<int> frames;
    for (int frame = 0; frame <= last; frame += step) {
        frames.push_back(frame);
    }
    return frames;
}

ScoreSamples samples_of(const FrameMetricResult& metric) { return {metric.frame, metric.time, metric.values, false}; }

std::vector<std::pair<int, int>> spans(const std::vector<WeakSegment>& segments) {
    std::vector<std::pair<int, int>> found;
    for (const WeakSegment& segment : segments) {
        found.emplace_back(segment.first_frame, segment.last_frame);
    }
    return found;
}

// One entry per dip, the dip's worst frame, worst first: a run of bad
// frames and a bad frame within 1 s of a worse one are one dip each; NaN
// frames never count; ranges and the metric's direction are respected.
TEST(WeakFrames, WorstFramesGivesEachDipsWorstFrame) {
    const FrameMetricResult vmaf = frame_metric("vmaf", every(1, 59), 95.0F,
                                                {{10, 70.0F}, {11, 50.0F}, {12, 40.0F}, {13, 50.0F}, {14, 70.0F},
                                                 {30, NAN}, {40, 60.0F}, {45, 70.0F}, {58, 80.0F}});
    const auto frames_of = [](const std::vector<WorstFrame>& worst) {
        std::vector<int> frames;
        for (const WorstFrame& frame : worst) {
            frames.push_back(frame.frame);
        }
        return frames;
    };
    // Frame 45 is within 1 s of 40, 58 is not of 45; every frame scoring 95 is
    // within 1 s of a worse one.
    const std::vector<WorstFrame> worst = worst_frames(samples_of(vmaf), MetricDirection::higher_is_better, 10);
    EXPECT_EQ(frames_of(worst), (std::vector<int>{12, 40, 58}));
    EXPECT_DOUBLE_EQ(worst[0].score, 40.0);
    EXPECT_DOUBLE_EQ(worst[0].time, 1.2);
    EXPECT_EQ(frames_of(worst_frames(samples_of(vmaf), MetricDirection::higher_is_better, 2)),
              (std::vector<int>{12, 40}));
    const std::vector<FrameInterval> late{{30, 59}};
    EXPECT_EQ(frames_of(worst_frames(samples_of(vmaf), MetricDirection::higher_is_better, 10, late)),
              (std::vector<int>{40, 58}));
    // Butteraugli's worst frames are its highest.
    const FrameMetricResult butteraugli = frame_metric("butteraugli", every(1, 59), 1.0F, {{20, 5.0F}, {50, 3.0F}});
    EXPECT_EQ(frames_of(worst_frames(samples_of(butteraugli), MetricDirection::lower_is_better, 2)),
              (std::vector<int>{20, 50}));
}

// Weak frames: a frame metric's own, a subsample's frames between two weak
// samples, CVVDP's whole second; combined with any or all, merged across
// short gaps, short segments left out; each with its metrics' scores.
TEST(WeakFrames, WeakSegmentsFollowTheRules) {
    MetricResultSet results;
    results.add(frame_metric(
        "vmaf", every(1, 59), 95.0F,
        {{10, 80.0F}, {11, 80.0F}, {12, 40.0F}, {13, 80.0F}, {14, 80.0F}, {17, 85.0F}, {18, 85.0F}}));
    results.add(frame_metric("psnr", every(5, 59), 45.0F, {{10, 25.0F}, {15, 20.0F}}));  // a subsample
    SequenceMetricResult cvvdp;
    cvvdp.key = "cvvdp";
    cvvdp.frame = every(10, 59);
    for (const int frame : cvvdp.frame) {
        cvvdp.time.push_back(frame / fps);
        cvvdp.values.push_back(frame == 10 || frame == 50 ? 8.0F : 9.8F);
    }
    results.add(cvvdp);
    const WeakCondition vmaf_below{"vmaf", '<', 90.0};
    const WeakCondition psnr_below{"psnr", '<', 30.0};
    const auto segments_for = [&results](SegmentRules rules) { return weak_segments(results, rules, fps, 59); };

    EXPECT_EQ(spans(segments_for({{vmaf_below}})), (std::vector<std::pair<int, int>>{{10, 14}, {17, 18}}));
    SegmentRules merged{{vmaf_below}};
    merged.merge_frames = 2;
    EXPECT_EQ(spans(segments_for(merged)), (std::vector<std::pair<int, int>>{{10, 18}}));
    SegmentRules long_only{{vmaf_below}};
    long_only.min_frames = 3;
    EXPECT_EQ(spans(segments_for(long_only)), (std::vector<std::pair<int, int>>{{10, 14}}));
    // PSNR's samples at 10 and 15 are weak, so 11-14 are too.
    EXPECT_EQ(spans(segments_for({{vmaf_below, psnr_below}})), (std::vector<std::pair<int, int>>{{10, 15}, {17, 18}}));
    SegmentRules both{{vmaf_below, psnr_below}};
    both.all = true;
    const std::vector<WeakSegment> all = segments_for(both);
    ASSERT_EQ(spans(all), (std::vector<std::pair<int, int>>{{10, 14}}));
    EXPECT_DOUBLE_EQ(all[0].start_time, 1.0);
    EXPECT_DOUBLE_EQ(all[0].end_time, 1.5);
    ASSERT_EQ(all[0].metrics.size(), 2U);
    EXPECT_EQ(all[0].metrics[0].key, "vmaf");
    EXPECT_DOUBLE_EQ(*all[0].metrics[0].min, 40.0);
    EXPECT_DOUBLE_EQ(*all[0].metrics[0].max, 80.0);
    EXPECT_DOUBLE_EQ(*all[0].metrics[0].mean, 72.0);
    EXPECT_DOUBLE_EQ(*all[0].metrics[1].min, 25.0);  // PSNR's one sample inside
    // CVVDP's weak seconds cover their frames, the last one to the run's end.
    EXPECT_EQ(spans(segments_for({{{"cvvdp", '<', 9.0}}})), (std::vector<std::pair<int, int>>{{10, 19}, {50, 59}}));
    SegmentRules ranged{{{"cvvdp", '<', 9.0}}};
    ranged.ranges = {{15, 52}};
    EXPECT_EQ(spans(segments_for(ranged)), (std::vector<std::pair<int, int>>{{15, 19}, {50, 52}}));
}

// The two exports, and the conditions they are read from.
TEST(WeakFrames, ExportsAndConditionsRead) {
    SegmentRules rules{{*parse_weak_condition("vmaf<90")}};
    rules.min_frames = 2;
    const std::vector<WeakSegment> segments{{10, 14, 1.0, 1.5, {{"vmaf", 40.0, 80.0, 72.0}}},
                                            {30, 31, 3.0, 3.2, {{"vmaf", 85.5, 86.0, std::nullopt}}}};
    EXPECT_EQ(segments_csv(segments, rules),
              "first_frame,last_frame,start_time,end_time,frames,vmaf_min,vmaf_max,vmaf_mean\n"
              "10,14,1.0,1.5,5,40.0,80.0,72.0\n"
              "30,31,3.0,3.2,2,85.5,86.0,\n");
    EXPECT_EQ(segments_json(segments, rules).dump(),
              R"({"conditions":["vmaf<90"],"combine":"any","min_frames":2,"merge_frames":0,"ranges":[],)"
              R"("segments":[{"first_frame":10,"last_frame":14,"start_time":1.0,"end_time":1.5,"frames":5,)"
              R"("metrics":{"vmaf":{"min":40.0,"max":80.0,"mean":72.0}}},)"
              R"({"first_frame":30,"last_frame":31,"start_time":3.0,"end_time":3.2,"frames":2,)"
              R"("metrics":{"vmaf":{"min":85.5,"max":86.0,"mean":null}}}]})");

    // Their names: the conditions' metrics, each once, in their order.
    EXPECT_EQ(segments_file_suffix(rules), "vmaf.segments");
    const SegmentRules several{{*parse_weak_condition("ssimulacra2<70"), *parse_weak_condition("vmaf<90"),
                                *parse_weak_condition("ssimulacra2<50"), *parse_weak_condition("vmaf_neg<88")}};
    EXPECT_EQ(segments_file_suffix(several), "ssimulacra2+vmaf+vmaf_neg.segments");

    const std::optional<WeakCondition> above = parse_weak_condition("butteraugli>2.5");
    ASSERT_TRUE(above);
    EXPECT_EQ(above->key, "butteraugli");
    EXPECT_TRUE(above->is_weak(3.0));
    EXPECT_FALSE(above->is_weak(2.5));
    EXPECT_FALSE(above->is_weak(NAN));
    EXPECT_EQ(above->text(), "butteraugli>2.5");
    for (const char* wrong : {"vmaf", "vmaf<", "<90", "vmaf<90x", "vmaf=90", "vmaf<inf"}) {
        EXPECT_FALSE(parse_weak_condition(wrong)) << wrong;
    }
}

// The window's helpers: a frame's score (a per-frame metric only at its
// own frames, a per-second one over its second), the run's last frame
// (the count compared, though the last sample is earlier), and the
// condition the Weak segments dialog starts from.
TEST(WeakFrames, ScoreAtLastFrameAndDefaultCondition) {
    const FrameMetricResult vmaf = frame_metric("vmaf", every(2, 8), 90.0F, {{4, NAN}});
    const ScoreSamples frames = samples_of(vmaf);
    EXPECT_EQ(score_at(frames, 2), 90.0);
    EXPECT_FALSE(score_at(frames, 3));  // not sampled
    EXPECT_FALSE(score_at(frames, 4));  // NaN
    EXPECT_FALSE(score_at(frames, -1));
    ScoreSamples seconds = frames;
    seconds.per_second = true;
    EXPECT_EQ(score_at(seconds, 3), 90.0);

    ComparisonResult result;
    result.metric_results.add(vmaf);
    result.compared_frame_count = 10;
    EXPECT_EQ(last_compared_frame(result), 9);
    result.compared_frame_count = 0;
    EXPECT_EQ(last_compared_frame(result), 8);

    EXPECT_EQ(default_weak_condition(metric_definition("vmaf")).text(), "vmaf<85");
    EXPECT_EQ(default_weak_condition(metric_definition("butteraugli")).text(), "butteraugli>2");
    EXPECT_EQ(default_weak_condition(metric_definition("cvvdp")).text(), "cvvdp<9");
}

}  // namespace
}  // namespace vml
