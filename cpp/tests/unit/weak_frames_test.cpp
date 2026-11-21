#include "vml/weak_frames.h"

#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
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
// Which condition made each segment, and each condition's frames: a long dip
// in VMAF and one frame of SSIMULACRA2 below its threshold, with a shortest
// segment of 6. The SSIMULACRA2 frame is in no segment, and the counts say
// so (the scores in the segments' table alone didn't tell why).
TEST(WeakFrames, EachConditionsFramesAreCounted) {
    MetricResultSet results;
    std::vector<std::pair<int, float>> dip;
    for (int frame = 10; frame < 20; ++frame) {
        dip.emplace_back(frame, 80.0F);
    }
    results.add(frame_metric("vmaf", every(1, 59), 95.0F, dip));
    results.add(frame_metric("ssimulacra2", every(1, 59), 80.0F, {{40, 55.0F}, {15, 50.0F}}));
    SegmentRules rules{{{"vmaf", '<', 85.0}, {"ssimulacra2", '<', 60.0}}};
    rules.min_frames = 6;
    const std::vector<WeakSegment> segments = weak_segments(results, rules, fps, 59);
    ASSERT_EQ(segments.size(), 1u);
    EXPECT_EQ(segments[0].first_frame, 10);
    EXPECT_EQ(segments[0].condition_frames, (std::vector<long long>{10, 1}));  // frame 15 is weak by both
    const std::vector<ConditionCount> counts = condition_counts(results, rules, 59, segments);
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts[0].holds, 10);
    EXPECT_EQ(counts[0].in_segments, 10);
    EXPECT_EQ(counts[1].holds, 2);        // frames 15 and 40
    EXPECT_EQ(counts[1].in_segments, 1);  // frame 40 alone is shorter than 6
    // Every condition required: no frame holds both but 15, a run of one.
    rules.all = true;
    const std::vector<WeakSegment> none = weak_segments(results, rules, fps, 59);
    EXPECT_TRUE(none.empty());
    EXPECT_EQ(condition_counts(results, rules, 59, none)[0].in_segments, 0);
    // A condition on a metric without scores holds nowhere.
    rules = SegmentRules{{{"vmaf", '<', 85.0}, {"cvvdp", '<', 9.0}}};
    const std::vector<WeakSegment> some = weak_segments(results, rules, fps, 59);
    ASSERT_EQ(some.size(), 1u);
    EXPECT_EQ(some[0].condition_frames, (std::vector<long long>{10, 0}));
    EXPECT_EQ(condition_counts(results, rules, 59, some)[1].holds, 0);
}

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

    // The rules Settings keeps: an unreadable condition, or one on a metric
    // the app does not have, left out; a shortest segment below 1 and a gap
    // below 0 taken as 1 and 0.
    const SegmentRules saved =
        saved_segment_rules({"vmaf<90", "vmaf<", "nometric<5", "butteraugli>2.5"}, true, 0, -4);
    EXPECT_EQ(condition_texts(saved.conditions), (std::vector<std::string>{"vmaf<90", "butteraugli>2.5"}));
    EXPECT_TRUE(saved.all);
    EXPECT_EQ(saved.min_frames, 1);
    EXPECT_EQ(saved.merge_frames, 0);

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

// The report of several videos: the rules once, then each video's segments
// with its name, files and counts; as CSV, segments_csv's rows led by the
// video and its reference (a name with a comma quoted).
TEST(WeakFrames, ReportHoldsEveryVideosSegments) {
    ComparisonResult first;
    first.fps = fps;
    first.metric_results.add(frame_metric("vmaf", every(1, 19), 95.0F, {{3, 80.0F}, {4, 70.0F}, {12, 85.0F}}));
    ComparisonResult second;
    second.fps = fps;
    second.metric_results.add(frame_metric("vmaf", every(1, 19), 95.0F));
    SegmentRules rules{{*parse_weak_condition("vmaf<90")}};
    rules.ranges = {{0, 5}};  // a dialog's range is not the report's
    const std::vector<SegmentsReportVideo> videos{{"a, b", "D:/a, b.mkv", "D:/ref.mkv", &first},
                                                  {"c", "D:/c.mkv", "D:/ref.mkv", &second}};
    const nlohmann::ordered_json report = segments_report_json(videos, rules);
    EXPECT_EQ(report["conditions"], nlohmann::ordered_json::array({"vmaf<90"}));
    EXPECT_EQ(report["combine"], "any");
    EXPECT_FALSE(report.contains("ranges"));
    ASSERT_EQ(report["videos"].size(), 2U);
    const nlohmann::ordered_json& a = report["videos"][0];
    EXPECT_EQ(a["name"], "a, b");
    EXPECT_EQ(a["path"], "D:/a, b.mkv");
    EXPECT_EQ(a["reference"], "D:/ref.mkv");
    EXPECT_EQ(a["segment_count"], 2);
    EXPECT_EQ(a["weak_frames"], 3);
    EXPECT_EQ(a["segments"][1]["first_frame"], 12);
    EXPECT_EQ(report["videos"][1]["segment_count"], 0);
    EXPECT_TRUE(report["videos"][1]["segments"].empty());

    EXPECT_EQ(segments_report_csv(videos, rules),
              "video,reference,first_frame,last_frame,start_time,end_time,frames,vmaf_min,vmaf_max,vmaf_mean\n"
              "\"a, b\",D:/ref.mkv,3,4,0.3,0.5,2,70.0,80.0,75.0\n"
              "\"a, b\",D:/ref.mkv,12,12,1.2,1.3,1,85.0,85.0,85.0\n");

    // Issue #132: every condition must hold, one on a metric no video has:
    // it is left out (and said), not taken as weak nowhere.
    SegmentRules every{{*parse_weak_condition("vmaf<90"), *parse_weak_condition("cvvdp<9")}};
    every.all = true;
    const nlohmann::ordered_json strict = segments_report_json(videos, every);
    EXPECT_EQ(strict["videos"][0]["segment_count"], 2);
    EXPECT_EQ(strict["videos"][0]["left_out"], nlohmann::ordered_json::array({"cvvdp<9"}));
}

// Weak segment profiles: Default from the weak_segment_* settings, then the
// others in their saved order (one without a name, or with Default's or
// another's in any case, left out); looked up in any case; the one in use
// Default where its name is none's; kept back in the same places.
TEST(WeakFrames, ProfilesAreKeptInSettings) {
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<90"};
    settings.weak_segment_profiles_json =
        R"([{"name": " Animation ", "conditions": ["vmaf<93", "ssimulacra2<75"], "min_frames": 2},)"
        R"( {"name": "default", "conditions": ["psnr<30"]}, {"conditions": ["psnr<30"]},)"
        R"( {"name": "ANIMATION", "conditions": ["psnr<30"]}, {"name": "Film", "all": true, "merge_frames": -3}])";
    std::vector<std::string> names;
    for (const SegmentProfile& profile : segment_profiles(settings)) {
        names.push_back(profile.name);
    }
    EXPECT_EQ(names, (std::vector<std::string>{"", "Animation", "Film"}));
    const std::optional<SegmentProfile> animation = segment_profile_named(settings, "animation");
    ASSERT_TRUE(animation);
    EXPECT_EQ(condition_texts(animation->rules.conditions), (std::vector<std::string>{"vmaf<93", "ssimulacra2<75"}));
    EXPECT_EQ(animation->rules.min_frames, 2);
    EXPECT_EQ(segment_profile_named(settings, "Film")->rules.merge_frames, 0);
    EXPECT_FALSE(segment_profile_named(settings, "Anime"));
    EXPECT_EQ(segment_profile_named(settings, "Default")->name, "");

    EXPECT_EQ(segment_profile_in_use(settings).name, "");
    settings.weak_segment_profile = "film";
    EXPECT_EQ(segment_profile_in_use(settings).name, "Film");
    settings.weak_segment_profile = "gone";
    EXPECT_EQ(segment_profile_in_use(settings).name, "");

    store_segment_profile(settings, "Film", SegmentRules{{*parse_weak_condition("ssimulacra2<60")}});
    store_segment_profile(settings, "HDR", SegmentRules{{*parse_weak_condition("cvvdp<9")}});
    store_segment_profile(settings, "", SegmentRules{{*parse_weak_condition("vmaf<88")}});
    EXPECT_EQ(settings.weak_segment_conditions, std::vector<std::string>{"vmaf<88"});
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Film")->rules.conditions),
              std::vector<std::string>{"ssimulacra2<60"});
    EXPECT_EQ(segment_profiles(settings).size(), 4U);  // Default, Animation, Film, HDR

    EXPECT_TRUE(segment_profile_name_free(segment_profiles(settings), "Grain"));
    for (const char* taken : {"", "  ", "Default", "hdr", " Animation", "Film, 35mm"}) {
        EXPECT_FALSE(segment_profile_name_free(segment_profiles(settings), taken)) << taken;
    }
}

// The presets: each with a key of its own, conditions on metrics the app
// has, each the way its metric's direction reads them (as the window's
// editor would save them), within the metric's scale, and options a profile
// can hold; found by key or name in any case.
TEST(WeakFrames, PresetsAreProfilesTheAppCanUse) {
    std::vector<std::string_view> keys;
    for (const SegmentPreset& preset : segment_presets()) {
        SCOPED_TRACE(std::string(preset.key));
        EXPECT_EQ(std::count(keys.begin(), keys.end(), preset.key), 0);
        keys.push_back(preset.key);
        ASSERT_FALSE(preset.rules.conditions.empty());
        for (const WeakCondition& condition : preset.rules.conditions) {
            const MetricDefinition* metric = find_metric(condition.key);
            ASSERT_NE(metric, nullptr) << condition.key;
            EXPECT_EQ(condition.comparison, metric->direction == MetricDirection::lower_is_better ? '>' : '<');
            EXPECT_GT(condition.threshold, 0.0);
            EXPECT_LE(condition.threshold, 100.0);
        }
        EXPECT_GE(preset.rules.min_frames, 1);
        EXPECT_GE(preset.rules.merge_frames, 0);
        // Kept as a profile and read back unchanged.
        Settings settings;
        store_segment_profile(settings, "P", preset.rules);
        const SegmentRules back = segment_profile_named(settings, "P")->rules;
        EXPECT_EQ(condition_texts(back.conditions), condition_texts(preset.rules.conditions));
        EXPECT_EQ(back.min_frames, preset.rules.min_frames);
        EXPECT_EQ(back.merge_frames, preset.rules.merge_frames);
    }
    EXPECT_EQ(keys.size(), 6U);
    ASSERT_NE(segment_preset(" Grainy Film "), nullptr);
    EXPECT_EQ(segment_preset(" Grainy Film ")->key, "grainy-film");
    EXPECT_EQ(segment_preset("HDR")->key, "hdr");
    EXPECT_EQ(segment_preset("anime"), nullptr);
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
