// A range's frames (frame_range.h): its seconds counted as CVVDP counts
// them, the frames decoded around them for the metrics that look around, a
// subsampled run's first frame on its grid.
#include "vml/frame_range.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "vml/analysis_request.h"
#include "vml/options.h"

namespace vml {
namespace {

// Second 1001 at 24000/1001 fps starts at frame 24000 exactly, where the
// division in doubles may round either way: the range's first frame is the
// first frame CVVDP counts in that second, whatever the rounding.
TEST(FrameRange, ItsSecondsAreCvvdpsAtTheirEdges) {
    const double ntsc_film = 24000.0 / 1001.0;
    for (const double fps : {ntsc_film, 24.0, 25.0, 30000.0 / 1001.0, 60.0, 0.5, 3.0}) {
        for (const long long second : {1LL, 2LL, 59LL, 1001LL, 6000LL}) {
            const long long first = first_frame_of_second(second, fps);
            EXPECT_GE(second_of_frame(first, fps), second) << fps << " fps, second " << second;
            EXPECT_LT(second_of_frame(first - 1, fps), second) << fps << " fps, second " << second;
        }
    }
    EXPECT_EQ(first_frame_of_second(0, ntsc_film), 0);
    EXPECT_EQ(first_frame_of_second(10, 24.0), 240);
    EXPECT_EQ(first_frame_of_second(3, 0.5), 2);  // frame 2 is at 4 s: second 3 has none of its own
}

// The end of the ranges that reach a video's last frame: a 300-frame encode
// at 24000/1001 (12.5 s) ends in second 12, so 13; 288 frames (12.012 s)
// end in second 11, so 12. A second before it has a frame of the video, it
// none.
TEST(FrameRange, EndsAtTheSecondAfterTheLastFrames) {
    const double ntsc_film = 24000.0 / 1001.0;
    EXPECT_EQ(end_second_of(300, ntsc_film), 13);
    EXPECT_EQ(end_second_of(288, ntsc_film), 12);
    EXPECT_EQ(end_second_of(288, 24.0), 12);
    EXPECT_EQ(end_second_of(289, 24.0), 13);
    EXPECT_EQ(end_second_of(0, 24.0), 0);
    for (const double fps : {ntsc_film, 24.0, 0.5}) {
        for (const long long frames : {1LL, 287LL, 288LL, 300LL, 1441LL}) {
            const long long end = end_second_of(frames, fps);
            EXPECT_LT(first_frame_of_second(end - 1, fps), frames) << fps << " fps, " << frames << " frames";
            EXPECT_GE(first_frame_of_second(end, fps), frames) << fps << " fps, " << frames << " frames";
        }
    }
}

// The whole video; a range at 24 fps with and without the metrics that look
// around; the warm-up at 4 frames at least below 4 fps; subsampled.
TEST(FrameRange, AScoresRangeFramesGiveTheSecondsTheyCameFrom) {
    const double ntsc = 24000.0 / 1001.0;
    MetricProvenance provenance;
    EXPECT_EQ(scored_seconds(provenance, ntsc), std::nullopt);  // the whole videos
    for (const auto& [first, end] : {std::pair<long long, long long>{0, 13}, {20, 30}, {3599, 3600}}) {
        mark_range(provenance, frame_range(first, end, ntsc, 1, false));
        EXPECT_EQ(scored_seconds(provenance, ntsc), std::pair(first, end));
    }
    mark_range(provenance, frame_range(40, 0, ntsc, 1, false));  // to the end
    EXPECT_EQ(scored_seconds(provenance, ntsc), std::pair(40LL, 0LL));
    // A video that ended within the range's last second: the range's end.
    mark_range(provenance, frame_range(0, 13, ntsc, 1, false), 300);
    EXPECT_EQ(scored_seconds(provenance, ntsc), std::pair(0LL, 13LL));
}

TEST(FrameRange, DecodesTheFramesItsMetricsLookAt) {
    EXPECT_TRUE(frame_range(0, 0, 24.0, 1, true).whole());
    EXPECT_EQ(frame_range(0, 0, 24.0, 1, true), (FrameRange{0, std::nullopt, 0, std::nullopt}));
    EXPECT_EQ(frame_range(0, 30, 24.0, 1, true), (FrameRange{0, 720, 0, 721}));
    EXPECT_EQ(frame_range(10, 30, 24.0, 1, true), (FrameRange{240, 720, 216, 721}));
    EXPECT_EQ(frame_range(10, 30, 24.0, 1, false), (FrameRange{240, 720, 240, 720}));
    EXPECT_EQ(frame_range(10, 0, 24.0, 1, true), (FrameRange{240, std::nullopt, 216, std::nullopt}));
    EXPECT_EQ(frame_range(10, 30, 3.0, 1, true), (FrameRange{30, 90, 26, 91}));
    EXPECT_EQ(frame_range(1, 2, 1.0, 1, true), (FrameRange{1, 2, 0, 3}));
    EXPECT_EQ(frame_range(10, 30, 24.0, 5, true), (FrameRange{240, 720, 215, 721}));
    EXPECT_EQ(frame_range(10, 30, 24.0, 7, false), (FrameRange{240, 720, 238, 720}));
}

// A range of frames compares exactly its frames, decoding a second before
// for the metrics that look back; CVVDP, which scores whole seconds, takes
// the nearest ones: at least one, and none past the shorter video's end.
TEST(FrameRange, ARangeOfFramesIsExactAndCvvdpsTheNearestSeconds) {
    const ComparedRange frames{100, 220, true};  // frames 100 to 219
    EXPECT_EQ(compared_frames(frames, 50.0), std::pair(100LL, std::optional(220LL)));
    EXPECT_EQ(compared_frames(ComparedRange{100, 0, true}, 50.0), std::pair(100LL, std::optional<long long>()));
    EXPECT_EQ(compared_frames(ComparedRange{2, 4, false}, 50.0), std::pair(100LL, std::optional(200LL)));
    EXPECT_EQ(frame_range(frames, 50.0, 1, true), (FrameRange{100, 220, 50, 221}));
    EXPECT_EQ(frame_range(frames, 50.0, 1, false), (FrameRange{100, 220, 100, 220}));

    EXPECT_EQ(cvvdp_seconds(frames, 50.0, 500), (ComparedRange{2, 4, false}));  // 2.0 to 4.4 s
    EXPECT_EQ(cvvdp_seconds(ComparedRange{100, 110, true}, 50.0, 500), (ComparedRange{2, 3, false}));
    EXPECT_EQ(cvvdp_seconds(ComparedRange{0, 10, true}, 50.0, 500), (ComparedRange{0, 1, false}));
    // Under a second, rounded to no second: the second with its middle frame.
    EXPECT_EQ(cvvdp_seconds(ComparedRange{12, 24, true}, 24.0, 480), (ComparedRange{0, 1, false}));
    EXPECT_EQ(cvvdp_seconds(ComparedRange{400, 0, true}, 50.0, 500), (ComparedRange{8, 0, false}));
    EXPECT_EQ(cvvdp_seconds(ComparedRange{400, 500, true}, 50.0, 450), (ComparedRange{8, 9, false}));
    EXPECT_EQ(cvvdp_seconds(ComparedRange{490, 500, true}, 50.0, 450), (ComparedRange{8, 9, false}));
    EXPECT_EQ(cvvdp_seconds(ComparedRange{0, 0, true}, 50.0, 500), ComparedRange{});
    EXPECT_EQ(cvvdp_seconds(ComparedRange{3, 7, false}, 50.0, 500), (ComparedRange{3, 7, false}));
    // At 30000/1001, frame 45 is 1.5015 s in and frame 44 1.468 s.
    const double ntsc = 30000.0 / 1001.0;
    EXPECT_EQ(cvvdp_seconds(ComparedRange{45, 300, true}, ntsc, 900), (ComparedRange{2, 10, false}));
    EXPECT_EQ(cvvdp_seconds(ComparedRange{44, 300, true}, ntsc, 900), (ComparedRange{1, 10, false}));
}

// A time to the millisecond (the Options panel's Range, hh:mm:ss.zzz) covers
// the frames whose times are in it, as a range of whole seconds does; CVVDP
// scores the nearest whole seconds, as of a range of frames.
TEST(FrameRange, ARangeOfMillisecondsIsTheFramesInIt) {
    const ComparedRange ms{2500, 4250, false, true};
    // At 24 fps: frame 60 is at 2.5 s, frame 102 at 4.25 s (past the range).
    EXPECT_EQ(in_frames_at(ms, 24.0), (ComparedRange{60, 102, true}));
    EXPECT_EQ(compared_frames(ms, 24.0), std::pair(60LL, std::optional(102LL)));
    // Between frames: from the first after it (2.51 s: frame 61, at 2.5417 s).
    EXPECT_EQ(compared_frames(ComparedRange{2510, 0, false, true}, 24.0),
              std::pair(61LL, std::optional<long long>()));
    // At 24000/1001 fps, frame 1438 is at 59.9766 s, 1439 at 60.0183 s and
    // 1440 at 60.06 s: 59.977 to 60.018 s has no frame.
    const double film = 24000.0 / 1001.0;
    EXPECT_EQ(compared_frames(ComparedRange{60000, 60020, false, true}, film), std::pair(1439LL, std::optional(1440LL)));
    EXPECT_EQ(compared_frames(ComparedRange{59976, 60019, false, true}, film), std::pair(1438LL, std::optional(1440LL)));
    EXPECT_EQ(compared_frames(ComparedRange{59977, 60018, false, true}, film), std::pair(1439LL, std::optional(1439LL)));
    // Whole seconds in milliseconds are the seconds' frames.
    for (const double fps : {24.0, film, 25.0, 30000.0 / 1001.0, 50.0, 60000.0 / 1001.0}) {
        EXPECT_EQ(compared_frames(ComparedRange{7000, 93000, false, true}, fps),
                  compared_frames(ComparedRange{7, 93, false}, fps))
            << fps;
    }
    // An end under a frame in is frame 1, never "to the end".
    EXPECT_EQ(in_frames_at(ComparedRange{0, 1, false, true}, 24.0), (ComparedRange{0, 1, true}));
    // Decoded and scored as the frames they are; CVVDP the nearest seconds.
    EXPECT_EQ(frame_range(ms, 24.0, 1, true), frame_range(ComparedRange{60, 102, true}, 24.0, 1, true));
    EXPECT_EQ(cvvdp_seconds(ms, 24.0, 480), (ComparedRange{3, 4, false}));  // 2.5 to 4.25 s
    EXPECT_FALSE(ms.in_seconds());
    EXPECT_TRUE((ComparedRange{2, 4, false}).in_seconds());
    // Frames win over milliseconds, should both be set.
    EXPECT_EQ(in_frames_at(ComparedRange{100, 200, true, true}, 24.0), (ComparedRange{100, 200, true, true}));

    EXPECT_EQ(range_time_text(62500), "0:01:02.500");
    EXPECT_EQ(range_time_text(3723004), "1:02:03.004");
    EXPECT_EQ(range_time_text(5576279), "1:32:56.279");
}

// A range of time is whole seconds where both its ends are, so that it is the
// range (and saved scores' identity) it has always been; milliseconds
// otherwise, with an identity of their own.
TEST(FrameRange, ATimeRangeIsWholeSecondsWhereItCanBe) {
    VmafOptions options;
    options.set_time_range(10000, 20000);
    EXPECT_EQ(options.compared_range(), (ComparedRange{10, 20, false}));
    options.set_time_range(10500, 20000);
    EXPECT_EQ(options.compared_range(), (ComparedRange{10500, 20000, false, true}));
    EXPECT_EQ(options.range_time_ms(options.range_end), 20000);
    options.set_time_range(0, 0);
    EXPECT_TRUE(options.compared_range().whole());
    EXPECT_FALSE(options.range_in_milliseconds);

    const VideoInfo video = [] {
        VideoInfo info;
        info.fps = 24.0;
        info.duration = 60.0;
        info.width = 1920;
        info.height = 1080;
        return info;
    }();
    const auto identity = [&video](long long start_ms, long long end_ms) {
        VmafOptions ranged;
        ranged.set_time_range(start_ms, end_ms);
        return analysis_request(ranged, {"psnr"}, {}, CvvdpSettings{}, video, video).recipe.identity();
    };
    VmafOptions seconds;
    seconds.range_start = 10;
    seconds.range_end = 20;
    EXPECT_EQ(identity(10000, 20000),
              analysis_request(seconds, {"psnr"}, {}, CvvdpSettings{}, video, video).recipe.identity());
    const nlohmann::json ms = identity(10500, 20000);
    EXPECT_EQ(ms["range_unit"], "milliseconds");
    EXPECT_NE(ms, identity(10000, 20000));
    EXPECT_FALSE(identity(10000, 20000).contains("range_unit"));
}

// The Range unit switched (Options > Range, Time or Frames): the same frames
// in the other unit, and where it is switched back with no edit, the range
// as it was typed rather than rounded to frames' times.
TEST(FrameRange, SwitchingTheRangeUnitKeepsTheRange) {
    const double film = 24000.0 / 1001.0;
    // Time to frames and back: whole seconds at 24 fps are whole seconds again.
    EXPECT_EQ(range_in_unit(ComparedRange{10, 20, false}, true, 24.0), (ComparedRange{240, 480, true}));
    EXPECT_EQ(range_in_unit(ComparedRange{240, 480, true}, false, 24.0), (ComparedRange{10, 20, false}));
    // At 24000/1001 a second's first frame is past its start: 30 s is frame
    // 720, at 30.030 s.
    const ComparedRange seconds{30, 60, false};
    const std::optional<ComparedRange> frames = range_in_unit(seconds, true, film);
    ASSERT_TRUE(frames);
    EXPECT_EQ(*frames, (ComparedRange{720, 1439, true}));
    const std::optional<ComparedRange> times = range_in_unit(*frames, false, film);
    ASSERT_TRUE(times);
    EXPECT_EQ(*times, (ComparedRange{30030, 60018, false, true}));
    EXPECT_EQ(range_in_unit(*times, true, film), frames);  // the same frames
    // Every frame's time reads back as that frame, at the usual rates.
    for (const double fps : {film, 24.0, 25.0, 30000.0 / 1001.0, 50.0, 60000.0 / 1001.0, 120.0}) {
        for (long long frame = 1; frame < 20000; frame += 7) {
            const std::optional<ComparedRange> time = range_in_unit(ComparedRange{frame, frame + 5, true}, false, fps);
            ASSERT_TRUE(time) << fps;
            EXPECT_EQ(range_in_unit(*time, true, fps), (ComparedRange{frame, frame + 5, true})) << fps << " " << frame;
        }
    }
    // To the end stays to the end; the whole videos stay whole; a unit's own range is itself.
    EXPECT_EQ(range_in_unit(ComparedRange{100, 0, true}, false, 24.0), (ComparedRange{4166, 0, false, true}));
    EXPECT_EQ(range_in_unit(ComparedRange{}, true, film), (ComparedRange{0, 0, true}));
    EXPECT_EQ(range_in_unit(ComparedRange{0, 0, true}, false, film), ComparedRange{});
    EXPECT_EQ(range_in_unit(ComparedRange{2500, 4000, false, true}, false, 0.0),
              (ComparedRange{2500, 4000, false, true}));
    // No frame rate (no reference yet): no other unit.
    EXPECT_FALSE(range_in_unit(seconds, true, 0.0));

    // Switched to frames and back with no edit: the range as typed.
    const ComparedRange to_frames = switch_range_unit(seconds, std::nullopt, true, film);
    EXPECT_EQ(to_frames, (ComparedRange{720, 1439, true}));
    EXPECT_EQ(switch_range_unit(to_frames, seconds, false, film), seconds);
    const ComparedRange typed_frames{100, 220, true};
    const ComparedRange to_time = switch_range_unit(typed_frames, std::nullopt, false, film);
    EXPECT_EQ(switch_range_unit(to_time, typed_frames, true, film), typed_frames);
    // Edited in between: the edited range in the other unit, not the old one.
    EXPECT_EQ(switch_range_unit(ComparedRange{720, 1500, true}, seconds, false, film),
              range_in_unit(ComparedRange{720, 1500, true}, false, film).value());
    // Without a frame rate: the whole videos, and the range typed again on the way back.
    EXPECT_EQ(switch_range_unit(seconds, std::nullopt, true, 0.0), (ComparedRange{0, 0, true}));
    EXPECT_EQ(switch_range_unit(ComparedRange{0, 0, true}, seconds, false, 0.0), seconds);
    EXPECT_EQ(switch_range_unit(ComparedRange{5, 9, true}, seconds, false, 0.0), ComparedRange{});
}

// Scores loaded from a file tell their range by their frames: on whole
// seconds' edges a range of seconds, else of frames.
TEST(FrameRange, AScoresFramesSayWhetherItWasARangeOfFrames) {
    MetricProvenance provenance;
    EXPECT_EQ(scored_frames(provenance, 50.0), std::nullopt);
    mark_range(provenance, frame_range(ComparedRange{100, 220, true}, 50.0, 1, true));
    EXPECT_EQ(scored_frames(provenance, 50.0), (ComparedRange{100, 220, true}));
    mark_range(provenance, frame_range(ComparedRange{101, 0, true}, 50.0, 1, true));
    EXPECT_EQ(scored_frames(provenance, 50.0), (ComparedRange{101, 0, true}));
    mark_range(provenance, frame_range(2, 4, 50.0, 1, true));
    EXPECT_EQ(scored_frames(provenance, 50.0), (ComparedRange{2, 4, false}));
}

}  // namespace
}  // namespace vml
