// A range's frames (frame_range.h): its seconds counted as CVVDP counts
// them, the frames decoded around them for the metrics that look around, a
// subsampled run's first frame on its grid.
#include "vml/frame_range.h"

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace vml
