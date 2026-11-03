// A picture's number from its timestamp (range_seek.h): what a seek's correctness
// rests on, in the timestamps real containers give.
#include "vml/range_seek.h"

#include <gtest/gtest.h>

#include "vml/frame_range.h"

namespace vml {

namespace {

constexpr Rational milliseconds{1, 1000};
constexpr Rational film{24000, 1001};

// A Matroska file's rounded timestamp of picture `number`, at the film rate.
long long matroska_pts(long long number, long long first) {
    return first + (number * 1001 + 12) / 24;
}

}  // namespace

TEST(RangeSeek, NumbersRoundedTimestampsAcrossAFilm) {
    for (const long long number : {0LL, 1LL, 24LL, 14400LL, 151918LL}) {
        EXPECT_EQ(picture_number_at(matroska_pts(number, 42), 42, milliseconds, film), number);
    }
}

TEST(RangeSeek, RefusesATimestampBetweenPictures) {
    EXPECT_EQ(picture_number_at(42 + 20, 42, milliseconds, film), -1);  // half a picture on
    EXPECT_EQ(picture_number_at(0, 42, milliseconds, film), -1);        // before the first
}

TEST(RangeSeek, GridNeedsEveryPicture) {
    const long long last = matroska_pts(151918, 0);
    EXPECT_TRUE(pictures_fill_grid(0, last, 151919, milliseconds, film));
    EXPECT_FALSE(pictures_fill_grid(0, last, 151918, milliseconds, film));  // one dropped
    EXPECT_FALSE(pictures_fill_grid(0, last, 170507, milliseconds, film));  // a stale tag's count
    EXPECT_FALSE(pictures_fill_grid(0, last, 0, milliseconds, film));
}

TEST(RangeSeek, SeeksOnlyLateEnoughAndEarlyOfTheRange) {
    EXPECT_FALSE(seek_is_worthwhile(719, film));
    EXPECT_TRUE(seek_is_worthwhile(14400, film));
    const double seconds = seek_seconds(14400, film);
    EXPECT_NEAR(seconds, 14400 * 1001.0 / 24000 - 2, 1e-9);
}

// FFmpeg keeps a seeked video's pictures by their timestamps: half a picture before
// each cut, from the file's first picture, whose timestamp need not be 0.
TEST(RangeSeek, FfmpegCutsHalfAPictureBeforeTheRange) {
    const RangeSeek seek{598.0, 42, milliseconds, Rational{25, 1}};
    FrameRange range;
    range.decode_start = 15000;
    range.decode_end = 15125;
    EXPECT_EQ(range_trim_filter(range), "trim=start_frame=15000:end_frame=15125");
    EXPECT_EQ(range_trim_filter(range, seek), "trim=start=600.022000:end=605.022000");
    range.decode_end.reset();
    EXPECT_EQ(range_trim_filter(range, seek), "trim=start=600.022000");
}

TEST(RangeSeek, FfmpegInputOptionsFollowTheVideosOwn) {
    const std::vector<std::string> own{"-export_side_data", "film_grain"};
    EXPECT_EQ(with_seek(own, std::nullopt), own);
    const std::vector<std::string> seeking{"-export_side_data", "film_grain", "-copyts", "-noaccurate_seek",
                                           "-ss",               "598.000000000"};
    EXPECT_EQ(with_seek(own, RangeSeek{598.0, 0, milliseconds, Rational{25, 1}}), seeking);
}

}  // namespace vml
