// The Options panel's Film grain dropdown for the selected videos
// (options_panel.h's film_grain_control): usable where a video's grain can be
// scored with or without, the choice they share, and why it cannot be used.
#include <QComboBox>
#include <QSignalSpy>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "options_panel.h"
#include "vml/film_grain_detect.h"

namespace vml::app {
namespace {

VideoInfo video(const char* codec, std::optional<unsigned> kinds, std::optional<bool> with = std::nullopt) {
    VideoInfo info;
    info.codec_name = codec;
    info.pix_fmt = "yuv420p10le";
    info.film_grain = kinds;
    info.with_film_grain = with;
    return info;
}

TEST(FilmGrainControl, OnlyGrainThatCanBeChosenIsAndTheReasonOtherwise) {
    const VideoInfo with = video("vvc", film_grain_h274);  // no choice yet: with
    const VideoInfo without = video("av1", film_grain_av1, false);
    const VideoInfo forced = video("hevc", film_grain_h274);  // H.274 outside VVC
    const VideoInfo none = video("vvc", 0u);
    const VideoInfo unread = video("vvc", std::nullopt);

    FilmGrainControl control = film_grain_control({{&with}});
    EXPECT_TRUE(control.enabled);
    EXPECT_EQ(control.with, true);
    control = film_grain_control({{&without}, {&none}});  // the video without grain is left as it is
    EXPECT_TRUE(control.enabled);
    EXPECT_EQ(control.with, false);
    control = film_grain_control({{&with}, {&without}});
    EXPECT_TRUE(control.enabled);
    EXPECT_EQ(control.with, std::nullopt);  // mixed

    for (const std::vector<FilmGrainVideo>& cannot :
         {std::vector<FilmGrainVideo>{{&forced}}, {{&none}}, {{&unread}}, {{nullptr}}, {{&with, true}}}) {
        control = film_grain_control(cannot);
        EXPECT_FALSE(control.enabled);
        EXPECT_EQ(control.with, std::nullopt);
        EXPECT_FALSE(control.tooltip.isEmpty());
    }
    EXPECT_TRUE(film_grain_control({{&forced}}).tooltip.contains("HEVC"));  // why: H.274 is VVC's only
    EXPECT_NE(film_grain_control({{&none}}).tooltip, film_grain_control({{&unread}}).tooltip);
}

// The dropdown shows the choice without reporting it, reports a choice the
// user makes, and shows "Mixed" (which chooses nothing) only while needed;
// disabled, it shows nothing.
TEST(FilmGrainControl, TheDropdownReportsOnlyTheUsersChoice) {
    OptionsPanel panel({GpuVendor::nvidia});
    QSignalSpy chosen(&panel, &OptionsPanel::film_grain_chosen);
    const VideoInfo with = video("vvc", film_grain_h274);
    const VideoInfo without = video("vvc", film_grain_h274, false);
    const VideoInfo none = video("vvc", 0u);
    QComboBox& combo = *panel.film_grain_combo;

    panel.show_film_grain({{&none}});
    EXPECT_FALSE(combo.isEnabled());
    EXPECT_EQ(combo.currentText(), QString());
    panel.show_film_grain({{&with}, {&without}});
    EXPECT_EQ(combo.currentText(), QStringLiteral("Mixed (the selected videos differ)"));

    panel.show_film_grain({{&without}});
    EXPECT_EQ(combo.currentData(), QVariant(false));
    EXPECT_EQ(combo.count(), 2);
    combo.setCurrentIndex(combo.findData(true));
    ASSERT_EQ(chosen.count(), 1);
    EXPECT_EQ(chosen.takeFirst().at(0).toBool(), true);

    panel.show_film_grain({{&with}, {&without}});
    EXPECT_EQ(combo.count(), 3);
    EXPECT_FALSE(combo.currentData().isValid());  // Mixed
    combo.setCurrentIndex(combo.findData(false));
    combo.setCurrentIndex(0);  // back to Mixed: nothing chosen
    EXPECT_EQ(chosen.count(), 1);
    panel.show_film_grain({{&with}});
    EXPECT_EQ(combo.count(), 2);
    EXPECT_EQ(combo.currentData(), QVariant(true));
    EXPECT_EQ(chosen.count(), 1);  // shown, not chosen
}

}  // namespace
}  // namespace vml::app
