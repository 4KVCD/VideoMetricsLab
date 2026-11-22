// Test videos' names as the window shows them (video_names.h): encodes of
// one source with long names told apart by what differs, or by their ends.
#include <gtest/gtest.h>

#include "video_names.h"
#include "vml/settings.h"

namespace vml::app {
namespace {

using Names = std::vector<QString>;

TEST(VideoNames, ShowsWhatTellsTheVideosApart) {
    const VideoNameRule differs;
    const QString show = QStringLiteral("Show Name - S01E03 - A Long Title - 1080p WEB-DL x265 10-bit CRF ");
    // The shared start left out at a word's edge, with the word before the
    // difference kept.
    EXPECT_EQ(shown_video_names({show + "20", show + "22", show + "24"}, differs),
              (Names{QStringLiteral("…CRF 20"), QStringLiteral("…CRF 22"), QStringLiteral("…CRF 24")}));
    // A shared end too.
    EXPECT_EQ(shown_video_names({QStringLiteral("Film 2160p x265 CRF 18 grain-synth v2"),
                                 QStringLiteral("Film 2160p x265 CRF 22 grain-synth v2")},
                                differs),
              (Names{QStringLiteral("…CRF 18…"), QStringLiteral("…CRF 22…")}));
    // One name all the others start with.
    EXPECT_EQ(shown_video_names({show + "20", show + "20 (2)"}, differs),
              (Names{QStringLiteral("…CRF 20"), QStringLiteral("…CRF 20 (2)")}));
    // Whole: one video; short names (too little to leave out); the same name twice.
    EXPECT_EQ(shown_video_names({show + "20"}, differs), Names{show + "20"});
    EXPECT_EQ(shown_video_names({QStringLiteral("crf20"), QStringLiteral("crf22")}, differs),
              (Names{QStringLiteral("crf20"), QStringLiteral("crf22")}));
    EXPECT_EQ(shown_video_names({show + "20", show + "20"}, differs), (Names{show + "20", show + "20"}));
}

TEST(VideoNames, KeepsTheEndsChosen) {
    Settings settings;
    settings.video_names = "ends";
    settings.video_name_start = 5;
    settings.video_name_end = 6;
    const VideoNameRule rule = video_name_rule(settings);
    EXPECT_EQ(shown_video_names({QStringLiteral("Example Show x265 CRF 20"), QStringLiteral("Short name")}, rule),
              (Names{QStringLiteral("Examp…CRF 20"), QStringLiteral("Short name")}));
    // A character outside the BMP is not cut in two.
    settings.video_name_start = 1;
    settings.video_name_end = 1;
    EXPECT_EQ(shown_video_names({QStringLiteral("\U0001F3A5abcdef\U0001F3AC")}, video_name_rule(settings)),
              Names{QStringLiteral("\U0001F3A5…\U0001F3AC")});
    // Both 0, and an unknown kind.
    settings.video_name_start = 0;
    settings.video_name_end = 0;
    EXPECT_EQ(video_name_rule(settings).kind, VideoNameRule::Kind::full);
    settings.video_names = "other";
    EXPECT_EQ(video_name_rule(settings).kind, VideoNameRule::Kind::differs);
}

}  // namespace
}  // namespace vml::app
