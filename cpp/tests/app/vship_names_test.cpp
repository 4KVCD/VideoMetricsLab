// How the window names the Vship builds (vship_names.h), in the Options panel's
// choice, the mark of a cell whose score the other build made, and its tooltip's
// word.
#include <QComboBox>

#include <gtest/gtest.h>

#include "options_panel.h"
#include "vship_names.h"

namespace vml::app {
namespace {

TEST(VshipNames, TheChoicesSayWhenToUseEachBuild) {
    EXPECT_EQ(vship_choice_label(VshipBuild::fast), QStringLiteral("GPU (fastest and most accurate)"));
    EXPECT_EQ(vship_choice_label(VshipBuild::stock), QStringLiteral("Vship Vulkan (preferred)"));
    EXPECT_EQ(vship_choice_label(VshipBuild::hip), QStringLiteral("Vship HIP (use only if Vulkan has problems)"));
    EXPECT_EQ(vship_choice_label(VshipBuild::cuda), QStringLiteral("Vship CUDA (use only if Vulkan has problems)"));
}

// Square brackets say which build made a score; parentheses already say where
// it was calculated ("(CPU)").
TEST(VshipNames, AScoreOfTheOtherBuildIsMarkedInBrackets) {
    EXPECT_EQ(vship_build_mark(VshipBuild::stock), QStringLiteral(" [Vship]"));
    EXPECT_EQ(vship_build_mark(VshipBuild::fast), QStringLiteral(" [GPU]"));
    EXPECT_EQ(other_vship_build_note(VshipBuild::stock, VshipBuild::fast),
              QStringLiteral("\n\nCalculated with Vship Vulkan, not with this video's GPU; their scores differ "
                             "slightly, so the next run calculates it again with GPU."));
}

// Every build a choice lists comes back as itself, whichever are bundled (the Mac has
// no HIP or CUDA), and the long names do not widen the panel.
TEST(VshipNames, TheChoiceGivesBackTheBuildItShows) {
    const std::vector<VshipBuild> all{VshipBuild::fast, VshipBuild::stock, VshipBuild::hip, VshipBuild::cuda};
    QComboBox combo;
    add_vship_choices(combo, all);
    ASSERT_EQ(combo.count(), 4);
    for (const VshipBuild build : all) {
        show_vship_choice(combo, build);
        EXPECT_EQ(shown_vship_choice(combo), build);
    }
    QComboBox short_one;
    short_one.addItem(QStringLiteral("Vship Vulkan (preferred)"));
    EXPECT_LE(combo.sizeHint().width(), short_one.sizeHint().width());

    QComboBox on_mac;
    add_vship_choices(on_mac, {VshipBuild::fast, VshipBuild::stock});
    show_vship_choice(on_mac, VshipBuild::hip);
    EXPECT_EQ(shown_vship_choice(on_mac), VshipBuild::fast);
}

TEST(VshipNames, TheOptionsTooltipNamesNoLibrary) {
    OptionsPanel panel({GpuVendor::nvidia});
    const QComboBox& combo = *panel.vship_build_combo;
    EXPECT_FALSE(combo.toolTip().contains(QStringLiteral("Vship-fast")));
    EXPECT_FALSE(combo.toolTip().contains(QStringLiteral("5.1.2")));
}

}  // namespace
}  // namespace vml::app
