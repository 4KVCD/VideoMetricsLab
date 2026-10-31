// How the window names the two Vship builds (vship_names.h): "GPU" for the
// app's own, "GPU - Vship" for stock Vship, in the Options panel's choice, the
// mark of a cell whose score the other build made, and its tooltip's word.
#include <QComboBox>

#include <gtest/gtest.h>

#include "options_panel.h"
#include "vship_names.h"

namespace vml::app {
namespace {

TEST(VshipNames, TheWindowNamesTheBuildsGpuAndGpuVship) {
    EXPECT_EQ(vship_choice_label(VshipBuild::fast), QStringLiteral("GPU"));
    EXPECT_EQ(vship_choice_label(VshipBuild::stock), QStringLiteral("GPU - Vship"));
}

// Square brackets say which build made a score; parentheses already say where
// it was calculated ("(CPU)").
TEST(VshipNames, AScoreOfTheOtherBuildIsMarkedInBrackets) {
    EXPECT_EQ(vship_build_mark(VshipBuild::stock), QStringLiteral(" [Vship]"));
    EXPECT_EQ(vship_build_mark(VshipBuild::fast), QStringLiteral(" [GPU]"));
    EXPECT_EQ(other_vship_build_note(VshipBuild::stock, VshipBuild::fast),
              QStringLiteral("\n\nCalculated with GPU - Vship, not with this video's GPU; their scores differ "
                             "slightly, so the next run calculates it again with GPU."));
}

// The choice lists the builds in VshipBuild's order (what the panel stores).
TEST(VshipNames, TheOptionsChoiceShowsTheNames) {
    OptionsPanel panel({GpuVendor::nvidia});
    const QComboBox& combo = *panel.vship_build_combo;
    ASSERT_EQ(combo.count(), 2);
    EXPECT_EQ(combo.itemText(static_cast<int>(VshipBuild::fast)), QStringLiteral("GPU"));
    EXPECT_EQ(combo.itemText(static_cast<int>(VshipBuild::stock)), QStringLiteral("GPU - Vship"));
    EXPECT_FALSE(combo.toolTip().contains(QStringLiteral("Vship-fast")));
    EXPECT_FALSE(combo.toolTip().contains(QStringLiteral("5.1.2")));
}

}  // namespace
}  // namespace vml::app
