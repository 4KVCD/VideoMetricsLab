// The form that adds a resolution test: a width or a height typed gives the
// other, and the test the row will have.
#include <QSpinBox>

#include <gtest/gtest.h>

#include "resample_target_dialog.h"

namespace vml::app {
namespace {

TEST(ResampleTargetDialog, AWidthOrHeightTypedGivesTheOther) {
    ResampleTargetDialog dialog({3840, 2160}, "lanczos", nullptr);
    auto* width = dialog.findChild<QSpinBox*>("resample_width");
    auto* height = dialog.findChild<QSpinBox*>("resample_height");
    ASSERT_NE(width, nullptr);
    ASSERT_NE(height, nullptr);

    height->setValue(720);  // as Return, or leaving the box, enters it
    EXPECT_EQ(width->value(), 1280);
    EXPECT_EQ(dialog.target().label, "720p");
    EXPECT_EQ(dialog.target().scaler, "lanczos");

    width->setValue(1003);  // odd: the nearest even number, a half to even
    EXPECT_EQ(width->value(), 1004);
    EXPECT_EQ(height->value(), 564);
    EXPECT_EQ(dialog.target().width, 1004);
    EXPECT_EQ(dialog.target().label, "1004w");
}

// On a tall picture one step of the width moves the height by more than the
// arrows' 2, so the arrows must move the width.
TEST(ResampleTargetDialog, TheHeightsArrowsMoveItOnATallPicture) {
    ResampleTargetDialog dialog({1080, 1920}, "bicubic", nullptr);
    auto* height = dialog.findChild<QSpinBox*>("resample_height");
    ASSERT_NE(height, nullptr);
    height->setValue(1280);
    for (int step = 1; step <= 10; ++step) {
        const int shown = height->value();
        height->stepUp();
        EXPECT_GT(height->value(), shown) << "step " << step;
    }
}

}  // namespace
}  // namespace vml::app
