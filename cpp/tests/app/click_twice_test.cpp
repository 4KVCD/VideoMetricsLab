// Remove all and Start over are clicked twice in place of a window asking
// (click_twice.h): the second click confirms, however soon (a double-click's), and
// arming another button or letting go starts again.
#include <gtest/gtest.h>

#include <QApplication>
#include <QPushButton>

#include "click_twice.h"

namespace vml::app {
namespace {

// However soon it comes: a second click within the double-click time was
// ignored, the button left armed until a third (the user, 2026-11-25).
TEST(ClickTwice, TheSecondClickConfirms) {
    ClickTwice guard(nullptr);
    QPushButton remove("Remove all");
    remove.setProperty("vmlRole", "quiet");
    int changes = 0;
    QObject::connect(&guard, &ClickTwice::changed, [&changes] { ++changes; });

    EXPECT_FALSE(guard.click(&remove, "Click Remove all again"));  // armed
    EXPECT_EQ(guard.armed(), &remove);
    EXPECT_EQ(guard.line(), "Click Remove all again");
    EXPECT_EQ(remove.property("vmlRole").toString(), "danger");

    EXPECT_TRUE(guard.click(&remove, "Click Remove all again"));
    EXPECT_EQ(guard.armed(), nullptr);
    EXPECT_TRUE(guard.line().isEmpty());
    EXPECT_EQ(remove.property("vmlRole").toString(), "quiet");  // its own role back
    EXPECT_EQ(changes, 2);                                       // armed, let go
}

TEST(ClickTwice, ArmingAnotherOrLettingGoStartsAgain) {
    ClickTwice guard(nullptr);
    QPushButton remove("Remove all");
    QPushButton start_over("Start over");

    guard.click(&remove, "remove");
    EXPECT_FALSE(guard.click(&start_over, "start over"));  // the other's first click arms it instead
    EXPECT_EQ(guard.armed(), &start_over);
    EXPECT_EQ(guard.line(), "start over");
    EXPECT_FALSE(guard.click(&remove, "remove"));  // armed again, not confirmed

    guard.disarm();  // another decision (a run starting, another button)
    EXPECT_FALSE(guard.click(&remove, "remove"));
}

}  // namespace
}  // namespace vml::app
