// A tab clicked focuses nothing on its page (widgets.h,
// keep_focus_on_tab_click): with a box focused on one tab, clicking another
// tab focused that tab's first box, so Video Compare's Zoom, Videos'
// reference or BD-rate's Name showed focused once any box had been clicked.
#include <QApplication>
#include <QLineEdit>
#include <QTabBar>
#include <QTest>
#include <QVBoxLayout>
#include <QWidget>

#include <gtest/gtest.h>

#include "widgets.h"

namespace vml::app {
namespace {

TEST(TabFocus, AClickedTabFocusesNothingOnItsPage) {
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* tabs = new QuietTabWidget;
    layout->addWidget(tabs);
    auto* first = new QLineEdit;
    auto* second = new QLineEdit;
    auto* third = new QLineEdit;
    tabs->addTab(first, "First");
    tabs->addTab(second, "Second");
    tabs->addTab(third, "Third");
    window.resize(400, 200);
    window.show();
    window.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&window));

    first->setFocus();
    ASSERT_TRUE(first->hasFocus());
    QTest::mouseClick(tabs->tabBar(), Qt::LeftButton, {}, tabs->tabBar()->tabRect(1).center());
    EXPECT_EQ(tabs->currentIndex(), 1);
    EXPECT_FALSE(second->hasFocus());
    EXPECT_EQ(QApplication::focusWidget(), nullptr);

    // From the keyboard the focus goes along, as Qt has it.
    second->setFocus();
    QTest::keyClick(second, Qt::Key_Tab, Qt::ControlModifier);
    EXPECT_EQ(tabs->currentIndex(), 2);
    EXPECT_TRUE(third->hasFocus());
}

}  // namespace
}  // namespace vml::app
