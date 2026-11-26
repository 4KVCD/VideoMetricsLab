// A tab clicked focuses nothing on its page (widgets.h,
// keep_focus_on_tab_click): with a box focused on one tab, clicking another
// tab focused that tab's first box, so Video Compare's Zoom, Videos'
// reference or BD-rate's Name showed focused once any box had been clicked.
#include <QApplication>
#include <QLineEdit>
#include <QTabBar>
#include <QTest>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

#include <gtest/gtest.h>

#include <set>
#include <vector>

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

// A tab widget's tabs as pills that wrap onto rows (widgets.h, TabPills):
// Metric Graphs' metric tabs scrolled, a name cut at one end. Every pill
// whole and as wide checked as not, a click makes its tab current and
// focuses nothing, and the pills follow the bar, which stays hidden.
TEST(TabPills, TabsWrapWholeOntoRowsAndFollowTheBar) {
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* tabs = new QuietTabWidget;
    tabs->tabBar()->hide();
    auto* pills = new TabPills(tabs->tabBar());
    pills->setStyleSheet(QStringLiteral("QToolButton:checked { font-weight: bold; }"));
    layout->addWidget(pills);
    layout->addWidget(tabs);
    std::vector<QLineEdit*> pages;
    for (int tab = 0; tab < 12; ++tab) {
        pages.push_back(new QLineEdit);
        tabs->addTab(pages.back(), QStringLiteral("Metric %1").arg(tab));
    }
    window.resize(300, 300);
    window.show();
    window.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&window));
    QApplication::processEvents();  // the one rebuild for the 12 tabs added
    ASSERT_EQ(pills->pills().size(), 12U);
    EXPECT_FALSE(tabs->tabBar()->isVisible());
    std::set<int> rows;
    for (const QToolButton* pill : pills->pills()) {
        EXPECT_EQ(pill->width(), pill->sizeHint().width()) << pill->text().toStdString();
        EXPECT_TRUE(pills->rect().contains(pill->geometry())) << pill->text().toStdString();
        rows.insert(pill->y());
    }
    EXPECT_GT(rows.size(), 1U);

    QToolButton* sixth = pills->pills()[5];
    const int unchecked_width = sixth->sizeHint().width();
    pages[0]->setFocus();
    QTest::mouseClick(sixth, Qt::LeftButton);
    EXPECT_EQ(tabs->currentIndex(), 5);
    EXPECT_TRUE(sixth->isChecked());
    EXPECT_FALSE(pills->pills()[0]->isChecked());
    EXPECT_EQ(sixth->sizeHint().width(), unchecked_width);  // bold, it moves no pill
    EXPECT_EQ(QApplication::focusWidget(), nullptr);         // as a tab's click (keep_focus_on_tab_click)

    tabs->setCurrentIndex(2);
    EXPECT_TRUE(pills->pills()[2]->isChecked());
    QTest::keyClick(pills, Qt::Key_End);
    EXPECT_EQ(tabs->currentIndex(), 11);

    tabs->removeTab(0);
    QApplication::processEvents();
    ASSERT_EQ(pills->pills().size(), 11U);
    EXPECT_EQ(pills->pills()[0]->text(), QStringLiteral("Metric 1"));
    EXPECT_FALSE(tabs->tabBar()->isVisible());
}

}  // namespace
}  // namespace vml::app
