// The window from the keyboard: Tab moves from control to control and out of
// every table, as on Windows.
#include <QAbstractItemView>
#include <QApplication>
#include <QHeaderView>
#include <QPushButton>
#include <QTableWidget>
#include <QTest>
#include <QVBoxLayout>

#include <gtest/gtest.h>

#include "main_window.h"

namespace vml::app {
namespace {

class KeyboardInWindow : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        window_ = new MainWindow;
        window_->show();
        QApplication::processEvents();
    }
    static void TearDownTestSuite() {
        delete window_;
        window_ = nullptr;
    }
    static MainWindow* window_;
};

MainWindow* KeyboardInWindow::window_ = nullptr;

// Qt's tables take Tab for themselves by default (tabKeyNavigation): it moves
// to the next cell, and from the last back to the first, so once a table had
// rows Tab never left it. Every table in the window lets Tab go on to the
// next control; its arrows move between its cells. A table added without it
// fails here.
TEST_F(KeyboardInWindow, TabLeavesEveryTable) {
    int tables = 0;
    for (QAbstractItemView* view : window_->findChildren<QAbstractItemView*>()) {
        if (qobject_cast<QHeaderView*>(view) != nullptr || view->focusPolicy() == Qt::NoFocus ||
            !view->inherits("QTableView")) {
            continue;
        }
        ++tables;
        EXPECT_FALSE(view->tabKeyNavigation())
            << view->metaObject()->className() << " " << view->objectName().toStdString() << " in "
            << (view->parentWidget() != nullptr ? view->parentWidget()->objectName().toStdString() : "");
    }
    EXPECT_GE(tables, 6);  // the window's tables were found at all
}

// What the property means: Tab from a table with rows reaches the next
// control.
TEST(Keyboard, TabFromATableWithRowsLeavesIt) {
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* table = new QTableWidget(2, 2);
    table->setTabKeyNavigation(false);
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            table->setItem(row, column, new QTableWidgetItem(QStringLiteral("x")));
        }
    }
    auto* after = new QPushButton(QStringLiteral("after"));
    layout->addWidget(table);
    layout->addWidget(after);
    window.show();
    window.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&window));
    table->setFocus(Qt::TabFocusReason);
    table->setCurrentCell(0, 0);
    QTest::keyClick(table, Qt::Key_Tab);
    EXPECT_EQ(QApplication::focusWidget(), after);
    QTest::keyClick(after, Qt::Key_Backtab, Qt::ShiftModifier);
    EXPECT_EQ(QApplication::focusWidget(), table);
    QTest::keyClick(table, Qt::Key_Right);
    EXPECT_EQ(table->currentColumn(), 1);  // the arrows still move between the cells
}

}  // namespace
}  // namespace vml::app
