// Control+Tab and Control+Shift+Tab switch the window's tabs on the Mac
// (QuietTabWidget, widgets.cpp), as Ctrl+Tab does on Windows. On the Mac's
// keyboard the keys reach the window only as a ShortcutOverride (Qt's
// MetaModifier is the Control key): the tests send it to the window as the
// system does, and QTest's KeyPress after it.
#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QKeyEvent>
#include <QLineEdit>
#include <QTest>
#include <QVBoxLayout>
#include <QWidget>
#include <QWindow>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "widgets.h"

namespace vml::app {
namespace {

// A window of tabs, each page with a line edit; the page `nested` holds tabs
// of its own (two, each with a line edit).
class TabKeys : public ::testing::Test {
protected:
    void make(int pages, int nested = -1) {
        edits.clear();
        inner_edits.clear();
        inner = nullptr;
        window = std::make_unique<QWidget>();
        auto* layout = new QVBoxLayout(window.get());
        tabs = new QuietTabWidget;
        layout->addWidget(tabs);
        for (int page = 0; page < pages; ++page) {
            auto* widget = new QWidget;
            auto* page_layout = new QVBoxLayout(widget);
            auto* edit = new QLineEdit;
            page_layout->addWidget(edit);
            edits.push_back(edit);
            if (page == nested) {
                inner = new QuietTabWidget;
                for (int inner_page = 0; inner_page < 2; ++inner_page) {
                    auto* inner_edit = new QLineEdit;
                    inner_edits.push_back(inner_edit);
                    inner->addTab(inner_edit, QStringLiteral("inner %1").arg(inner_page));
                }
                page_layout->addWidget(inner);
            }
            tabs->addTab(widget, QStringLiteral("page %1").arg(page));
        }
        window->show();
        window->activateWindow();
        ASSERT_TRUE(QTest::qWaitForWindowActive(window.get()));
        edits[0]->setFocus();
    }

    // Control+Tab, or Control+Shift+Tab, as QTest presses it on the window.
    void control_tab(bool back = false) {
        QTest::keyClick(window->windowHandle(), back ? Qt::Key_Backtab : Qt::Key_Tab,
                        back ? Qt::MetaModifier | Qt::ShiftModifier : Qt::MetaModifier);
    }

    std::unique_ptr<QWidget> window;
    QuietTabWidget* tabs = nullptr;
    QuietTabWidget* inner = nullptr;
    std::vector<QLineEdit*> edits;
    std::vector<QLineEdit*> inner_edits;
};

// The override alone switches (no KeyPress comes on the Mac), and is taken.
TEST_F(TabKeys, ControlTabSwitchesAtTheOverride) {
    make(3);
    QKeyEvent override(QEvent::ShortcutOverride, Qt::Key_Tab, Qt::MetaModifier);
    QApplication::sendEvent(window->windowHandle(), &override);
    EXPECT_EQ(tabs->currentIndex(), 1);
    EXPECT_TRUE(override.isAccepted());
}

// One tab a press, forwards and back, round from the ends, past a disabled
// and a hidden tab.
TEST_F(TabKeys, ControlTabStepsThroughTheTabs) {
    make(4);
    control_tab();
    EXPECT_EQ(tabs->currentIndex(), 1);
    control_tab(true);
    EXPECT_EQ(tabs->currentIndex(), 0);
    control_tab(true);
    EXPECT_EQ(tabs->currentIndex(), 3);
    control_tab();
    EXPECT_EQ(tabs->currentIndex(), 0);
    tabs->setTabEnabled(1, false);
    tabs->setTabVisible(2, false);
    control_tab();
    EXPECT_EQ(tabs->currentIndex(), 3);
    control_tab(true);
    EXPECT_EQ(tabs->currentIndex(), 0);
}

// With the focus in tabs within a tab, those switch; with the focus in none,
// the window's outermost.
TEST_F(TabKeys, TheInnermostTabsHoldingTheFocusSwitch) {
    make(2, 1);
    tabs->setCurrentIndex(1);
    inner_edits[0]->setFocus();
    control_tab();
    EXPECT_EQ(inner->currentIndex(), 1);
    EXPECT_EQ(tabs->currentIndex(), 1);
    QApplication::focusWidget()->clearFocus();
    ASSERT_EQ(QApplication::focusWidget(), nullptr);
    control_tab();
    EXPECT_EQ(tabs->currentIndex(), 0);
    EXPECT_EQ(inner->currentIndex(), 1);
}

// Tabs with no other tab to go to let the key through to the tabs around
// them; a dialog
// opened from a tab's page does not switch the window's tabs behind it.
TEST_F(TabKeys, OnlyTabsInTheFocusedWindowWithTabsToGoToSwitch) {
    make(2, 1);
    tabs->setCurrentIndex(1);
    inner->removeTab(1);
    inner_edits[0]->setFocus();
    control_tab();
    EXPECT_EQ(tabs->currentIndex(), 0);
    // Nor do tabs whose other tabs are all disabled or hidden.
    make(2, 1);
    tabs->setCurrentIndex(1);
    inner->setTabEnabled(1, false);
    inner_edits[0]->setFocus();
    control_tab();
    EXPECT_EQ(tabs->currentIndex(), 0);
    EXPECT_EQ(inner->currentIndex(), 0);

    QDialog dialog(edits[0]);
    auto* field = new QLineEdit(&dialog);
    dialog.show();
    dialog.windowHandle()->requestActivate();
    ASSERT_TRUE(QTest::qWaitFor([&dialog] { return QApplication::activeWindow() == &dialog; }));
    field->setFocus();
    const int behind = tabs->currentIndex();
    QTest::keyClick(dialog.windowHandle(), Qt::Key_Tab, Qt::MetaModifier);
    EXPECT_EQ(tabs->currentIndex(), behind);
}

// Tab alone, Option+Control+Tab, and Control+Tab while a popup is open, are
// not the tabs'.
TEST_F(TabKeys, OtherKeysAreNotTaken) {
    make(3);
    QTest::keyClick(window->windowHandle(), Qt::Key_Tab);
    EXPECT_EQ(tabs->currentIndex(), 0);
    QTest::keyClick(window->windowHandle(), Qt::Key_Tab, Qt::MetaModifier | Qt::AltModifier);
    EXPECT_EQ(tabs->currentIndex(), 0);
    auto* combo = new QComboBox;
    combo->addItems({QStringLiteral("a"), QStringLiteral("b")});
    edits[0]->parentWidget()->layout()->addWidget(combo);
    combo->showPopup();
    ASSERT_NE(QApplication::activePopupWidget(), nullptr);
    QKeyEvent override(QEvent::ShortcutOverride, Qt::Key_Tab, Qt::MetaModifier);
    QApplication::sendEvent(window->windowHandle(), &override);
    EXPECT_EQ(tabs->currentIndex(), 0);
    combo->hidePopup();
}

}  // namespace
}  // namespace vml::app
