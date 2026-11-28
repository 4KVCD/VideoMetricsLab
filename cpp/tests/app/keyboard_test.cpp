// The window from the keyboard: Tab moves from control to control and out of
// every table and time box, as on Windows; Enter presses a button the keys
// moved to; the main actions have shortcuts (shortcuts.h).
#include <QAbstractItemView>
#include <QApplication>
#include <QCheckBox>
#include <QHeaderView>
#include <QKeyEvent>
#include <QPushButton>
#include <QTabWidget>
#include <QTableWidget>
#include <QTest>
#include <QTimeEdit>
#include <QVBoxLayout>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "keyboard.h"
#include "main_window.h"
#include "options_panel.h"
#include "shortcuts.h"
#include "vml/options.h"

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

// The window's shortcuts work wherever the focus is in it: Ctrl+, opens
// Settings, Ctrl+3 the third tab, F1 the list of keys. A button's key is in
// its tooltip.
TEST_F(KeyboardInWindow, ShortcutsWorkAnywhereInTheWindow) {
    window_->activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(window_));
    auto* tabs = window_->findChild<QTabWidget*>();
    ASSERT_NE(tabs, nullptr);
    tabs->setCurrentIndex(0);
    QWidget* focus = QApplication::focusWidget();
    ASSERT_NE(focus, nullptr);
    QTest::keyClick(focus, Qt::Key_Comma, Qt::ControlModifier);
    EXPECT_EQ(tabs->currentIndex(), tabs->count() - 1);  // Settings, the last
    QTest::keyClick(QApplication::focusWidget(), Qt::Key_3, Qt::ControlModifier);
    EXPECT_EQ(tabs->currentIndex(), 2);
    QTest::keyClick(QApplication::focusWidget(), Qt::Key_F1);
    auto* list = window_->findChild<ShortcutsDialog*>();
    ASSERT_NE(list, nullptr);
    EXPECT_TRUE(list->isVisible());
    list->close();
    tabs->setCurrentIndex(0);
    for (QPushButton* button : window_->findChildren<QPushButton*>()) {
        if (button->text() == "Calculate metrics") {
            EXPECT_TRUE(button->toolTip().endsWith("Shortcut: F5")) << button->toolTip().toStdString();
        }
    }
}

// A shortcut whose button is grayed (Ctrl+O during a run) does nothing: it
// does not switch to the Videos tab either.
TEST_F(KeyboardInWindow, AGrayedButtonsShortcutLeavesTheTab) {
    window_->activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(window_));
    auto* tabs = window_->findChild<QTabWidget*>();
    ASSERT_NE(tabs, nullptr);
    QPushButton* add = nullptr;
    for (QPushButton* button : window_->findChildren<QPushButton*>()) {
        if (button->text() == "Add files") {
            add = button;
        }
    }
    ASSERT_NE(add, nullptr);
    add->setEnabled(false);
    tabs->setCurrentIndex(2);
    QTest::keyClick(QApplication::focusWidget(), Qt::Key_O, Qt::ControlModifier);
    EXPECT_EQ(tabs->currentIndex(), 2);
    add->setEnabled(true);
    tabs->setCurrentIndex(0);
}

// Settings' "Use the keyboard shortcuts" unticked: the shortcuts do nothing
// and the buttons' tooltips leave out their keys; ticked again, both are
// back.
TEST_F(KeyboardInWindow, TheShortcutsCanBeTurnedOff) {
    window_->activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(window_));
    auto* tabs = window_->findChild<QTabWidget*>();
    ASSERT_NE(tabs, nullptr);
    tabs->setCurrentIndex(tabs->count() - 1);  // Settings, built now
    QApplication::processEvents();
    // Offscreen, building Settings leaves the window inactive (not so on
    // screen, where the keys go on working): active again, as window
    // shortcuts need.
    window_->activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(window_));
    auto* use = window_->findChild<QCheckBox*>("useKeyboardShortcuts");
    ASSERT_NE(use, nullptr);
    QPushButton* run = nullptr;
    for (QPushButton* button : window_->findChildren<QPushButton*>()) {
        if (button->text() == "Calculate metrics") {
            run = button;
        }
    }
    ASSERT_NE(run, nullptr);
    use->setChecked(false);
    tabs->setCurrentIndex(0);
    tabs->tabBar()->setFocus();  // the box ticked is on a page now hidden
    QTest::keyClick(tabs->tabBar(), Qt::Key_3, Qt::ControlModifier);
    EXPECT_EQ(tabs->currentIndex(), 0);
    EXPECT_FALSE(run->toolTip().contains("Shortcut")) << run->toolTip().toStdString();
    use->setChecked(true);
    QTest::keyClick(tabs->tabBar(), Qt::Key_3, Qt::ControlModifier);
    EXPECT_EQ(tabs->currentIndex(), 2);
    EXPECT_TRUE(run->toolTip().endsWith("Shortcut: F5")) << run->toolTip().toStdString();
    tabs->setCurrentIndex(0);
}

// With the focus inside Settings, Ctrl+Tab switches the window's tabs, not
// Settings' sections; Ctrl+PgDn and Ctrl+PgUp step through the sections.
TEST_F(KeyboardInWindow, CtrlTabIsTheWindowsTabsEvenInsideSettings) {
    window_->activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(window_));
    auto* tabs = window_->findChild<QTabWidget*>();
    ASSERT_NE(tabs, nullptr);
    const int settings_tab = tabs->count() - 1;
    tabs->setCurrentIndex(settings_tab);
    QApplication::processEvents();
    window_->activateWindow();  // as in TheShortcutsCanBeTurnedOff
    ASSERT_TRUE(QTest::qWaitForWindowActive(window_));
    auto* sections = window_->findChild<QTabWidget*>("settingsSections");
    ASSERT_NE(sections, nullptr);
    sections->setCurrentIndex(0);
    auto* field = window_->findChild<QWidget*>("ffmpegFolder");  // a control inside Settings
    ASSERT_NE(field, nullptr);
    field->setFocus();
    QTest::keyClick(field, Qt::Key_PageDown, Qt::ControlModifier);
    EXPECT_EQ(sections->currentIndex(), 1);
    EXPECT_EQ(tabs->currentIndex(), settings_tab);
    QTest::keyClick(sections->currentWidget(), Qt::Key_PageUp, Qt::ControlModifier);
    EXPECT_EQ(sections->currentIndex(), 0);
    field->setFocus();
    QTest::keyClick(field, Qt::Key_Tab, Qt::ControlModifier);
    EXPECT_EQ(tabs->currentIndex(), 0);  // past the last tab: the first
    EXPECT_EQ(sections->currentIndex(), 0);
    tabs->setCurrentIndex(0);
}

// Settings' panel is built when first wanted, after the rest of the window
// (title bar buttons included): Tab from the tab bar still goes into it,
// not to the title bar, and on through it before leaving.
TEST_F(KeyboardInWindow, TabFromTheTabsGoesIntoSettings) {
    auto* tabs = window_->findChild<QTabWidget*>();
    ASSERT_NE(tabs, nullptr);
    tabs->setCurrentIndex(tabs->count() - 1);
    QApplication::processEvents();
    QWidget* settings = tabs->widget(tabs->count() - 1);
    // The Tab stops, as Qt's Tab goes through the window's focus chain
    // (visible, enabled, taking Tab): the first after the tab bar and its
    // own scroll arrows is Settings'.
    std::vector<QWidget*> stops;
    for (QWidget* widget = window_->nextInFocusChain(); widget != window_; widget = widget->nextInFocusChain()) {
        if ((widget->focusPolicy() & Qt::TabFocus) != 0 && widget->focusProxy() == nullptr && widget->isVisible() &&
            widget->isEnabled() && widget->window() == window_) {
            stops.push_back(widget);
        }
    }
    auto next = std::find(stops.begin(), stops.end(), tabs->tabBar());
    ASSERT_NE(next, stops.end());
    while (next != stops.end() && (*next == tabs->tabBar() || tabs->tabBar()->isAncestorOf(*next))) {
        ++next;
    }
    ASSERT_NE(next, stops.end());
    EXPECT_TRUE(settings->isAncestorOf(*next)) << (*next)->metaObject()->className() << " "
                                               << (*next)->objectName().toStdString();
    tabs->setCurrentIndex(0);
}

// No two of the window's keys are the same, and none is a key the text
// boxes, the tables or Video Compare already take for themselves.
TEST(Keyboard, TheShortcutsTakeNoKeyAlreadyTaken) {
    std::vector<QKeySequence> taken;
    for (const QKeySequence::StandardKey standard :
         {QKeySequence::Copy, QKeySequence::Cut, QKeySequence::Paste, QKeySequence::Undo, QKeySequence::Redo,
          QKeySequence::SelectAll, QKeySequence::Delete, QKeySequence::Find, QKeySequence::Save,
          QKeySequence::Close, QKeySequence::Quit, QKeySequence::Print, QKeySequence::NextChild,
          QKeySequence::PreviousChild}) {
        for (const QKeySequence& keys : QKeySequence::keyBindings(standard)) {
            taken.push_back(keys);
        }
    }
    for (const int key : {int(Qt::Key_S), int(Qt::Key_Space), int(Qt::Key_Left), int(Qt::Key_Right)}) {
        taken.emplace_back(key);  // Video Compare's
    }
    taken.emplace_back(Qt::CTRL | Qt::Key_Tab);
    taken.emplace_back(Qt::CTRL | Qt::SHIFT | Qt::Key_Tab);
    taken.emplace_back(Qt::CTRL | Qt::Key_PageDown);  // Settings' sections, Metric Graphs' metrics
    taken.emplace_back(Qt::CTRL | Qt::Key_PageUp);
    for (int index = 0; index < 8; ++index) {
        taken.push_back(tab_keys(index));
    }
    for (const Shortcut& shortcut : shortcuts()) {
        EXPECT_FALSE(shortcut.keys.isEmpty());
        EXPECT_EQ(std::count(taken.begin(), taken.end(), shortcut.keys), 0)
            << shortcut.keys.toString().toStdString();
        taken.push_back(shortcut.keys);
        EXPECT_FALSE(command_text(shortcut.command).isEmpty());
    }
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

// The Range's time boxes: one Tab leaves each (Qt's went through its four
// sections first); the arrows still move between the sections.
TEST(Keyboard, TabLeavesATimeBoxAtOnce) {
    OptionsPanel panel({});
    VmafOptions options;
    options.set_time_range(0, 1000);
    panel.show_options(options, {options});
    panel.show();
    panel.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&panel));
    QTimeEdit* start = panel.range_start_edit;
    ASSERT_TRUE(start->isVisible());
    start->setFocus(Qt::TabFocusReason);
    start->setCurrentSection(QDateTimeEdit::HourSection);
    QTest::keyClick(start, Qt::Key_Tab);
    EXPECT_NE(QApplication::focusWidget(), start);
    start->setFocus(Qt::TabFocusReason);
    start->setCurrentSection(QDateTimeEdit::HourSection);
    QTest::keyClick(start, Qt::Key_Right, Qt::NoModifier);
    QTest::keyClick(start, Qt::Key_Right, Qt::NoModifier);
    QTest::keyClick(start, Qt::Key_Right, Qt::NoModifier);
    EXPECT_EQ(start->currentSection(), QDateTimeEdit::MinuteSection);  // past "00:" to the minutes
}

// Enter presses a button the keys moved to, as Space does; after a click
// Enter is left alone (nothing shows where the focus is), and a dialog's
// buttons keep Qt's own Enter, its default button's.
TEST(Keyboard, EnterPressesAButtonTheKeysMovedTo) {
    keyboard::install();
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* first = new QPushButton(QStringLiteral("first"));
    auto* second = new QPushButton(QStringLiteral("second"));
    layout->addWidget(first);
    layout->addWidget(second);
    int pressed = 0;
    QObject::connect(second, &QPushButton::clicked, [&pressed] { ++pressed; });
    window.show();
    window.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&window));
    first->setFocus(Qt::OtherFocusReason);
    QTest::keyClick(first, Qt::Key_Tab);
    ASSERT_EQ(QApplication::focusWidget(), second);
    EXPECT_TRUE(keyboard::focus_visible());
    QTest::keyClick(second, Qt::Key_Return);
    EXPECT_EQ(pressed, 1);
    QTest::mouseClick(first, Qt::LeftButton);
    EXPECT_FALSE(keyboard::focus_visible());
    second->setFocus(Qt::OtherFocusReason);
    QTest::keyClick(second, Qt::Key_Return);
    QApplication::processEvents();
    EXPECT_EQ(pressed, 1);
}

// An arrow a panel takes for itself (Video Compare's switch the test
// video) moves no focus, and shows no ring: a control clicked before it
// keeps Space for the panel. An arrow that moves the focus shows it.
TEST(Keyboard, AnArrowShowsTheRingOnlyIfItMovedTheFocus) {
    keyboard::install();
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* clicked = new QPushButton(QStringLiteral("clicked"));
    auto* table = new QTableWidget(3, 1);
    layout->addWidget(clicked);
    layout->addWidget(table);
    struct EatsArrows : QObject {
        bool eventFilter(QObject* /*watched*/, QEvent* event) override {
            return event->type() == QEvent::KeyPress && static_cast<QKeyEvent*>(event)->key() == Qt::Key_Right;
        }
    } eats;
    clicked->installEventFilter(&eats);
    window.show();
    window.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&window));
    keyboard::set_focus_visible(false);
    clicked->setFocus(Qt::MouseFocusReason);
    QTest::keyClick(clicked, Qt::Key_Right);
    QApplication::processEvents();
    EXPECT_FALSE(keyboard::focus_visible());
    // A focus moved by an arrow: shown.
    QWidget::setTabOrder(clicked, table);
    QTest::keyClick(clicked, Qt::Key_Down);  // a push button's Down moves to the next control
    QApplication::processEvents();
    if (QApplication::focusWidget() != clicked) {
        EXPECT_TRUE(keyboard::focus_visible());
    }
    keyboard::set_focus_visible(false);
}

}  // namespace
}  // namespace vml::app
