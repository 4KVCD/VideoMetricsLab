// The developers' tools (dev_tools.h): the Tab order they draw is the one
// Tab follows, and the window's state they write names what is shown.
#include <QApplication>
#include <QLineEdit>
#include <QPushButton>
#include <QTest>
#include <QVBoxLayout>
#include <QWidget>

#include <gtest/gtest.h>

#include <vector>

#include "dev_tools.h"

namespace vml::app {
namespace {

TEST(DevTools, TheTabOrderDrawnIsTheOneTabFollows) {
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* first = new QLineEdit;
    auto* hidden = new QLineEdit;
    auto* off = new QPushButton(QStringLiteral("Off"));
    auto* second = new QPushButton(QStringLiteral("Second"));
    auto* third = new QLineEdit;
    for (QWidget* widget : std::vector<QWidget*>{first, hidden, off, second, third}) {
        layout->addWidget(widget);
    }
    second->setFocusPolicy(Qt::StrongFocus);  // a push button takes Tab on Windows; on macOS only so
    hidden->hide();
    off->setEnabled(false);
    QWidget::setTabOrder(first, third);  // out of the layout's order
    QWidget::setTabOrder(third, second);
    window.show();
    window.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&window));
    const std::vector<QWidget*> stops = dev::tab_stops(&window);
    ASSERT_EQ(stops, (std::vector<QWidget*>{first, third, second}));
    first->setFocus(Qt::TabFocusReason);
    for (std::size_t index = 1; index < stops.size(); ++index) {
        QTest::keyClick(QApplication::focusWidget(), Qt::Key_Tab);
        EXPECT_EQ(QApplication::focusWidget(), stops[index]) << index;
    }
}

TEST(DevTools, TheStateWrittenNamesTheShownControlsAndTheFocus) {
    QWidget window;
    auto* layout = new QVBoxLayout(&window);
    auto* name = new QLineEdit(QStringLiteral("typed"));
    name->setObjectName(QStringLiteral("nameBox"));
    auto* hidden = new QPushButton(QStringLiteral("Hidden"));
    layout->addWidget(name);
    layout->addWidget(hidden);
    hidden->hide();
    window.show();
    window.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&window));
    name->setFocus();
    const nlohmann::json state = dev::describe_widget(&window, &window);
    ASSERT_TRUE(state.contains("children"));
    const nlohmann::json& children = state["children"];
    ASSERT_EQ(children.size(), 1U);  // the hidden button left out (the layout is no widget)
    EXPECT_EQ(children[0]["name"], "nameBox");
    EXPECT_EQ(children[0]["class"], "QLineEdit");
    EXPECT_EQ(children[0]["text"], "typed");
    EXPECT_EQ(children[0]["focus"], true);
    EXPECT_EQ(children[0]["geometry"][2], name->width());
}

}  // namespace
}  // namespace vml::app
