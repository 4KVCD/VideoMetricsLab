// The window's interfaces (theme.h): the settings read as an interface and
// an accent, the text an accent button takes, and VML Modern's style sheet.
// Applying one changes the whole application's style, which every later
// test in the process would inherit: that is checked in the window instead.
#include <QBoxLayout>
#include <QColor>
#include <QFrame>
#include <QGroupBox>
#include <QLabel>
#include <QString>

#include <gtest/gtest.h>

#include <memory>

#include "theme.h"

namespace vml::app {
namespace {

TEST(Theme, TheSettingsReadAsAnInterfaceAndAnAccent) {
    EXPECT_EQ(theme::interface_of("modern"), theme::Interface::modern);
    EXPECT_EQ(theme::interface_of("classic"), theme::Interface::classic);
    EXPECT_EQ(theme::interface_of(""), theme::Interface::classic);  // older settings, or anything else
    EXPECT_EQ(theme::interface_of("Modern"), theme::Interface::classic);

    EXPECT_EQ(theme::accent_of("#4f8fe8"), QColor("#4f8fe8"));
    EXPECT_FALSE(theme::accent_of(""));  // the system's
    EXPECT_FALSE(theme::accent_of("blue"));
    EXPECT_FALSE(theme::accent_of("#4f8fe"));
    EXPECT_FALSE(theme::accent_of("#zzzzzz"));
}

// In Classic, the interface these tests run in: a hint is italic, and a
// Classic-only template is the widget's sheet (VML Modern's are checked in
// vml_appearance_tests).
TEST(Theme, ClassicKeepsItalicHintsAndItsOwnSheets) {
    ASSERT_EQ(theme::current_interface(), theme::Interface::classic);
    QLabel hint;
    theme::style(&hint, "color: {muted}; font-style: {hint_style};");
    EXPECT_TRUE(hint.styleSheet().contains("font-style: italic;"));
    QLabel own;
    theme::style_classic(&own, "QLabel {{ color: {muted}; }}");
    EXPECT_TRUE(own.styleSheet().startsWith("QLabel { color: #"));
}

// Layout that is VML Modern's alone leaves Classic's as it was: a
// container without margins, Modern's widgets hidden and Classic's shown, a
// Classic box titled, a separator the gap Classic had (Modern's are checked
// in vml_appearance_tests).
TEST(Theme, ModernsLayoutLeavesClassicAsItWas) {
    ASSERT_EQ(theme::current_interface(), theme::Interface::classic);
    QBoxLayout* layout = nullptr;
    const std::unique_ptr<QWidget> card(theme::container("card", true, &layout));
    EXPECT_EQ(layout->contentsMargins(), QMargins());
    QWidget page;
    auto* column = new QVBoxLayout(&page);
    auto* modern = new QLabel;
    theme::modern_only(modern);
    auto* classic = new QLabel;
    theme::classic_only(classic);
    // Not shown before it has a parent: it would be shown as a window of its
    // own, made and unmade again as it is added to one (about 40 ms of the
    // main window's build).
    EXPECT_FALSE(classic->isVisible());
    column->addWidget(modern);
    column->addWidget(classic);
    page.show();
    EXPECT_FALSE(modern->isVisible());
    EXPECT_TRUE(classic->isVisible());
    QGroupBox box;
    theme::classic_box(&box, "Videos");
    EXPECT_EQ(box.title(), "Videos");
    const std::unique_ptr<QFrame> gap(theme::separator(16));
    EXPECT_EQ(gap->minimumWidth(), 16);
    EXPECT_EQ(gap->frameShape(), QFrame::NoFrame);
    const std::unique_ptr<QFrame> none(theme::separator(0));
    EXPECT_TRUE(none->isHidden());  // no gap in Classic: not there
}

TEST(Theme, TextOnAnAccentIsWhicheverReads) {
    EXPECT_EQ(theme::text_on(QColor("#c38fe8")), QColor("#141416"));  // a light lilac: dark text
    EXPECT_EQ(theme::text_on(QColor("#ffd54f")), QColor("#141416"));
    EXPECT_EQ(theme::text_on(QColor("#1a237e")), QColor("#ffffff"));  // a deep blue: white
    EXPECT_EQ(theme::text_on(QColor("#b3261e")), QColor("#ffffff"));
}

TEST(Theme, ModernsStyleSheetHasEveryColourFilledIn) {
    for (const bool dark : {false, true}) {
        const QString sheet = theme::modern_style_sheet(dark, QColor("#4f8fe8"));
        EXPECT_FALSE(sheet.contains('{' + QStringLiteral("card")));
        // Each rule's braces balance and no {name} is left: a name unfilled
        // would leave one brace more than its rule's.
        EXPECT_EQ(sheet.count('{'), sheet.count('}'));
        for (const char* name : {"{accent", "{card", "{border", "{text", "{raised", "{field", "{danger"}) {
            EXPECT_FALSE(sheet.contains(QString::fromLatin1(name))) << name << (dark ? " (dark)" : " (light)");
        }
        EXPECT_TRUE(sheet.contains("#4f8fe8"));  // the accent, on the primary buttons and the tabs
        EXPECT_TRUE(sheet.contains(theme::modern_colour("card", dark).name()));
        EXPECT_TRUE(sheet.contains("QPushButton[vmlRole=\"primary\"]"));
        EXPECT_TRUE(sheet.contains("QPushButton[vmlRole=\"danger\"]"));
    }
    // The text on the primary buttons follows the accent.
    EXPECT_TRUE(theme::modern_style_sheet(true, QColor("#1a237e")).contains("color: #ffffff; border: 1px solid #1a237e"));
    // Light and dark differ where Modern's colours do.
    EXPECT_NE(theme::modern_colour("window", true), theme::modern_colour("window", false));
}

}  // namespace
}  // namespace vml::app
