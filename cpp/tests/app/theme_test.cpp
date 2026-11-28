// The window's interfaces (theme.h): the settings read as an interface and
// an accent, the text an accent button takes, and VML Modern's style sheet.
// Applying one changes the whole application's style, which every later
// test in the process would inherit: that is checked in the window instead.
#include <QApplication>
#include <QBoxLayout>
#include <QColor>
#include <QFont>
#include <QFrame>
#include <QGroupBox>
#include <QLabel>
#include <QRegularExpression>
#include <QSet>
#include <QString>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

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

// Issue #249: VML Modern's sizes follow the text size. The sheet's sizes of
// 3 pixels or more grow with it, rounded; its 1- and 2-pixel lines stay thin;
// digits that are not a size (a colour) are left alone.
TEST(Theme, TheSheetsSizesGrowWithTheText) {
    EXPECT_EQ(theme::scaled_sheet("QPushButton { padding: 4px 12px; border: 1px solid #a0a3px; "
                                  "border-bottom: 2px solid #123456; margin: 3px; min-height: 18px; }",
                                  1.5),
              "QPushButton { padding: 6px 18px; border: 1px solid #a0a3px; "
              "border-bottom: 2px solid #123456; margin: 5px; min-height: 27px; }");
    EXPECT_EQ(theme::scaled_sheet("QLabel { padding: 7px; }", 1.0), "QLabel { padding: 7px; }");
}

// The text size is the application font's against the system's own at 100 %,
// never less than 1; at twice the text, Modern's sheet (as apply_sheet sets
// it) and containers are twice the size.
TEST(Theme, TwiceTheTextIsTwiceModernsSizes) {
    const QFont before = QApplication::font();
    QFont large = before;
#ifdef __APPLE__
    large.setPointSizeF(26.0);
#else
    large.setPointSizeF(18.0);
#endif
    QApplication::setFont(large);
    EXPECT_DOUBLE_EQ(theme::text_scale(), 2.0);
    EXPECT_EQ(theme::scaled(14), 28);
    const QString sheet = theme::scaled_sheet(theme::modern_style_sheet(false, QColor("#4f8fe8")), theme::text_scale());
    EXPECT_TRUE(sheet.contains("border-radius: 20px; margin-top: 0; padding: 72px 24px 24px 24px;"))
        << sheet.left(200).toStdString();
    QFont small = before;
    small.setPointSizeF(6.0);
    QApplication::setFont(small);
    EXPECT_DOUBLE_EQ(theme::text_scale(), 1.0);
    QApplication::setFont(before);
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

// Windows' Night Sky contrast theme, each colour told apart from the others
// (its button face nudged off its background) so that a name drawn in the
// wrong one shows.
theme::SystemColours night_sky() {
    return {QColor("#000000"), QColor("#ffffff"), QColor("#a6a6a6"), QColor("#d6b4fd"),
            QColor("#2b2b2b"), QColor("#8080ff"), QColor("#010101"), QColor("#fefefe")};
}

// Every colour Modern names, its window's and its sheet's, has one of the
// contrast theme's: a colour added later without one fails here rather than
// keeping its own colour in a contrast theme.
TEST(Theme, EveryColourHasAContrastThemesColour) {
    const theme::SystemColours system = night_sky();
    const QSet<QRgb> allowed{system.window.rgb(),    system.window_text.rgb(),    system.gray_text.rgb(),
                             system.highlight.rgb(), system.highlight_text.rgb(), system.link.rgb(),
                             system.button.rgb(),    system.button_text.rgb()};
    for (const QString& name : theme::contrast_names()) {
        QColor colour;
        EXPECT_NO_THROW(colour = theme::contrast_colour(name.toUtf8().constData(), system)) << name.toStdString();
        EXPECT_TRUE(allowed.contains(colour.rgb())) << name.toStdString();
    }
    EXPECT_THROW(theme::contrast_colour("no_such_colour", system), std::out_of_range);
    const theme::ChartColours chart = theme::contrast_chart_colours(system);
    for (const QColor& colour : {chart.background, chart.text, chart.axis, chart.grid, chart.crosshair,
                                 chart.difference, chart.highlight}) {
        EXPECT_TRUE(allowed.contains(colour.rgb())) << colour.name().toStdString();
    }
}

// Modern's sheet in a contrast theme holds the theme's colours alone (its
// chevrons' too), and every {name} in it is filled.
TEST(Theme, TheContrastSheetHasTheSystemsColoursAlone) {
    const theme::SystemColours system = night_sky();
    const QSet<QString> allowed{system.window.name(),    system.window_text.name(),    system.gray_text.name(),
                                system.highlight.name(), system.highlight_text.name(), system.link.name(),
                                system.button.name(),    system.button_text.name()};
    const QString sheet = theme::contrast_style_sheet(system);
    const QRegularExpression colour(QStringLiteral(R"re(#([0-9a-f]{8}|[0-9a-f]{6})\b|chevron-[a-z]+-([0-9a-f]{6})|rgba?\()re"));
    int found = 0;
    for (const QRegularExpressionMatch& match : colour.globalMatch(sheet)) {
        ++found;
        const QString text = match.captured(0);
        ASSERT_FALSE(text.startsWith("rgb")) << "a tint: " << text.toStdString();
        const QString hex = match.captured(1).isEmpty() ? match.captured(2) : match.captured(1);
        EXPECT_TRUE(allowed.contains("#" + hex)) << text.toStdString();
    }
    EXPECT_GT(found, 50);
    const QRegularExpression unfilled(QStringLiteral(R"re(\{[a-z_]+\})re"));
    EXPECT_FALSE(unfilled.match(sheet).hasMatch()) << unfilled.match(sheet).captured(0).toStdString();
}

}  // namespace
}  // namespace vml::app
