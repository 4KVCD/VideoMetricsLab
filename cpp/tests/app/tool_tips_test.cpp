// Long tooltips wrap (WrapLongToolTips): Qt shows a plain-text tooltip as
// one line however long, and wraps rich text.
#include <QLabel>
#include <QTextDocument>

#include <gtest/gtest.h>

#include <memory>

#include "widgets.h"

namespace vml::app {
namespace {

// A long plain tooltip becomes rich text, with its words and line breaks
// (and text such as "<video>" shown as typed); a short one, and one already
// rich text, stay as they were. With the filter installed, a widget's is
// wrapped as it is set.
TEST(ToolTips, LongPlainTooltipsWrap) {
    const QString long_tip =
        "After each video is scored, saves its weak segments by the rules above as JSON and CSV, named after the "
        "video (<video>.vmaf.segments.json).\nExisting files of the same names are replaced.";
    const QString wrapped = wrapped_tool_tip(long_tip);
    EXPECT_TRUE(Qt::mightBeRichText(wrapped));
    QTextDocument shown;
    shown.setHtml(wrapped);
    EXPECT_EQ(shown.toPlainText(), long_tip);
    EXPECT_EQ(wrapped_tool_tip("Open the log folder"), "Open the log folder");
    const QString rich = "<p>Already <b>rich</b> text, which Qt wraps, and which is long enough to wrap on its own</p>";
    EXPECT_EQ(wrapped_tool_tip(rich), rich);
    EXPECT_EQ(wrapped_tool_tip(QString()), QString());

    const auto filter = std::make_unique<WrapLongToolTips>(nullptr);
    QLabel label;
    label.setToolTip(long_tip);
    EXPECT_EQ(label.toolTip(), wrapped);
    label.setToolTip("Short");
    EXPECT_EQ(label.toolTip(), "Short");
}

}  // namespace
}  // namespace vml::app
